# import needed library
import os
import random
import warnings

import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp

from utils import net_builder
from datasets.ssl_dataset import SSLDataset
from datasets.data_utils import get_data_loader
import pickle as pkl

torch.set_num_threads(1)


def main(args):
    '''
    For (Distributed)DataParallelism,
    main(args) spawn each process (main_worker) to each GPU.
    '''

    save_path = os.path.join(args.save_dir, args.save_name)
    if os.path.exists(save_path) and args.overwrite and args.resume == False:
        import shutil
        shutil.rmtree(save_path)
    if args.resume:
        if args.load_path is None:
            raise Exception(
                'Resume of training requires --load_path in the args')
        if os.path.abspath(save_path) == os.path.abspath(args.load_path) and not args.overwrite:
            raise Exception('Saving & Loading pathes are same. \
                            If you want over-write, give --overwrite in the argument.')

    if args.seed is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    # distributed: true if manually selected or if world_size > 1
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()  # number of gpus of each node

    if args.multiprocessing_distributed:
        # now, args.world_size means num of total processes in all nodes
        args.world_size = ngpus_per_node * args.world_size

        # args=(,) means the arguments of main_worker
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def get_dataloader_list(args, dataset_list):
    data_loader_list = []
    for dataset in dataset_list:
        data_loader = get_data_loader(dataset,
                                      args.eval_batch_size,
                                      num_workers=args.num_workers,
                                      drop_last=False)
        data_loader_list.append(data_loader)
    return data_loader_list


def load_model(args, mode="target"):
    args.bn_momentum = 1.0 - 0.999
    args.bn_momentum = 1.0 - 0.999
    if 'imagenet' in args.dataset.lower():
        _net_builder = net_builder('ResNet50', False, None, is_remix=False)
    else:
        _net_builder = net_builder(args.net,
                                   args.net_from_name,
                                   {'first_stride': 2 if 'stl' in args.dataset else 1,
                                    'depth': args.depth,
                                    'widen_factor': args.widen_factor,
                                    'leaky_slope': args.leaky_slope,
                                    'bn_momentum': args.bn_momentum,
                                    'dropRate': args.dropout,
                                    'use_embed': False,
                                    'is_remix': False},
                                   )

    model_path = os.path.join(args.save_dir, args.save_name)
    target_model_name = "target_model_%s.pth" % (args.target_epoch)
    shadow_model_name = "shadow_model_%s.pth" % (args.target_epoch)
    if mode == "target":
        model_name = target_model_name
    elif mode == "shadow":
        model_name = shadow_model_name
    else:
        raise ValueError("mode should be either target or shadow, ty")

    total_path = os.path.join(model_path, model_name)
    print(model_path)
    print(model_name)
    print("load model from: ", total_path)
    checkpoint = torch.load(total_path, map_location="cpu")
    model = _net_builder(num_classes=args.num_classes)

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint["model"].items():
        new_k = k.replace("module.", "")
        new_state_dict[new_k] = v

    model.load_state_dict(new_state_dict)
    model.cuda()
    return model


def main_worker(gpu, ngpus_per_node, args):
    '''
    main_worker is conducted on each GPU.
    '''

    global best_acc1
    args.gpu = gpu

    # random seed has to be set for the syncronization of labeled data sampling in each process.
    assert args.seed is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True

    # Construct Dataset & DataLoader
    s = SSLDataset(args, alg='attack', name=args.dataset,
                   num_classes=args.num_classes, data_dir=args.data_dir)
    dataset_list = s.get_data()
    transform_dataset_list = s.transform_dataset_list(dataset_list)
    target_train_ulb, target_train_lb, target_test, shadow_train_ulb, shadow_train_lb, shadow_test = transform_dataset_list

    loader_list = get_dataloader_list(args, transform_dataset_list)
    loader_dict = {
        'target_train_ulb': loader_list[0],
        'target_train_lb': loader_list[1],
        'target_test': loader_list[2],
        'shadow_train_ulb': loader_list[3],
        'shadow_train_lb': loader_list[4],
        'shadow_test': loader_list[5],
    }

    target_model = load_model(args, mode="target")
    shadow_model = load_model(args, mode="shadow")

    res_target_train_ulb = query_model(
        target_model, loader_dict['target_train_ulb'])
    res_target_train_lb = query_model(
        target_model, loader_dict['target_train_lb'])
    res_target_test = query_model(target_model, loader_dict['target_test'])
    res_shadow_train_ulb = query_model(
        shadow_model, loader_dict['shadow_train_ulb'])
    res_shadow_train_lb = query_model(
        shadow_model, loader_dict['shadow_train_lb'])
    res_shadow_test = query_model(shadow_model, loader_dict['shadow_test'])

    res = {
        'target_train_ulb': res_target_train_ulb,
        'target_train_lb': res_target_train_lb,
        'target_test': res_target_test,
        'shadow_train_ulb': res_shadow_train_ulb,
        'shadow_train_lb': res_shadow_train_lb,
        'shadow_test': res_shadow_test,
    }
    model_path = os.path.join(args.save_dir, args.save_name)
    if args.adjust_augmentation == "no":
        with open(os.path.join(model_path, "query_results_%s.pkl" % (args.target_epoch)), "wb") as wf:
            pkl.dump(res, wf)
    else:
        with open(os.path.join(model_path, "query_results_%s_adjust_augmentation_%d_%d.pkl" % (args.target_epoch, args.randaug_n, args.randaug_m)), "wb") as wf:
            pkl.dump(res, wf)


@torch.no_grad()
def query_model(model, dataloader):
    model.eval()
    res = {}
    cnt = 0
    for index, (x_weak_list, x_strong_list, x), y in dataloader:
        cnt += 1
        print(index)
        outputs_weak = []
        outputs_strong = []
        index = index.cpu().numpy()
        x_weak_list = [_.cuda(args.gpu) for _ in x_weak_list]
        x_strong_list = [_.cuda(args.gpu) for _ in x_strong_list]
        x, y = x.cuda(args.gpu), y.cuda(args.gpu)
        logits = model(x)
        outputs_original = torch.softmax(logits, dim=-1).cpu().numpy()
        y = y.cpu().numpy()

        for i in range(len(x_weak_list)):
            outputs_weak.append(torch.softmax(
                model(x_weak_list[i]), dim=-1).cpu().numpy())
            outputs_strong.append(torch.softmax(
                model(x_strong_list[i]), dim=-1).cpu().numpy())

        # augmented_num, batch_size, n_class (10, 1024, 10)
        outputs_weak = np.array(outputs_weak)
        outputs_strong = np.array(outputs_strong)
        # batch_size, augmented_num, n_class (1024, 10, 10)
        outputs_weak = np.swapaxes(outputs_weak, 0, 1)
        outputs_strong = np.swapaxes(outputs_strong, 0, 1)

        for j in range(len(index)):
            res[index[j]] = {"original": outputs_original[j],
                             "weak": outputs_weak[j],
                             "strong": outputs_strong[j],
                             "label": y[j]}
    return res


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')

    '''
    Saving & loading of the model.
    '''
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    # parser.add_argument('-sn', '--save_name', type=str, default='fixmatch')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('-o', '--overwrite', action='store_true')
    parser.add_argument('--use_tensorboard', action='store_true',
                        help='Use tensorboard to plot and save curves, otherwise save the curves locally.')

    '''
    Training Configuration of different ssl methods (fullysupervised, uda, fixmatch, flexmatch)
    '''

    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--num_train_iter', type=int, default=2 ** 20,
                        help='total number of training iterations')
    parser.add_argument('--num_eval_iter', type=int, default=5000,
                        help='evaluation frequency')
    parser.add_argument('-nl', '--num_labels', type=int, default=500)
    parser.add_argument('-bsz', '--batch_size', type=int, default=64)
    parser.add_argument('--uratio', type=int, default=7,
                        help='the ratio of unlabeled data to labeld data in each mini-batch')
    parser.add_argument('--eval_batch_size', type=int, default=1024,
                        help='batch size of evaluation data loader (it does not affect the accuracy)')

    parser.add_argument('--hard_label', type=str2bool, default=True)
    parser.add_argument('--T', type=float, default=0.5)
    parser.add_argument('--p_cutoff', type=float, default=0.95)
    parser.add_argument('--ema_m', type=float, default=0.999,
                        help='ema momentum for eval_model')
    parser.add_argument('--ulb_loss_ratio', type=float, default=1.0)

    '''
    Optimizer configurations
    '''
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=3e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--amp', type=str2bool, default=False,
                        help='use mixed precision training or not')
    parser.add_argument('--clip', type=float, default=0)
    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='WideResNet')
    parser.add_argument('--net_from_name', type=str2bool, default=False)
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--widen_factor', type=int, default=2)
    parser.add_argument('--leaky_slope', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.0)

    '''
    Data Configurations
    '''

    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('-ds', '--dataset', type=str, default='cifar10')
    parser.add_argument('--train_sampler', type=str, default='RandomSampler')
    parser.add_argument('-nc', '--num_classes', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=5)

    '''
    multi-GPUs & Distrbitued Training
    '''

    # args for distributed training (from https://github.com/pytorch/examples/blob/master/imagenet/main.py)
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='**node rank** for distributed training')
    parser.add_argument('-du', '--dist-url', default='tcp://127.0.0.1:22222', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', type=str2bool, default=False,
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    # attack related params
    # parser.add_argument('--attack_type', type=str, default='normal', help="normal or augmented ")
    parser.add_argument('--ssl_method', type=str, default="fixmatch")
    parser.add_argument('--augmented_num', default=10, type=int,
                        help='how many queries with different augmentations, e.g., 10 means generate 10 weak view and 10 augmented views to query the target model')
    parser.add_argument('--target_epoch', default=100, type=int,
                        help='seed for initializing training. ')

    # augmentation
    parser.add_argument('--adjust_augmentation', type=str,
                        default="no", help="no or yes")
    parser.add_argument('--randaug_n', type=int, default=2,
                        help="number of augmentation")
    parser.add_argument('--randaug_m', type=int, default=10,
                        help="number of augmentation")
    args = parser.parse_args()

    args.save_name = "%s_%s_%s_0" % (
        args.ssl_method, args.dataset, args.num_labels)
    main(args)
    print("Finish")
