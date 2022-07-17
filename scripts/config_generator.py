"""
Create the .yaml for each experiment
"""
import os


def create_configuration(cfg, cfg_file):
    cfg['save_name'] = "{alg}_{dataset}_{num_lb}_{seed}".format(
        alg=cfg['alg'],
        dataset=cfg['dataset'],
        num_lb=cfg['num_labels'],
        seed=cfg['seed'],
    )

    alg_file = cfg_file + cfg['alg'] + '/'
    if not os.path.exists(alg_file):
        os.mkdir(alg_file)

    print(alg_file + cfg['save_name'] + '.yaml')
    with open(alg_file + cfg['save_name'] + '.yaml', 'w', encoding='utf-8') as w:
        lines = []
        for k, v in cfg.items():
            line = str(k) + ': ' + str(v)
            lines.append(line)
        for line in lines:
            w.writelines(line)
            w.write('\n')


def create_base_config(alg, seed,
                       dataset, net, num_classes,
                       num_labels,
                       port,
                       weight_decay,
                       depth, widen_factor,
                       num_train_sample
                       ):
    cfg = {}

    # save config
    cfg['save_dir'] = './saved_models'
    cfg['save_name'] = None
    cfg['resume'] = False
    cfg['load_path'] = None
    cfg['overwrite'] = True
    cfg['use_tensorboard'] = True

    # algorithm config
    cfg['epoch'] = 1
    cfg['log_every_epoch'] = 5
    cfg['save_every_epoch'] = 5
    cfg['num_train_sample'] = num_train_sample
    # 1 is only used for saving the target model of 100 epochs
    cfg['num_train_iter'] = (2 ** 10) * 100 + 1
    cfg['num_eval_iter'] = 5000
    cfg['num_labels'] = num_labels
    cfg['batch_size'] = 64
    cfg['eval_batch_size'] = 1024
    if alg == 'fixmatch':
        cfg['hard_label'] = True
        cfg['T'] = 0.5
        cfg['p_cutoff'] = 0.95
        cfg['ulb_loss_ratio'] = 1.0
        cfg['uratio'] = 7
    elif alg == 'flexmatch':
        cfg['hard_label'] = True
        cfg['T'] = 0.5
        cfg['p_cutoff'] = 0.95
        cfg['ulb_loss_ratio'] = 1.0
        cfg['uratio'] = 7
    elif alg == 'uda':
        cfg['TSA_schedule'] = 'none'
        cfg['T'] = 0.4
        cfg['p_cutoff'] = 0.8
        cfg['ulb_loss_ratio'] = 1.0
        cfg['uratio'] = 7
    elif alg == 'pseudolabel':
        cfg['ulb_loss_ratio'] = 1.0
        cfg['uratio'] = 1
    elif alg == 'mixmatch':
        cfg['uratio'] = 1
        cfg['alpha'] = 0.5
        cfg['T'] = 0.5
        cfg['ulb_loss_ratio'] = 100
        cfg['ramp_up'] = 0.4
    elif alg == 'remixmatch':
        cfg['alpha'] = 0.75
        cfg['T'] = 0.5
        cfg['ulb_loss_ratio'] = 1.0
        cfg['w_kl'] = 0.5
        cfg['w_match'] = 1.5
        cfg['w_rot'] = 0.5
        cfg['use_dm'] = True
        cfg['use_xe'] = True
        cfg['warm_up'] = 1 / 64
        cfg['uratio'] = 1
    elif alg == 'meanteacher':
        cfg['uratio'] = 1
        cfg['ulb_loss_ratio'] = 50
        cfg['unsup_warm_up'] = 0.4
    elif alg == 'pimodel':
        cfg['ulb_loss_ratio'] = 10
        cfg['uratio'] = 1

    cfg['ema_m'] = 0.999

    # optim config
    cfg['optim'] = 'SGD'
    cfg['lr'] = 0.03
    cfg['momentum'] = 0.9
    cfg['weight_decay'] = weight_decay
    cfg['amp'] = False

    # net config
    cfg['net'] = net
    cfg['net_from_name'] = False
    cfg['depth'] = depth
    cfg['widen_factor'] = widen_factor
    cfg['leaky_slope'] = 0.1
    cfg['dropout'] = 0.0

    # data config
    # cfg['data_dir'] = '/p/project/hai_clsec/dataset'
    cfg['data_dir'] = './data'
    cfg['dataset'] = dataset
    cfg['train_sampler'] = 'RandomSampler'
    cfg['num_classes'] = num_classes
    cfg['num_workers'] = 5

    # basic config
    cfg['alg'] = alg
    cfg['seed'] = seed

    # distributed config
    cfg['world_size'] = 1
    cfg['rank'] = 0
    cfg['multiprocessing_distributed'] = False
    cfg['dist_url'] = 'tcp://127.0.0.1:' + str(port)
    cfg['dist_backend'] = 'nccl'
    cfg['gpu'] = 0

    # other config
    cfg['overwrite'] = True
    cfg['amp'] = False
    # cfg["mode"]=mode

    # # to follow fixmatch settings
    # if dataset == "imagenet":
    #     cfg['batch_size'] = 1024
    #     cfg['uratio'] = 5
    #     cfg['ulb_loss_ratio'] = 10
    #     cfg['p_cutoff'] = 0.7
    #     cfg['lr'] = 0.1
    #     cfg['num_train_iter'] = 12000000

    if dataset == "imagenet":
        cfg['batch_size'] = 32
        cfg['eval_batch_size'] = 256
        cfg['lr'] = 0.03
        cfg['num_train_iter'] = 2000000
        cfg['num_eval_iter'] = 10000

    return cfg


# prepare the configuration for baseline model, use_penalty == False
def exp_baseline(label_amount):
    config_file = r'../config/'
    save_path = r'../saved_models/'

    if not os.path.exists(config_file):
        os.mkdir(config_file)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    algs = ['flexmatch', 'fixmatch', 'uda', 'fullysupervised']
    datasets = ['cifar10', 'cifar100', 'svhn']
    dataset_train_sample = [60000//4, 60000//4, 99289//4]
    label_list = [500, 1000, 2000, 4000]
    seeds = [0]  # 1, 22, 333

    dist_port = range(10001, 11120, 1)
    count = 0
    for label in label_list:
        for alg in algs:
            for dataset in datasets:
                for seed in seeds:
                    # change the configuration of each dataset
                    if dataset == 'cifar10':
                        net = 'WideResNet'
                        num_classes = 10
                        num_labels = label
                        weight_decay = 5e-4
                        depth = 28
                        widen_factor = 2
                        train_sample = 15000
                    elif dataset == 'cifar100':
                        net = 'WideResNet'
                        num_classes = 100
                        num_labels = label
                        weight_decay = 1e-3
                        depth = 28
                        widen_factor = 2
                        train_sample = 15000
                    elif dataset == 'svhn':
                        net = 'WideResNet'
                        num_classes = 10
                        num_labels = label
                        weight_decay = 5e-4
                        depth = 28
                        widen_factor = 2
                        train_sample = 99289//4

                    port = dist_port[count]
                    # prepare the configuration file
                    cfg = create_base_config(alg, seed,
                                             dataset, net, num_classes, num_labels,
                                             port,
                                             weight_decay, depth, widen_factor, train_sample)
                    count += 1
                    create_configuration(cfg, config_file)


def flexmatch_config_generation():
    config_file = r'../data_statistics/'

    if not os.path.exists(config_file):
        os.mkdir(config_file)

    import json

    datasets = ['cifar10', 'cifar100', "svhn"]
    datasets_class_dict = {'cifar10': 10, 'cifar100': 100, "svhn": 10}
    label_list = [500, 1000, 2000, 4000]
    mode_list = ["target", "shadow"]

    for dataset in datasets:
        for label in label_list:
            for mode in mode_list:
                n_class = datasets_class_dict[dataset]
                p = 1 / n_class
                p_list = [p for _ in range(n_class)]
                res = {"distribution": p_list}
                res = json.dumps(res)
                with open(config_file + "%s_%s_%s.json" % (dataset, label, mode), "w") as wf:
                    wf.write(res)


def exp_flex_component(label_amount):
    config_file = r'../config/'
    save_path = r'../saved_models/'

    if not os.path.exists(config_file):
        os.mkdir(config_file)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    algs = ['uda', 'pseudolabel']
    datasets = ['cifar10', 'cifar100', 'svhn', 'stl10']
    num_labels = [500, 1000, 2000, 4000]
    # seeds = [1, 11, 111]
    seeds = [0]

    dist_port = range(11121, 12120, 1)
    count = 0
    for label in num_labels:
        for alg in algs:
            for dataset in datasets:
                for seed in seeds:
                    # change the configuration of each dataset
                    if dataset == 'cifar10':
                        net = 'WideResNet'
                        num_classes = 10
                        num_labels = label
                        weight_decay = 5e-4
                        depth = 28
                        widen_factor = 2
                    elif dataset == 'cifar100':
                        net = 'WideResNet'
                        num_classes = 100
                        num_labels = label
                        weight_decay = 1e-3
                        depth = 28
                        widen_factor = 8
                    elif dataset == 'svhn':
                        net = 'WideResNet'
                        num_classes = 10
                        num_labels = label
                        weight_decay = 5e-4
                        depth = 28
                        widen_factor = 2
                    elif dataset == 'stl10':
                        net = 'WideResNetVar'
                        num_classes = 10
                        num_labels = label
                        weight_decay = 5e-4
                        depth = 28
                        widen_factor = 2

                    port = dist_port[count]
                    # prepare the configuration file
                    cfg = create_base_config(alg, seed,
                                             dataset, net, num_classes, num_labels,
                                             port,
                                             weight_decay, depth, widen_factor
                                             )
                    count += 1
                    cfg['use_flex'] = True
                    cfg['alg'] += str('_flex')
                    create_configuration(cfg, config_file)


if __name__ == '__main__':
    if not os.path.exists('../saved_models/'):
        os.mkdir('../saved_models/')
    if not os.path.exists('../config/'):
        os.mkdir('../config/')
    label_amount = {'s': [40, 400, 40, 40],
                    'm': [250, 2500, 250, 250],
                    'l': [4000, 10000, 1000, 1000]}

    for i in label_amount:
        exp_baseline(label_amount=[label_amount[i]])
        # exp_flex_component(label_amount=label_amount[i])
    flexmatch_config_generation()
