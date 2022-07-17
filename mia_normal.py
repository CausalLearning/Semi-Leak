from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import os
import pickle as pkl
from torch.utils.data import WeightedRandomSampler, DataLoader
import time

torch.manual_seed(0)
torch.set_num_threads(1)


class MLP_BLACKBOX(nn.Module):
    def __init__(self, dim_in):
        super(MLP_BLACKBOX, self).__init__()
        self.dim_in = dim_in
        self.fc1 = nn.Linear(self.dim_in, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = x.view(-1, self.dim_in)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AttackTrainingBlackBox():
    def __init__(self, args):
        self.args = args
        self.device = args.gpu
        self.attack_model = MLP_BLACKBOX(args.num_classes)

        self.attack_model.apply(self._weights_init_normal)
        self.attack_model.cuda(self.device)

        self.optimizer = torch.optim.Adam(self.attack_model.parameters(),
                                          lr=0.001, weight_decay=args.weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.target_performance = [0.0, 0.0, 0.0, 0.0]
        self.generate_data()

    def _weights_init_normal(self, m):
        '''Takes in a module and initializes all linear layers with weight
           values taken from a normal distribution.'''

        classname = m.__class__.__name__
        # for every Linear layer in a model
        if classname.find('Linear') != -1:
            y = m.in_features
            # m.weight.data shoud be taken from a normal distribution
            m.weight.data.normal_(0.0, 1 / np.sqrt(y))
            # m.bias.data should be 0
            m.bias.data.fill_(0)

    def generate_dataloader(self, data, membsership_label=1):

        data = np.array(data)
        label = np.array([membsership_label] * len(data))

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(data),
            torch.from_numpy(label).long())
        data_loader = DataLoader(
            dataset,
            batch_size=args.attack_batch_size,
            num_workers=args.num_workers,
            shuffle=False,)

        return data_loader

    def generate_data(self):
        args = self.args
        model_path = os.path.join(args.save_dir, "%s_%s_%s_0" % (
            args.ssl_method, args.dataset, args.num_labels))
        with open(os.path.join(model_path, "query_results_%s.pkl" % (args.target_epoch)), "rb") as rf:
            print("load from", os.path.join(model_path,
                  "query_results_%s.pkl" % (args.target_epoch)))
            res = pkl.load(rf)
        self.cal_target_performance(res)
        train_non_mem = self.parse_posteriors(res["shadow_test"])
        train_mem_labeled = self.parse_posteriors(res["shadow_train_lb"])
        train_mem_unlabeled = self.parse_posteriors(res["shadow_train_ulb"])

        test_non_mem = self.parse_posteriors(res["target_test"])
        test_mem_labeled = self.parse_posteriors(res["target_train_lb"])
        test_mem_unlabeled = self.parse_posteriors(res["target_train_ulb"])

        # generate seperate dataloader for evaluation :D
        self.dataloader_train_non_mem = self.generate_dataloader(
            train_non_mem, membsership_label=0)
        self.dataloader_train_mem_labeled = self.generate_dataloader(
            train_mem_labeled, membsership_label=1)
        self.dataloader_train_mem_unlabeled = self.generate_dataloader(
            train_mem_unlabeled, membsership_label=1)
        self.dataloader_test_non_mem = self.generate_dataloader(
            test_non_mem, membsership_label=0)
        self.dataloader_test_mem_labeled = self.generate_dataloader(
            test_mem_labeled, membsership_label=1)
        self.dataloader_test_mem_unlabeled = self.generate_dataloader(
            test_mem_unlabeled, membsership_label=1)

        train_data = np.array(train_mem_labeled +
                              train_mem_unlabeled + train_non_mem)
        train_target = np.array(
            [1] * len(train_mem_labeled + train_mem_unlabeled) + [0] * len(train_non_mem))
        train_all = torch.utils.data.TensorDataset(
            torch.from_numpy(train_data),
            torch.from_numpy(train_target).long())

        # weight = [1 / len(train_mem_labeled)] * len(train_mem_labeled) + [
        #     1 / len(train_non_mem)] * len(train_non_mem)
        # sampler = WeightedRandomSampler(
        #     weight, len(weight), replacement=True)
        self.train_loader = DataLoader(
            train_all,
            batch_size=args.attack_batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            # sampler=sampler
        )

        test_target = np.array(
            [1] * len(test_mem_labeled + test_mem_unlabeled) + [0] * len(test_non_mem))
        test_data = np.array(test_mem_labeled +
                             test_mem_unlabeled + test_non_mem)
        test_all = torch.utils.data.TensorDataset(
            torch.from_numpy(test_data),
            torch.from_numpy(test_target).long())

        self.test_loader = DataLoader(
            test_all,
            batch_size=args.attack_batch_size,
            num_workers=args.num_workers,
            shuffle=False,)

    def train(self):
        for epoch in range(50):
            print("Epoch: %d" % epoch)
            self.attack_model.train()
            for inputs, targets in self.train_loader:
                # print(torch.count_nonzero(targets),)
                self.optimizer.zero_grad()
                inputs, targets = inputs.cuda(
                    self.device), targets.cuda(self.device)
                outputs = self.attack_model(inputs)
                posteriors = F.softmax(outputs, dim=1)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

        train_acc, train_precision, train_recall, train_f1, train_auc = self.evaluate(
            self.train_loader)
        test_acc, test_precision, test_recall, test_f1, test_auc = self.evaluate(
            self.test_loader)
        labeled_auc = self.cal_seperate_auc(
            [self.dataloader_test_mem_labeled, self.dataloader_test_non_mem])
        unlabeled_auc = self.cal_seperate_auc(
            [self.dataloader_test_mem_unlabeled, self.dataloader_test_non_mem])
        self.save_attack_result()

        print(('Epoch: %d, Overall Train Acc: %.3f%%, precision:%.3f, recall:%.3f, f1:%.3f, auc: %.3f' % (
            epoch, 100. * train_acc, train_precision, train_recall, train_f1, train_auc)))
        print(('Epoch: %d, Overall Test Acc: %.3f%%, precision:%.3f, recall:%.3f, f1:%.3f, auc: %.3f, labeled_auc: %.3f, unlabeled_auc: %.3f' % (
            epoch, 100. * test_acc, test_precision, test_recall, test_f1, test_auc, labeled_auc, unlabeled_auc)))

        train_tuple = (train_acc, train_precision,
                       train_recall, train_f1, train_auc)
        test_tuple = (test_acc, test_precision, test_recall, test_f1, test_auc)
        seperate_auc_tuple = (labeled_auc, unlabeled_auc)
        return train_tuple, test_tuple, seperate_auc_tuple

    @torch.no_grad()
    def evaluate(self, dataloader):
        labels = []
        pred_labels = []
        pred_posteriors = []

        self.attack_model.eval()
        for inputs, targets in dataloader:
            inputs, targets = inputs.cuda(self.device), targets.cuda(
                self.device)
            outputs = self.attack_model(inputs)
            posteriors = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            labels += targets.cpu().tolist()
            pred_labels += predicted.cpu().tolist()
            pred_posteriors += posteriors.cpu().tolist()
        pred_posteriors = [row[1] for row in pred_posteriors]

        test_acc, test_precision, test_recall, test_f1, test_auc = self.cal_metrics(
            labels, pred_labels, pred_posteriors)
        return test_acc, test_precision, test_recall, test_f1, test_auc

    def save_attack_result(self):
        self.sample_info = {}
        self.sample_info["target_test"] = self.cal_attack_performance(
            self.dataloader_test_non_mem)
        self.sample_info["target_train_lb"] = self.cal_attack_performance(
            self.dataloader_test_mem_labeled)
        self.sample_info["target_train_ulb"] = self.cal_attack_performance(
            self.dataloader_test_mem_unlabeled)

        self.sample_info["shadow_test"] = self.cal_attack_performance(
            self.dataloader_train_non_mem)
        self.sample_info["shadow_train_lb"] = self.cal_attack_performance(
            self.dataloader_train_mem_labeled)
        self.sample_info["shadow_train_ulb"] = self.cal_attack_performance(
            self.dataloader_train_mem_unlabeled)

    @torch.no_grad()
    def cal_attack_performance(self, dataloader):
        labels = []
        pred_labels = []
        pred_posteriors = []
        for inputs, targets in dataloader:
            inputs, targets = inputs.cuda(self.device), targets.cuda(
                self.device)
            outputs = self.attack_model(inputs)
            posteriors = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            labels += targets.cpu().tolist()
            pred_labels += predicted.cpu().tolist()
            pred_posteriors += posteriors.cpu().tolist()
        res = {}
        for i in range(len(labels)):
            res[i] = {"label": labels[i], "pred_label": pred_labels[i],
                      "pred_posteiors": pred_posteriors[i]}
        return res

    @torch.no_grad()
    def cal_seperate_auc(self, dataloader_list):
        labels = []
        pred_labels = []
        pred_posteriors = []

        for dataloader in dataloader_list:
            for inputs, targets in dataloader:
                inputs, targets = inputs.cuda(self.device), targets.cuda(
                    self.device)
                outputs = self.attack_model(inputs)
                posteriors = F.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                labels += targets.cpu().tolist()
                pred_labels += predicted.cpu().tolist()
                pred_posteriors += posteriors.cpu().tolist()

        pred_posteriors = [row[1] for row in pred_posteriors]

        auc = roc_auc_score(labels, pred_posteriors)
        return auc

    def cal_metrics(self, label, pred_label, pred_posteriors):
        acc = accuracy_score(label, pred_label)
        precision = precision_score(label, pred_label)
        recall = recall_score(label, pred_label)
        f1 = f1_score(label, pred_label)
        auc = roc_auc_score(label, pred_posteriors)
        return acc, precision, recall, f1, auc

    def cal_target_performance(self, res):
        sl0, sp0 = self.get_predion_info(res["shadow_test"])
        sl1, sp1 = self.get_predion_info(res["shadow_train_lb"])
        sl2, sp2 = self.get_predion_info(res["shadow_train_ulb"])

        tl0, tp0 = self.get_predion_info(res["target_test"])
        tl1, tp1 = self.get_predion_info(res["target_train_lb"])
        tl2, tp2 = self.get_predion_info(res["target_train_ulb"])

        target_train_acc = accuracy_score(tl1 + tl2, tp1 + tp2)
        target_test_acc = accuracy_score(tl0, tp0)
        shadow_train_acc = accuracy_score(sl1 + sl2, sp1 + sp2)
        shadow_test_acc = accuracy_score(sl0, sp0)
        print("target_performance: ", target_train_acc,
              target_test_acc, shadow_train_acc, shadow_test_acc)
        self.target_performance = [
            target_train_acc, target_test_acc, shadow_train_acc, shadow_test_acc]

    def get_predion_info(self, data):
        labels = []
        pred_labels = []
        for k in data.keys():
            label = data[k]["label"]
            posteriors = data[k]["original"]
            pred_label = np.argmax(posteriors)
            labels.append(label)
            pred_labels.append(pred_label)
        return labels, pred_labels

    def parse_posteriors(self, data):
        res = []
        for k in data.keys():
            # res.append(data[k]["weak"])
            # res.append(data[k]["strong"])
            res.append(sorted(data[k]["original"], reverse=True))
            # res.append(sorted(data[k]["weak"][0], reverse=True))
        return res

    def split_dataset(self, dataset):
        np.random.seed(0)
        np.random.shuffle(dataset)
        half = len(dataset) // 2
        training, testing = dataset[:half], dataset[half:]
        return training, testing


class AttackTrainingBlackBoxMetric():
    def __init__(self, args):
        self.args = args
        self.device = args.gpu
        self.num_classes = args.num_classes

        self.target_performance = [0.0, 0.0, 0.0, 0.0]
        self.generate_data()

    def _weights_init_normal(self, m):
        '''Takes in a module and initializes all linear layers with weight
           values taken from a normal distribution.'''

        classname = m.__class__.__name__
        # for every Linear layer in a model
        if classname.find('Linear') != -1:
            y = m.in_features
            # m.weight.data shoud be taken from a normal distribution
            m.weight.data.normal_(0.0, 1 / np.sqrt(y))
            # m.bias.data should be 0
            m.bias.data.fill_(0)

    def generate_dataloader(self, data, membsership_label=1):

        data = np.array(data)
        label = np.array([membsership_label] * len(data))

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(data),
            torch.from_numpy(label).long())
        data_loader = DataLoader(
            dataset,
            batch_size=args.attack_batch_size,
            num_workers=args.num_workers,
            shuffle=False,)

        return data_loader

    def train(self):
        train_tuple0, test_tuple0, seperate_auc_tuple0, test_results0 = self._mem_inf_via_corr()
        self.print_result("correct train", train_tuple0)
        self.print_result("correct test", test_tuple0)

        train_tuple1, test_tuple1, seperate_auc_tuple1, test_results1 = self._mem_inf_thre(
            'confidence', self.s_tr_conf, self.s_te_conf, self.t_tr_conf, self.t_te_conf)
        self.print_result("confidence train", train_tuple1)
        self.print_result("confidence test", test_tuple1)

        train_tuple2, test_tuple2,  seperate_auc_tuple2, test_results2 = self._mem_inf_thre(
            'entropy', -self.s_tr_entr, -self.s_te_entr, -self.t_tr_entr, -self.t_te_entr)
        self.print_result("entropy train", train_tuple2)
        self.print_result("entropy test", test_tuple2)

        train_tuple3, test_tuple3,  seperate_auc_tuple3, test_results3 = self._mem_inf_thre(
            'modified entropy', -self.s_tr_m_entr, -self.s_te_m_entr, -self.t_tr_m_entr, -self.t_te_m_entr)
        self.print_result("modified entropy train", train_tuple3)
        self.print_result("modified entropy test", test_tuple3)

        return train_tuple0, test_tuple0, seperate_auc_tuple0, train_tuple1, test_tuple1, seperate_auc_tuple1, train_tuple2, test_tuple2, seperate_auc_tuple2, train_tuple3, test_tuple3, seperate_auc_tuple3

    def inference(self):
        return self.train()

    def generate_data(self):
        args = self.args
        model_path = os.path.join(args.save_dir, "%s_%s_%s_0" % (
            args.ssl_method, args.dataset, args.num_labels))
        with open(os.path.join(model_path, "query_results_%s.pkl" % (args.target_epoch)), "rb") as rf:
            print("load from", os.path.join(model_path,
                  "query_results_%s.pkl" % (args.target_epoch)))
            res = pkl.load(rf)
        self.cal_target_performance(res)
        train_non_mem, train_non_mem_original_label = self.parse_posteriors_labels(
            res["shadow_test"])
        train_mem_labeled, train_mem_labeled_original_label = self.parse_posteriors_labels(
            res["shadow_train_lb"])
        train_mem_unlabeled, train_mem_unlabeled_original_label = self.parse_posteriors_labels(
            res["shadow_train_ulb"])

        test_non_mem, test_original_label = self.parse_posteriors_labels(
            res["target_test"])
        test_mem_labeled, test_mem_original_label = self.parse_posteriors_labels(
            res["target_train_lb"])
        test_mem_unlabeled, test_non_mem_original_label = self.parse_posteriors_labels(
            res["target_train_ulb"])

        self.num_label_train = len(train_mem_labeled)
        self.num_train = len(train_mem_labeled + train_mem_unlabeled)
        self.s_tr_outputs, self.s_tr_labels = np.array(train_mem_labeled + train_mem_unlabeled), np.array(
            train_mem_labeled_original_label + train_mem_unlabeled_original_label)
        self.s_te_outputs, self.s_te_labels = np.array(
            train_non_mem), np.array(train_non_mem_original_label)
        self.t_tr_outputs, self.t_tr_labels = np.array(
            test_mem_labeled + test_mem_unlabeled), np.array(test_mem_original_label + test_non_mem_original_label)
        self.t_te_outputs, self.t_te_labels = np.array(
            test_non_mem), np.array(test_original_label)

        self.s_tr_mem_labels = np.ones(len(train_non_mem))
        self.s_te_mem_labels = np.zeros(len(train_non_mem))
        self.t_tr_mem_labels = np.ones(len(train_non_mem))
        self.t_te_mem_labels = np.zeros(len(train_non_mem))

        # prediction correctness
        self.s_tr_corr = (np.argmax(self.s_tr_outputs, axis=1)
                          == self.s_tr_labels).astype(int)
        self.s_te_corr = (np.argmax(self.s_te_outputs, axis=1)
                          == self.s_te_labels).astype(int)
        self.t_tr_corr = (np.argmax(self.t_tr_outputs, axis=1)
                          == self.t_tr_labels).astype(int)
        self.t_te_corr = (np.argmax(self.t_te_outputs, axis=1)
                          == self.t_te_labels).astype(int)

        # prediction confidence
        self.s_tr_conf = np.array(
            [self.s_tr_outputs[i, self.s_tr_labels[i]] for i in range(len(self.s_tr_labels))])
        self.s_te_conf = np.array(
            [self.s_te_outputs[i, self.s_te_labels[i]] for i in range(len(self.s_te_labels))])
        self.t_tr_conf = np.array(
            [self.t_tr_outputs[i, self.t_tr_labels[i]] for i in range(len(self.t_tr_labels))])
        self.t_te_conf = np.array(
            [self.t_te_outputs[i, self.t_te_labels[i]] for i in range(len(self.t_te_labels))])

        # prediction entropy
        self.s_tr_entr = self._entr_comp(self.s_tr_outputs)
        self.s_te_entr = self._entr_comp(self.s_te_outputs)
        self.t_tr_entr = self._entr_comp(self.t_tr_outputs)
        self.t_te_entr = self._entr_comp(self.t_te_outputs)

        # prediction modified entropy
        self.s_tr_m_entr = self._m_entr_comp(
            self.s_tr_outputs, self.s_tr_labels)
        self.s_te_m_entr = self._m_entr_comp(
            self.s_te_outputs, self.s_te_labels)
        self.t_tr_m_entr = self._m_entr_comp(
            self.t_tr_outputs, self.t_tr_labels)
        self.t_te_m_entr = self._m_entr_comp(
            self.t_te_outputs, self.t_te_labels)

    def print_result(self, name, given_tuple):
        print("%s" % name, "acc:%.3f, precision:%.3f, recall:%.3f, f1:%.3f, auc:%.3f" % given_tuple)

    def _log_value(self, probs, small_value=1e-30):
        return -np.log(np.maximum(probs, small_value))

    def _entr_comp(self, probs):
        return np.sum(np.multiply(probs, self._log_value(probs)), axis=1)

    def _m_entr_comp(self, probs, true_labels):
        log_probs = self._log_value(probs)
        reverse_probs = 1-probs
        log_reverse_probs = self._log_value(reverse_probs)
        modified_probs = np.copy(probs)
        modified_probs[range(true_labels.size), true_labels] = reverse_probs[range(
            true_labels.size), true_labels]
        modified_log_probs = np.copy(log_reverse_probs)
        modified_log_probs[range(true_labels.size), true_labels] = log_probs[range(
            true_labels.size), true_labels]
        return np.sum(np.multiply(modified_probs, modified_log_probs), axis=1)

    def _thre_setting(self, tr_values, te_values):
        value_list = np.concatenate((tr_values, te_values))
        thre, max_acc = 0, 0
        for value in value_list:
            tr_ratio = np.sum(tr_values >= value)/(len(tr_values)+0.0)
            te_ratio = np.sum(te_values < value)/(len(te_values)+0.0)
            acc = 0.5*(tr_ratio + te_ratio)
            if acc > max_acc:
                thre, max_acc = value, acc
        return thre

    def _mem_inf_via_corr(self):
        # # perform membership inference attack based on whether the input is correctly classified or not
        train_mem_label = np.concatenate(
            [self.s_tr_mem_labels, self.s_te_mem_labels], axis=-1)
        train_pred_label = np.concatenate(
            [self.s_tr_corr, self.s_te_corr], axis=-1)
        train_pred_posteriors = np.concatenate(
            [self.s_tr_corr, self.s_te_corr], axis=-1)  # same as train_pred_label
        train_target_label = np.concatenate(
            [self.s_tr_labels, self.s_te_labels], axis=-1)

        test_mem_label = np.concatenate(
            [self.t_tr_mem_labels, self.t_te_mem_labels], axis=-1)
        test_pred_label = np.concatenate(
            [self.t_tr_corr, self.t_te_corr], axis=-1)
        test_pred_posteriors = np.concatenate(
            [self.t_tr_corr, self.t_te_corr], axis=-1)  # same as train_pred_label
        test_target_label = np.concatenate(
            [self.t_tr_labels, self.t_te_labels], axis=-1)

        train_acc, train_precision, train_recall, train_f1, train_auc = self.cal_metrics(
            train_mem_label, train_pred_label, train_pred_posteriors)
        test_acc, test_precision, test_recall, test_f1, test_auc = self.cal_metrics(
            test_mem_label, test_pred_label, test_pred_posteriors)

        labeled_auc, unlabeled_auc = self.cal_seperate_auc(
            test_mem_label, test_pred_label, test_pred_posteriors)

        test_results = {"test_mem_label": test_mem_label,
                        "test_pred_label": test_pred_label,
                        "test_pred_prob": test_pred_posteriors,
                        "test_target_label": test_target_label}

        train_tuple = (train_acc, train_precision,
                       train_recall, train_f1, train_auc)
        test_tuple = (test_acc, test_precision,
                      test_recall, test_f1, test_auc)
        seperate_auc_tuple = (labeled_auc, unlabeled_auc)
        # print(train_tuple, test_tuple)
        return train_tuple, test_tuple, seperate_auc_tuple, test_results

    def _mem_inf_thre(self, v_name, s_tr_values, s_te_values, t_tr_values, t_te_values):
        # perform membership inference attack by thresholding feature values: the feature can be prediction confidence,
        # (negative) prediction entropy, and (negative) modified entropy

        train_mem_label = []
        train_pred_label = []
        train_pred_posteriors = []
        train_target_label = []

        test_mem_label = []
        test_pred_label = []
        test_pred_posteriors = []
        test_target_label = []

        thre_list = [self._thre_setting(s_tr_values[self.s_tr_labels == num],
                                        s_te_values[self.s_te_labels == num]) for num in range(self.num_classes)]

        # shadow train
        for i in range(len(s_tr_values)):
            original_label = self.s_tr_labels[i]
            thre = thre_list[original_label]
            pred = s_tr_values[i]
            pred_label = int(s_tr_values[i] >= thre)

            train_mem_label.append(1)
            train_pred_label.append(pred_label)
            # indicator function, so the posterior equals to 0 or 1
            train_pred_posteriors.append(pred)
            train_target_label.append(original_label)

        # shadow test
        for i in range(len(s_te_values)):
            original_label = self.s_te_labels[i]
            thre = thre_list[original_label]
            pred = s_te_values[i]
            pred_label = int(s_te_values[i] >= thre)

            train_mem_label.append(0)
            train_pred_label.append(pred_label)
            # indicator function, so the posterior equals to 0 or 1
            train_pred_posteriors.append(pred)
            train_target_label.append(original_label)

        # target train
        for i in range(len(t_tr_values)):
            original_label = self.t_tr_labels[i]
            thre = thre_list[original_label]
            pred = t_tr_values[i]
            pred_label = int(t_tr_values[i] >= thre)

            test_mem_label.append(1)
            test_pred_label.append(pred_label)
            # indicator function, so the posterior equals to 0 or 1
            test_pred_posteriors.append(pred)
            test_target_label.append(original_label)

        # target test
        for i in range(len(t_te_values)):
            original_label = self.t_te_labels[i]
            thre = thre_list[original_label]
            pred = t_te_values[i]
            pred_label = int(t_te_values[i] >= thre)

            test_mem_label.append(0)
            test_pred_label.append(pred_label)
            # indicator function, so the posterior equals to 0 or 1
            test_pred_posteriors.append(pred)
            test_target_label.append(original_label)

        train_acc, train_precision, train_recall, train_f1, train_auc = self.cal_metrics(
            train_mem_label, train_pred_label, train_pred_posteriors)
        test_acc, test_precision, test_recall, test_f1, test_auc = self.cal_metrics(
            test_mem_label, test_pred_label, test_pred_posteriors)
        labeled_auc, unlabeled_auc = self.cal_seperate_auc(
            test_mem_label, test_pred_label, test_pred_posteriors)

        train_tuple = (train_acc, train_precision,
                       train_recall, train_f1, train_auc)
        test_tuple = (test_acc, test_precision,
                      test_recall, test_f1, test_auc)
        seperate_auc_tuple = (labeled_auc, unlabeled_auc)
        test_results = {"test_mem_label": test_mem_label,
                        "test_pred_label": test_pred_label,
                        "test_pred_prob": test_pred_posteriors,
                        "test_target_label": test_target_label}

        return train_tuple, test_tuple, seperate_auc_tuple, test_results

    def save_attack_result(self):
        self.sample_info = {}
        self.sample_info["target_test"] = self.cal_attack_performance(
            self.dataloader_test_non_mem)
        self.sample_info["target_train_lb"] = self.cal_attack_performance(
            self.dataloader_test_mem_labeled)
        self.sample_info["target_train_ulb"] = self.cal_attack_performance(
            self.dataloader_test_mem_unlabeled)

        self.sample_info["shadow_test"] = self.cal_attack_performance(
            self.dataloader_train_non_mem)
        self.sample_info["shadow_train_lb"] = self.cal_attack_performance(
            self.dataloader_train_mem_labeled)
        self.sample_info["shadow_train_ulb"] = self.cal_attack_performance(
            self.dataloader_train_mem_unlabeled)

    def cal_metrics(self, label, pred_label, pred_posteriors):
        acc = accuracy_score(label, pred_label)
        precision = precision_score(label, pred_label)
        recall = recall_score(label, pred_label)
        f1 = f1_score(label, pred_label)
        auc = roc_auc_score(label, pred_posteriors)
        return acc, precision, recall, f1, auc

    def cal_seperate_auc(self, label, pred_label, pred_posteriors):

        lb_label = []
        lb_pred_label = []
        lb_pred_posteriors = []
        ulb_label = []
        ulb_pred_label = []
        ulb_pred_posteriors = []

        for i in range(len(label)):
            if i < self.num_label_train or i >= self.num_train:  # labeled or non-mem
                lb_label.append(label[i])
                lb_pred_label.append(pred_label[i])
                lb_pred_posteriors.append(pred_posteriors[i])
            if i >= self.num_label_train:
                ulb_label.append(label[i])
                ulb_pred_label.append(pred_label[i])
                ulb_pred_posteriors.append(pred_posteriors[i])
        labeled_auc = roc_auc_score(lb_label, lb_pred_posteriors)
        unlabeled_auc = roc_auc_score(ulb_label, ulb_pred_posteriors)

        return labeled_auc, unlabeled_auc

    def cal_target_performance(self, res):
        sl0, sp0 = self.get_predion_info(res["shadow_test"])
        sl1, sp1 = self.get_predion_info(res["shadow_train_lb"])
        sl2, sp2 = self.get_predion_info(res["shadow_train_ulb"])

        tl0, tp0 = self.get_predion_info(res["target_test"])
        tl1, tp1 = self.get_predion_info(res["target_train_lb"])
        tl2, tp2 = self.get_predion_info(res["target_train_ulb"])

        target_train_acc = accuracy_score(tl1 + tl2, tp1 + tp2)
        target_test_acc = accuracy_score(tl0, tp0)
        shadow_train_acc = accuracy_score(sl1 + sl2, sp1 + sp2)
        shadow_test_acc = accuracy_score(sl0, sp0)
        print("target_performance: ", target_train_acc,
              target_test_acc, shadow_train_acc, shadow_test_acc)
        self.target_performance = [
            target_train_acc, target_test_acc, shadow_train_acc, shadow_test_acc]

    def get_predion_info(self, data):
        labels = []
        pred_labels = []
        for k in data.keys():
            label = data[k]["label"]
            posteriors = data[k]["original"]
            pred_label = np.argmax(posteriors)
            labels.append(label)
            pred_labels.append(pred_label)
        return labels, pred_labels

    def parse_posteriors_labels(self, data):
        res = []
        labels = []
        for k in data.keys():
            res.append(data[k]["original"])
            labels.append(data[k]["label"])

        return res, labels


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def write_res(opt, wf, attack_name, res):
    line = "%s,%s,%s,%s,%s," % (
        opt.ssl_method, opt.dataset, opt.net, opt.num_labels, opt.target_epoch)
    line += "%s," % attack_name
    line += ",".join(["%.3f" % (row) for row in res])
    line += "\n"
    wf.write(line)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')

    '''
    Saving & loading of the model.
    '''
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('-sn', '--save_name', type=str, default='fixmatch')
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
    parser.add_argument('--widen_factor', type=int, default=1)
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
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', type=str2bool, default=False,
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    # attack related params
    parser.add_argument('--ssl_method', type=str, default="fixmatch")
    # parser.add_argument('--attack_type', type=str, default='normal', help="normal or augmented ")
    parser.add_argument('--augmented_num', default=10, type=int,
                        help='how many queries with different augmentations, e.g., 10 means generate 10 weak view and 10 augmented views to query the target model')
    parser.add_argument('--target_epoch', default=100, type=int,
                        help='which model you are using.')
    parser.add_argument('--attack_batch_size', default=256, type=int,
                        help='attack batch size. ')
    parser.add_argument('--attack_name', type=str,
                        default="black-box", help="black-box or metric")

    # config file
    args = parser.parse_args()

    t_start = time.time()

    if args.attack_name == "black-box":
        s = AttackTrainingBlackBox(args)
        train_tuple, test_tuple, seperate_auc_tuple = s.train()
        target_train_acc, target_test_acc, shadow_train_acc, shadow_test_acc = s.target_performance
        res = [target_train_acc, target_test_acc, shadow_train_acc,
               shadow_test_acc] + list(train_tuple) + list(test_tuple) + list(seperate_auc_tuple)
        os.makedirs("log/exp_results/", exist_ok=True)
        with open("log/exp_results/mia.txt", "a") as wf:
            write_res(args, wf, "black-box", res)

        model_path = os.path.join(args.save_dir, "%s_%s_%s_0" % (
            args.ssl_method, args.dataset, args.num_labels))
        save_name = "mia_normal_%s.pkl" % (args.target_epoch)
        with open(os.path.join(model_path, save_name), "wb") as wf2:
            pkl.dump(s.sample_info, wf2)
    elif args.attack_name == "metric":
        s = AttackTrainingBlackBoxMetric(args)
        train_tuple0, test_tuple0, seperate_auc_tuple0, train_tuple1, test_tuple1, seperate_auc_tuple1, train_tuple2, test_tuple2, seperate_auc_tuple2, train_tuple3, test_tuple3, seperate_auc_tuple3 = s.train()
        target_train_acc, target_test_acc, shadow_train_acc, shadow_test_acc = s.target_performance

        res0 = [target_train_acc, target_test_acc, shadow_train_acc,
                shadow_test_acc] + list(train_tuple0) + list(test_tuple0) + list(seperate_auc_tuple0)
        res1 = [target_train_acc, target_test_acc, shadow_train_acc,
                shadow_test_acc] + list(train_tuple1) + list(test_tuple1) + list(seperate_auc_tuple1)
        res2 = [target_train_acc, target_test_acc, shadow_train_acc,
                shadow_test_acc] + list(train_tuple2) + list(test_tuple2) + list(seperate_auc_tuple2)
        res3 = [target_train_acc, target_test_acc, shadow_train_acc,
                shadow_test_acc] + list(train_tuple3) + list(test_tuple3) + list(seperate_auc_tuple3)
        os.makedirs("log/exp_results/", exist_ok=True)
        with open("log/exp_results/mia.txt", "a") as wf:
            write_res(args, wf, "metric-corr", res0)
            write_res(args, wf, "metric-conf", res1)
            write_res(args, wf, "metric-ent", res2)
            write_res(args, wf, "metric-ment", res3)
    print("Total time: %.3f" % (time.time() - t_start))
    print("Finish")
