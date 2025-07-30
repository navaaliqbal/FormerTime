import time
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from loss import CE, BCE
from torch.optim.lr_scheduler import LambdaLR


class Trainer():
    def __init__(self, args, model, train_loader, test_loader, verbose=False):
        self.args = args
        self.verbose = verbose
        self.device = args.device
        self.print_process(self.device)
        self.model = model.to(torch.device(self.device))

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_decay = args.lr_decay_rate
        self.lr_decay_steps = args.lr_decay_steps

        self.cr = CE(self.model) if args.loss == 'ce' else BCE(self.model)

        self.test_cr = torch.nn.CrossEntropyLoss() if args.loss == 'ce' else torch.nn.BCELoss()
        self.num_epoch = args.num_epoch
        self.eval_per_steps = args.eval_per_steps
        self.save_path = args.save_path
        if self.num_epoch:
            self.result_file = open(self.save_path + '/result.txt', 'w')
            self.result_file.close()

        self.step = 0
        self.best_metric = -1e9
        self.metric = 'acc'

    def train(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda step: self.lr_decay ** step, verbose=self.verbose)
        for epoch in range(self.num_epoch):
            loss_epoch, time_cost = self._train_one_epoch()
            self.result_file = open(self.save_path + '/result.txt', 'a+')
            self.print_process(
                'Model train epoch:{0},loss:{1},training_time:{2}'.format(epoch + 1, loss_epoch, time_cost))
            print('Model train epoch:{0},loss:{1},training_time:{2}'.format(epoch + 1, loss_epoch, time_cost),
                  file=self.result_file)
            self.result_file.close()
        self.print_process(self.best_metric)
        return self.best_metric

    def _train_one_epoch(self):
        t0 = time.perf_counter()
        self.model.train()
        dataloader = self.train_loader

        loss_sum = 0
        for idx, batch in enumerate(dataloader):
            batch = [x.to(self.device) for x in batch]

            self.optimizer.zero_grad()
            loss = self.cr.compute(batch)
            loss_sum += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()

            self.step += 1
            if self.step % self.lr_decay_steps == 0:
                self.scheduler.step()

            if self.step % self.eval_per_steps == 0:
                test_metric = self.eval_model()
                train_metric = self.eval_model(data_loader=self.train_loader, split_name='train')

                self.print_process(f"Train: {train_metric}")
                self.print_process(f"Test: {test_metric}")

                with open(self.save_path + '/result.txt', 'a+') as f:
                    print(f"\nStep {self.step}", file=f)
                    print("Train:", train_metric, file=f)
                    print("Test:", test_metric, file=f)

                if test_metric[self.metric] >= self.best_metric:
                    if self.args.save_model:
                        torch.save(self.model.state_dict(), self.save_path + '/model.pkl')
                    with open(self.save_path + '/result.txt', 'a+') as f:
                        print(f"Saving model at step {self.step}", file=f)
                    self.best_metric = test_metric[self.metric]

                self.model.train()

        return loss_sum / (idx + 1), time.perf_counter() - t0

    def eval_model(self, data_loader=None, split_name='test'):
        self.model.eval()
        data_loader = data_loader or self.test_loader
        metrics = {'acc': 0, 'f1': 0}
        pred = []
        label = []
        total_loss = 0

        with torch.no_grad():
            for idx, batch in enumerate(data_loader):
                batch = [x.to(self.device) for x in batch]
                ret = self.compute_metrics(batch)
                if len(ret) == 2:
                    pred_b, label_b = ret
                    pred += pred_b
                    label += label_b
                else:
                    pred_b, label_b, loss_b = ret
                    pred += pred_b
                    label += label_b
                    total_loss += loss_b.cpu().item()

        confusion_mat = self._confusion_mat(label, pred)
        self.print_process(f"{split_name} confusion matrix:")
        self.print_process(confusion_mat)
        with open(self.save_path + '/result.txt', 'a+') as f:
            print(f"{split_name} confusion matrix:", file=f)
            print(confusion_mat, file=f)

        if self.args.num_class == 2:
            metrics['f1'] = f1_score(label, pred)
            metrics['precision'] = precision_score(label, pred)
            metrics['recall'] = recall_score(label, pred)
        else:
            metrics['f1'] = f1_score(label, pred, average='macro')
            metrics['micro_f1'] = f1_score(label, pred, average='micro')

        metrics['acc'] = accuracy_score(label, pred)
        metrics[f'{split_name}_loss'] = total_loss / (idx + 1)
        return metrics

    def compute_metrics(self, batch):
        if len(batch) == 2:
            seqs, label = batch
            scores = self.model(seqs)
        else:
            seqs1, seqs2, label = batch
            scores = self.model((seqs1, seqs2))
        if self.args.loss == 'ce':
            _, pred = torch.topk(scores, 1)
            test_loss = self.test_cr(scores, label.view(-1).long())
            pred = pred.view(-1).tolist()
            return pred, label.tolist(), test_loss
        else:
            pred = (scores > self.threshold).int().view(-1).tolist()
            test_loss = self.test_cr(scores.view(-1), label.view(-1).float())
            return pred, label.tolist(), test_loss

    def _confusion_mat(self, label, pred):
        if self.args.loss == 'ce':
            mat = np.zeros((self.args.num_class, self.args.num_class))
        else:
            mat = np.zeros((2, 2))
        for _label, _pred in zip(label, pred):
            mat[_label, _pred] += 1
        return mat

    def print_process(self, *x):
        if self.verbose:
            print(*x)
