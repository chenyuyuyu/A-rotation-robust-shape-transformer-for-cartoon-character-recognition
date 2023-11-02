import os
import time
import shutil
import os.path as osp
from timeit import default_timer as timer

import numpy as np
import torch
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
from py.config import C
import math
from py.utils import recursive_to, ModelPrinter
import torch.nn as nn
import torch.nn.functional as F

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from tensorboardX import SummaryWriter


class Trainer(object):
    def __init__(self, device, model, optimizer, lr_scheduler, train_loader, val_loader, out, iteration=0, epoch=0,
                 bml=1e1000,dist=False):

        self.device = device

        self.model = model
        self.optim = optimizer
        self.lr_scheduler = lr_scheduler

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_size = C.model.batch_size
        self.eval_batch_size = C.model.eval_batch_size

        self.validation_interval = C.io.validation_interval

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.epoch = epoch
        self.iteration = iteration
        self.max_epoch = C.optim.max_epoch
        self.lr_decay_epoch = C.optim.lr_decay_epoch
        self.mean_loss = self.best_mean_loss = bml

        # self.avg_metrics = None
        # self.metrics = np.zeros(0)
        self.printer = ModelPrinter(out)
        self.dist=dist

    def cal_performance(self, pred, label, smoothing=False):
        ''' Apply label smoothing if needed '''

        loss = self.cal_loss(pred, label, smoothing=smoothing)

        pred = F.softmax(pred, dim=1)
        pred = pred.max(1)[1]
        # label = label.max(1)[1]
        n_correct = pred.eq(label).sum().item()

        return loss, n_correct

    def cal_loss(self, pred, label, smoothing=False):
        ''' Calculate cross entropy loss, apply label smoothing if needed. '''

        if smoothing:
            print("smooth")
        else:
            # cross_entropy_loss=nn.CrossEntropyLoss()
            # label = label.max(1)[1]
            # loss = cross_entropy_loss(pred,label)
            loss = F.cross_entropy(pred, label, reduction='sum')
        return loss

    def validate(self, isckpt=True):
        self.printer.tprint("Running validation...", " " * 55)
        # training = self.model.training
        self.model.eval()
        total_loss, n_cls_total, n_cls_correct = 0, 0, 0

        if self.dist is False:
            with torch.no_grad():
                for batch_idx, (sc, label) in enumerate(self.val_loader):
                    input_dict = {
                        "sc": recursive_to(sc, self.device),
                        "label": recursive_to(label, self.device),
                    }

                    # pred = self.model(input_dict["sc"])
                    pred = self.model(input_dict)
                    # pred = F.softmax(pred, dim=1)
                    loss, n_correct = self.cal_performance(
                        pred, input_dict["label"], smoothing=False)
                    n_cls_correct += n_correct
                    total_loss += loss.item()

                    self.printer.tprint(f"Validation [{batch_idx:5d}/{len(self.val_loader):5d}]", " " * 25)
        else:
            with torch.no_grad():
                # for batch_idx, (sc, label,dist_enc,angle_enc) in enumerate(self.val_loader):
                for batch_idx, (sc, label,dist_enc) in enumerate(self.val_loader):
                    input_dict = {
                        "sc": recursive_to(sc, self.device),
                        "label": recursive_to(label, self.device),
                        "dist_enc": recursive_to(dist_enc, self.device),
                        # "angle_enc": recursive_to(angle_enc, self.device),
                    }

                    # pred = self.model(input_dict["sc"],input_dict["dist_enc"])
                    pred=self.model(input_dict)
                    # pred = F.softmax(pred, dim=1)
                    loss, n_correct = self.cal_performance(
                        pred, input_dict["label"], smoothing=False)
                    n_cls_correct += n_correct
                    total_loss += loss.item()

                    self.printer.tprint(f"Validation [{batch_idx:5d}/{len(self.val_loader):5d}]", " " * 25)


        # self.printer.valid_log(len(self.val_loader), self.epoch, self.iteration, self.batch_size, self.metrics[0])
        n_cls_total=self.eval_batch_size*(batch_idx+1)
        self.mean_loss = total_loss / n_cls_total
        mean_accu = n_cls_correct / n_cls_total

        if isckpt:
            torch.save(
                {
                    "iteration": self.iteration,
                    "arch": self.model.__class__.__name__,
                    "optim_state_dict": self.optim.state_dict(),
                    "model_state_dict": self.model.state_dict(),
                    "best_mean_loss": self.best_mean_loss,
                    'lr_scheduler': self.lr_scheduler.state_dict(),
                },
                osp.join(self.out, "checkpoint_lastest.pth.tar"),
            )

            if self.mean_loss < self.best_mean_loss:
                self.best_mean_loss = self.mean_loss
                shutil.copy(
                    osp.join(self.out, "checkpoint_lastest.pth.tar"),
                    osp.join(self.out, "checkpoint_best.pth.tar"),
                )

        # if training:
        #     self.model.train()
        return self.mean_loss, mean_accu

    def train_epoch(self, smoothing):
        self.model.train()
        total_loss, n_cls_total, n_cls_correct = 0, 0, 0

        time = timer()

        if self.dist is False:
            for batch_idx, (sc, label) in enumerate(self.train_loader):

                self.optim.zero_grad()
                # self.metrics[...] = 0

                input_dict = {
                    "sc": recursive_to(sc, self.device),
                    "label": recursive_to(label, self.device),
                }
                # pred = self.model(input_dict["sc"])
                pred = self.model(input_dict)
                # pred = F.softmax(pred, dim=1)
                loss, n_correct = self.cal_performance(
                    pred, input_dict["label"], smoothing=smoothing)

                loss.backward()
                self.optim.step()

                n_cls_correct += n_correct
                total_loss += loss.item()

                if batch_idx % 100 == 0:
                    n_cls_total = self.batch_size * (batch_idx + 1)
                    print('  - loss: {loss: 8.5f}, accuracy: {accu:3.3f} '.format(
                        loss=total_loss / n_cls_total,
                        accu=100 * n_cls_correct / n_cls_total))

                self.iteration += 1
        else:
            # for batch_idx, (sc, label,dist_enc,angle_enc) in enumerate(self.train_loader):
            for batch_idx, (sc, label,dist_enc) in enumerate(self.train_loader):

                self.optim.zero_grad()

                input_dict = {
                    "sc": recursive_to(sc, self.device),
                    "label": recursive_to(label, self.device),
                    "dist_enc": recursive_to(dist_enc, self.device),
                    # "angle_enc": recursive_to(angle_enc, self.device),
                }
                # pred = self.model(input_dict["sc"],input_dict["dist_enc"])
                pred = self.model(input_dict)
                # pred = F.softmax(pred, dim=1)
                loss, n_correct = self.cal_performance(
                    pred, input_dict["label"], smoothing=smoothing)

                loss.backward()
                self.optim.step()

                n_cls_correct += n_correct
                total_loss += loss.item()

                if batch_idx % 100 == 0:
                    n_cls_total = self.batch_size * (batch_idx + 1)
                    print('  - loss: {loss: 8.5f}, accuracy: {accu:3.3f} '.format(
                        loss=total_loss / n_cls_total,
                        accu=100 * n_cls_correct / n_cls_total))

                self.iteration += 1

        n_cls_total = self.batch_size * (batch_idx + 1)
        train_loss = total_loss / n_cls_total
        train_accu = n_cls_correct / n_cls_total
        return train_loss, train_accu

    def train(self):
        ''' Start training '''

        # Use tensorboard to plot curves, e.g. perplexity, accuracy, learning rate
        if C.model.use_tb:
            print("[Info] Use Tensorboard")
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir=os.path.join(self.out, 'tensorboard'))

        log_train_file = os.path.join(self.out, 'train.log')
        log_valid_file = os.path.join(self.out, 'valid.log')

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

        def print_performances(header, ppl, accu, start_time, lr):
            print('  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, lr: {lr:8.5f}, ' \
                  'elapse: {elapse:3.3f} min'.format(
                header=f"({header})", ppl=ppl,
                accu=100 * accu, elapse=(time.time() - start_time) / 60, lr=lr))

        plt.rcParams["figure.figsize"] = (24, 24)
        epoch_size = len(self.train_loader)
        start_epoch = self.iteration // epoch_size

        # valid_losses = []
        for self.epoch in range(start_epoch, self.max_epoch):
            print('[ Epoch', self.epoch, ']')

            start = time.time()
            train_loss, train_accu = self.train_epoch(smoothing=C.model.label_smoothing)
            train_ppl = math.exp(min(train_loss, 100))
            # Current learning rate
            lr = self.optim.state_dict()['param_groups'][0]['lr']
            print_performances('Training', train_ppl, train_accu, start, lr)

            start = time.time()
            valid_loss, valid_accu = self.validate()
            valid_ppl = math.exp(min(valid_loss, 100))
            print_performances('Validation', valid_ppl, valid_accu, start, lr)

            # valid_losses += [valid_loss]
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=self.epoch, loss=train_loss,
                    ppl=train_ppl, accu=100 * train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=self.epoch, loss=valid_loss,
                    ppl=valid_ppl, accu=100 * valid_accu))

            if C.model.use_tb:
                tb_writer.add_scalars('ppl', {'train': train_ppl, 'val': valid_ppl}, self.epoch)
                tb_writer.add_scalars('accuracy', {'train': train_accu * 100, 'val': valid_accu * 100}, self.epoch)
                tb_writer.add_scalar('learning_rate', lr, self.epoch)

            self.lr_scheduler.step()

        # self.writer.close()
