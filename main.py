import torch
import argparse
import Datasetloder
import random
import torch.backends.cudnn as cudnn
import torch.optim
from   torch.utils import data
import numpy as np
from   image_dataset import DataSet
import torchvision.transforms as transforms
import argparse
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.parallel
import os
import time
import shutil
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math
import CNNcls
from plot import plot_function
from functools import reduce
import operator
import scipy.io as io


parser = argparse.parser = argparse.ArgumentParser(description='CNN for Flower classification')
parser.add_argument('--batchsize', default=10, type=int, metavar='NUM', help='batchsize')
parser.add_argument('--image_dir', default='./data/FLO/', metavar='STR', help='image directory')
parser.add_argument('--n_classes', default = 17, metavar='NUM', help='number of classes')
parser.add_argument('--lr', default = 0.001, metavar='NUM', help='learning rate')
parser.add_argument('--weight_decay', default= 0.0005,  type=float,metavar='NUM', help='weight decay')
parser.add_argument('--epochs', default= 50,  type=int,metavar='NUM', help='total epochs')
parser.add_argument('--print_freq', default= 10, type=int,metavar='NUM', help='print frequency')
parser.add_argument('--output', default= 'FLO17', type=str, metavar='DIRECTORY',help='name of output')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output_vec, target):
    """Computes the precision@k for the specified values of k"""
    output = output_vec

    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)

    prec = 0.0
    pred_prob = []
    for i in range(batch_size):
        t = target[i]
        pred_prob.append(output[i][t])

        if pred[i] == t:
            prec += 1

    return prec*100.0 / batch_size


def save_checkpoint(config, state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './models/{}_model_best.pth.tar'.format(config['output']))


def save_model(config, model,optimizer,epoch,best_meas,best_epoch,is_best,fname):
    save_checkpoint(config, {
        'epoch': epoch + 1,
        'arch': 'resnet101',
        'state_dict': model.state_dict(),
        'best_meas': best_meas,
        'best_epoch': best_epoch,
        'optimizer' : optimizer.state_dict(),
    }, is_best,fname)


def train(config, model, optimizer, criterion, train_loader, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    for i, (inputs, target) in enumerate(train_loader):

        inputs = inputs.cuda(non_blocking=True)  # [batch,3,224,224]
        target = target.cuda(non_blocking=True)  # [batch]

        # forward propagate
        outputs = model(inputs)   # [batch, 17]

        m = inputs.size(0)

        # calculate loss
        loss = criterion(outputs, target)

        # back propagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # calculate accuracy
        acc= accuracy(outputs, target)
        losses.update(loss, m)
        top1.update(acc, m)

        if i % config['print_freq'] == 0:
            print('Epoch: [{0}][{1}/{2}] '
                  'Loss {loss.val:.4f} (avg: {loss.avg:.4f}) '
                  'Prec@1 {top1.val:.3f} (avg: {top1.avg:.3f}) '.format(
                epoch, i, len(train_loader), loss=losses, top1=top1))
    return top1.avg, losses.avg


def test(config, model, criterion, valid_loader, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    model.eval()

    with torch.no_grad():
        for i, (inputs, target) in enumerate(valid_loader):

            inputs = inputs.cuda(non_blocking=True)  # [batch,3,224,224]
            target = target.cuda(non_blocking=True)  # [batch]

            # forward propagate
            outputs = model(inputs)  # [batch, 17]
            m = inputs.size(0)

            # calculate loss
            loss = criterion(outputs, target)

            # calculate accuracy
            acc = accuracy(outputs, target)
            losses.update(loss, m)
            top1.update(acc, m)

            if i % config['print_freq'] == 0:
                print('Epoch: [{0}][{1}/{2}] '
                      'Loss {loss.val:.4f} (avg: {loss.avg:.4f}) '
                      'Prec@1 {top1.val:.3f} (avg: {top1.avg:.3f}) '.format(
                    epoch, i, len(valid_loader), loss=losses, top1=top1))

        return top1.avg, losses.avg


def train_test(config, model, optimizer, criterion, train_loader, valid_loader):

    best_meas = 0
    plt.ion()
    tr_loss =[]
    tr_acc = []
    ts_loss = []
    ts_acc = []
    all_epoch = []
    for epoch in range(config['epochs']):

        train_acc, train_loss = train(config, model, optimizer, criterion, train_loader, epoch)
        test_acc, test_loss = test(config, model, criterion, valid_loader, epoch)

        tr_loss.append(train_loss)
        tr_acc.append(train_acc)
        ts_loss.append(test_loss)
        ts_acc.append(test_acc)
        all_epoch.append(epoch)

        is_best = test_acc > best_meas
        if is_best:
            best_epoch = epoch
            best_meas = test_acc
            # save_model(config, model, optimizer, epoch, best_meas, best_epoch, is_best,
            #                 './models/{}{}'.format(config['output'], '_checkpoint.pth.tar'))

        print('best valid {:.3f} best epoches {} pred acc {:.3f}'
            .format(best_meas, best_epoch, test_acc))

        plot_function(all_epoch,tr_acc,tr_loss,ts_acc,ts_loss)

    plt.ioff()
    plt.show()


def main():
    config = parser.parse_args()
    config = vars(config)

    # load dataset
    label_file='/home/lj/PycharmProject/CNN/data/FLO/imag_path.txt'
    examples, labels = Datasetloder.image_load(label_file)
    dataset = Datasetloder.split_train_test(examples, labels)

    # grab train and test data
    train_set = Datasetloder.grab_data(config, dataset[0], dataset[1], True)
    test_set = Datasetloder.grab_data(config, dataset[2], dataset[3], True)

    # create model 模型参数初始化

    model = models.__dict__['resnet50'](pretrained=True)
    model = CNNcls.mymodel(config, model)
    model = nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],
                                weight_decay = config['weight_decay'])

    train_test(config, model, optimizer, criterion, train_set, test_set)


if __name__=="__main__":
    main()

