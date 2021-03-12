import numpy as np
import torch
import operator
from functools import reduce
from torchvision import transforms
from torch.utils import data
from image_dataset import DataSet


def image_load(label_file):

    examples = []
    labels = {}
    with open(label_file, 'r') as f:
        image_label = [l.strip() for l in f.readlines()]
    for lines in image_label:
        items = lines.split()
        examples.append(items[0])
        labels[items[0]] = int(items[1])
    return examples, labels


def split_train_test(examples, labels):
    """
    :param examples:
    :param labels:
    :return:
    """
    train_examples = []
    test_examples = []
    train_labels = {}
    test_labels = {}

    for i in range(17):
        if i == 0:
            train_examples.append(examples[:70])
            test_examples.append(examples[70:80])
        else:
            train_examples.append(examples[i*70+10:(i+1)*70+10])
            test_examples.append(examples[(i+1)*80-10:(i+1)*80])

    train_examples=reduce(operator.add, train_examples)
    test_examples=reduce(operator.add, test_examples)


    for i in range (len(train_examples)):
        train_labels[train_examples[i]] = labels[train_examples[i]]

    for i in range (len(test_examples)):
        test_labels[test_examples[i]] = labels[test_examples[i]]

    return [train_examples, train_labels, test_examples, test_labels]

def grab_data(config, examples, labels, is_train = True):


    params = {'batch_size': config['batchsize'],
              'num_workers': 4,
              'pin_memory': True,
              'sampler': None}
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    tr_transforms, ts_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]), transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    if is_train:
        params['shuffle'] = True
        params['sampler'] = None
        data_set = data.DataLoader(DataSet(config, examples, labels, tr_transforms, is_train), **params)
    else:
        params['shuffle'] = False
        data_set = data.DataLoader(DataSet(config, examples, labels,ts_transforms, is_train), **params)

    return data_set