"""
    Code used for loading the datasets 
    returns a normal train and test data loaders
"""
# imports
from __future__ import print_function
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision import models
import numpy as np
import math


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# save files
test_loss_list,test_acc_list,train_loss_list, train_acc_list,c_list,norm_list,connect_list = [],[],[],[],[],[],[]

data = 'mnist'

def prepare_loader(x,y,n):
    """
        Splits the tensors into a list of size n 
        and then appends them into a list like dataloader
    """
    ind = torch.randperm(x.shape[0])
    x = x[ind]
    y = y[ind]
    x_l = torch.chunk(x,n)
    y_l = torch.chunk(y,n)
    loader = []
    for i in range(0,len(x_l)):
        loader.append([x_l[i],y_l[i]])
    return loader

splitsize = 10

if (data == 'cifar'):
    # for cifar
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(root='/home/ritesh/Desktop/datasets/cdata', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=50000,
                                            shuffle=True, num_workers=4)

    testset = datasets.CIFAR10(root='/home/ritesh/Desktop/datasets/cdata', train=False,
                                        download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=10000,
                                            shuffle=False, num_workers=2)
elif(data == 'mnist'):
    # for mnist
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/home/ritesh/Desktop/datasets/mdata', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=64, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/home/ritesh/Desktop/datasets/mdata', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=10000, shuffle=True, **kwargs)

elif(data == 'a4a'):
    train_x = torch.load('../logistic_data/a4a_train_x.pt')
    train_y = torch.load('../logistic_data/a4a_train_y.pt')
    test_x = torch.load('../logistic_data/a4a_test_x.pt')
    test_y = torch.load('../logistic_data/a4a_test_y.pt')
    if(splitsize == 1):
        train_loader = [[train_x, train_y]]
    else:
        train_loader = prepare_loader(train_x,train_y, splitsize)
    test_loader = [[test_x, test_y]]

elif(data == 'ijcnn'):
    train_x = torch.load('../logistic_data/ijcnn1_train_x.pt')
    train_y = torch.load('../logistic_data/ijcnn1_train_y.pt')
    test_x = torch.load('../logistic_data/ijcnn1_test_x.pt')
    test_y = torch.load('../logistic_data/ijcnn1_test_y.pt')
    if(splitsize == 1):
        train_loader = [[train_x, train_y]]
    else:
        train_loader = prepare_loader(train_x,train_y, splitsize)
    test_loader = [[test_x, test_y]]

print('created the data loaders')

# debugging codes just leave them commented
# Train data size is 60k and test data size is 10k
# The dataset is loading properly
# print(len(train_loader), len(test_loader))
