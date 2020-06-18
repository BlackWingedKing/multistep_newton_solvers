from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loader import train_loader, test_loader, device
import time, pickle

name = 'adam'

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.drop = nn.Dropout()
        self.drop2 = nn.Dropout2d()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.pool(self.conv1(x)))
        x = self.relu(self.pool(self.conv2(x)))
        x = self.drop2(x).view(x.shape[0], -1)
        x = self.drop(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def train(model, device, train_loader, optimizer, epoch, metrics):
    model.train()
    train_loss, train_acc = 0.0, 0.0
    for i, (data, target) in enumerate(train_loader):
        model.zero_grad()
        data, target = data.to(device), target.to(device)
        pred = model(data)
        loss = F.cross_entropy(pred, target)
        loss.backward()
        optimizer.step()
        pred = pred.max(1, keepdim=True)[1]  # get the index of the max log-probability
        accuracy = pred.eq(target.view_as(pred)).double().mean().item()
        train_loss+=loss.item()
        train_acc+=accuracy
        metrics['batch_loss'].append(loss.item())
        metrics['batch_acc'].append(accuracy)
    train_loss/=len(train_loader)
    train_acc/=len(train_loader)
    return train_loss, train_acc

def test(model, device, test_loader):
    model.eval()
    test_loss, test_acc = 0.0, 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            predictions = model(data)
            loss = F.cross_entropy(predictions, target)
            pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
            accuracy = pred.eq(target.view_as(pred)).double().mean().item()

            test_loss+= loss.item()  # sum up batch loss
            test_acc+= accuracy
        test_acc/=len(test_loader)
        test_loss/=len(test_loader)
        return test_loss, test_acc

def main():
    # Training settings
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.0)
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    metrics = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'batch_loss': [], 'batch_acc':[]}
    for epoch in range(0,25):
        tic = time.time()
        train_loss, train_acc = train(model, device, train_loader, optimizer, epoch, metrics)
        test_loss, test_acc = test(model, device, test_loader)
        metrics['train_loss'].append(train_loss)
        metrics['test_loss'].append(test_loss)
        metrics['train_acc'].append(train_acc)
        metrics['test_acc'].append(test_acc)
        print('completed epoch:', epoch+1,'time:', time.time()-tic,'train_loss:', train_loss, 
                'train_acc:', train_acc, 'test_loss:', test_loss, 'test_acc:', test_acc)
    
    # with open('multi_curveball.pkl','wb') as fp:
    with open(name+'.pkl','wb') as fp:
        pickle.dump(metrics, fp, protocol=pickle.HIGHEST_PROTOCOL)
    fp.close()
    print('saved the metrics')

if __name__ == '__main__':
    main()
