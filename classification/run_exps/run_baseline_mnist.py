# LeNet-5, MLP (noisy data, clean data)

import time
import random
import pdb
import torch
import torch.nn as nn
import numpy as np
from functools import partial
from torchvision import datasets, transforms


import argparse
parser = argparse.ArgumentParser(description='GPT')
parser.add_argument("-m", "--model_name", default='lenet', type=str,choices=['lenet','mlp'])
#parser.add_argument("-g", "--gpu_id", default=0, type=int)
parser.add_argument("-e", "--eps", default=0, type=float)
parser.add_argument("-t", "--type", default='const', type=str, choices=['const', 'unif', 'normal', 'sign'])
parser.add_argument("-v", "--eval", default=0, type=int)
args = parser.parse_args()
print(args)


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        #pdb.set_trace()
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

class MLP(nn.Module):
    def __init__(self, num_classes=10):
        super(MLP, self).__init__()
        self.fc = nn.Linear(1024, 300)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(300, 300)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(300, 100)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(100, num_classes)
        
    def forward(self, x):
        out = x.reshape(x.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


def add_noise(img, noise_type, eps):
    #pdb.set_trace()
    if noise_type == 'const':
        noise = eps
    elif noise_type == 'unif':
        noise = torch.rand_like(img) * 2 * eps - eps
    elif noise_type == 'normal':
        noise = torch.rand_like(img)
        max_val = noise.abs().max()
        noise = noise/max_val * eps
        print(noise.abs().max())
    elif noise_type == 'sign':
        noise = eps * (2 * torch.bernoulli(torch.ones_like(img)*0.5) - 1)        
        #print(noise.abs().max())
        #print(noise)

    noisy_img = torch.clamp(img + noise, min=0, max=1) #noisy_img = torch.clamp(noise, min=0, max=1) # for sanity check
    #print(torch.max(noisy_img - img))

    return noisy_img

def load_mnist():
   
    transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        #transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST('data', train=True, download=True,
                       transform=transform)
    test_set = datasets.MNIST('data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)

    X_train, y_train, X_test, y_test = [], [], [], []
    for batch_idx, (img, label) in enumerate(train_loader):
        X_train.append(img)
        y_train.append(label)
        #print(batch_idx, img.shape, label.shape)
    for batch_idx, (img, label) in enumerate(test_loader):
        X_test.append(img)
        y_test.append(label)

    X_train = torch.cat(X_train).cuda()    
    y_train = torch.cat(y_train).cuda()
    X_test = torch.cat(X_test).cuda()
    y_test = torch.cat(y_test).cuda()    

    return X_train, y_train, X_test, y_test, train_loader, test_loader

def train(model, train_loader, name='lenet'):
    model.train()

    criterion = nn.CrossEntropyLoss()
    if name.lower() == 'lenet':
        optimizer = torch.optim.SGD(
                        model.parameters(), 
                        momentum=0.9, 
                        lr=0.05, 
                        weight_decay = 0.0001
                    )
    elif name.lower() == 'mlp':
        optimizer = torch.optim.Adam(
                        model.parameters(), 
                        lr=0.0001, 
                        weight_decay = 0.0001
                    )
    
    epochs = 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    for epoch in range(epochs):
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()        
            optimizer.zero_grad()
        scheduler.step()

        print('epoch: {}, loss: {:.2f}'.format(epoch, loss.item()))
    
    return model

def test(model, test_loader, eps=0):
    model.eval()
    correct, nelement = 0, 0
    for batch_idx, (X, y) in enumerate(test_loader):
        if eps != 0:
            X = add_noise(X, args.type, eps)
        X, y = X.cuda(), y.cuda()
        pred = model(X)    
        correct += (torch.argmax(pred, axis=1) == y).sum().item()
        nelement += pred.shape[0]

    acc = correct/nelement
    print('Acc: {:.2f}'.format(acc*100))

##### Setting ########
random.seed(12345)
data_name = 'mnist'
model_name = args.model_name #'lenet' # 'mlp'
eps = args.eps  # 0 # noise

# load data (possibly noisy) & model
#print(model_name, data_name)
if model_name.lower() == 'lenet':
    model = LeNet5().cuda()
elif model_name.lower() == 'mlp':
    model = MLP().cuda()
else:
    raise NotImplementedError
X_train, y_train, X_test, y_test, train_loader, test_loader = load_mnist()
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# train a model & test
PATH = f'{model_name}_mnist.pth'
if args.eval:
    model.load_state_dict(torch.load(PATH))
    model.eval()
else:
    model = train(model, train_loader)
    torch.save(model.state_dict(), PATH)
test(model, train_loader)
test(model, test_loader, eps=eps)




