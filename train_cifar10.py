from __future__ import print_function  #这个是python当中让print都以python3的形式进行print，即把print视为函数
import argparse  # 使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
import torch     # 以下这几行导入相关的pytorch包，有疑问的参考我写的 Pytorch打怪路（一）系列博文
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Lock, Manager, Semaphore
from multiprocessing.managers import BaseManager, NamespaceProxy
import torchvision.models as models

def get_train_loader(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True)
    return train_loader

def get_val_loader(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False)
    return test_loader

class Model:
    def __init__(self, lr, batch_size, device_ID, model_ID="", epochs=1):
        self.net = Net()
        self.net.cuda(device_ID)
        self.lr = lr
        self.batch_size = batch_size
        self.model_ID = model_ID

  # 初始化优化器 model.train() 
def train(model, train_loader, optimizer, device):      # 定义每个epoch的训练细节
    #pbar = tqdm(total=len(train_loader), ascii=True, desc=str(device))
    correct, total = 0, 0
    model.train()      # 设置为trainning模式
    criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()  # 优化器梯度初始化为零
        outputs = model(inputs)   # 把数据输入网络并得到输出，即进行前向传播
        loss = criterion(outputs, labels) 
        loss.backward()        # 反向传播梯度
        optimizer.step()       # 结束一次前传+反传之后，更新优化器参数

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum().item()

        #pbar.set_postfix({"Loss":loss.data.item(), "Acc":correct/total})
        #pbar.update()
    #pbar.close()
 
def test(model, test_loader, device):
    model.eval()      # 设置为test模式  
    test_loss = 0     # 初始化测试损失值为0
    total, correct = 0, 0
    criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item() 
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum().item()
    test_loss /= len(test_loader.dataset)   # 因为把所有loss值进行过累加，所以最后要除以总得数据长度才得平均loss
    return correct / total

def save_txt(lr, batch_size, acc, save_dir="record"):
    import os 
    file_name = "{}_{}.txt".format(lr, batch_size)
    with open(os.path.join(save_dir, file_name), "w") as f:
        f.write("{},{},{}".format(lr, batch_size, acc))

def read_txt(lr, batch_size, save_dir="record"):
    import os
    file_name = "{}_{}.txt".format(lr, batch_size)
    with open(os.path.join(save_dir, file_name), "r") as f:
        lines = f.readlines()
        line = lines[0]
        lr, batch_size, acc = line.split(",")
    return float(acc)

def run(device_ID, lr, batch_size):
    device = torch.device('cuda:{}'.format(device_ID))
    net = models.resnet18()
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    train_loader = get_train_loader(batch_size)
    val_loader = get_val_loader(batch_size)
    acc = 0
    for epoch in range(5):
        train(net, train_loader, optimizer, device)
        acc = test(net, val_loader, device)
    # print("Device:{0} lr={1}, batch_size={2}, Epochs:{3}, get {4:.2f}%".format(device_ID, lr, batch_size, epoch, acc))
    save_txt(lr, batch_size, acc)

class Grid:
    def __init__(self, list_1, list_2):
        self.list_1 = list_1
        self.list_2 = list_2
        self.params_grid = []
        for i in range(len(list_1)):
            self.params_grid.append( [(list_1[i], list_2[j]) for j in range(len(self.list_2))] )
        self.result_grid = [ [0.0 for j in range(len(self.list_2))] for i in range(len(self.list_1)) ]

    def __len__(self):
        return len(self.list_1) * len(self.list_2)

    def get(self, idx):
        h = len(self.list_1)
        w = len(self.list_2)
        i, j = int(idx / w), int(idx % w)
        return self.params_grid[i][j]

    def read_records(self):
        for i in range(len(self.list_1)):
            for j in range(len(self.list_2)):
                lr, batch_size = self.params_grid[i][j]
                self.result_grid[i][j] = "{0:.2f}".format(read_txt(lr, batch_size))

    def __str__(self):
        s = " === Grid Search ===\n"
        for j in range(len(self.list_2)):
            if j == 0:
                s += "{:5} ||".format("")
            s += " {0:^5} |".format(self.list_2[j])
            if j == len(self.list_2)-1:
                s += "\n"
        s+= "="*len(s.split("\n")[-2]) + "\n"
        for i in range(len(self.list_1)):
            for j in range(len(self.list_2)):
                if j == 0:
                    s += "{:5} ||".format(self.list_1[i] if self.list_1[i] != 0.0 else "NaN")
                s += " {0:^5} |".format(self.result_grid[i][j])
                if j == len(self.list_2)-1:
                    s += "\n"
        s+= "="*len(s.split("\n")[-2]) + "\n"
        return s

if __name__ == "__main__":
    import time 
    start = time.time()
    n_gpu = 2
    
    lr_list = [(i/100) for i in range(1, 10)][-2:]
    batch_size_list = [16, 32, 64, 128, 256][-2:]
    grid = Grid(lr_list, batch_size_list)
    print(grid)

    cnt = 0
    pbar = tqdm(total = len(grid), ascii=True)
    while cnt < len(grid):
        Processes = []
        for device_ID in range(n_gpu):
            if cnt >= len(grid):
                break
            lr, batch_size = grid.get(cnt)
            cnt += 1
            p = mp.Process(target = run, args = (device_ID, lr, batch_size))
            Processes.append(p)
            p.start()

        for p in Processes:
            p.join()
            pbar.update()
    grid.read_records()
    print(grid)
    end = time.time()
    print("Execution Time:", end-start)