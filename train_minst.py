from __future__ import print_function  #这个是python当中让print都以python3的形式进行print，即把print视为函数
import argparse  # 使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
import torch     # 以下这几行导入相关的pytorch包，有疑问的参考我写的 Pytorch打怪路（一）系列博文
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from tqdm import tqdm
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Lock, Manager, Semaphore
from multiprocessing.managers import BaseManager, NamespaceProxy

import time
def get_train_loader(batch_size):
    train_loader = torch.utils.data.DataLoader(       # 加载训练数据，详细用法参考我的Pytorch打怪路（一）系列-（1）
        datasets.MNIST('./data', train=True, download=True,
               transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
               ])), batch_size=batch_size, shuffle=True)
    return train_loader

def get_val_loader(batch_size):
    test_loader = torch.utils.data.DataLoader(        # 加载训练数据，详细用法参考我的Pytorch打怪路（一）系列-（1）
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
   ])), batch_size=batch_size, shuffle=True)
    return test_loader

class Model:
    def __init__(self, lr, batch_size, device_ID, model_ID="", epochs=1):
        self.net = Net()
        self.net.cuda(device_ID)
        self.lr = lr
        self.batch_size = batch_size
        self.model_ID = model_ID

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
 
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
 
  # 初始化优化器 model.train() 
def train(model, train_loader, optimizer, device):      # 定义每个epoch的训练细节
    # pbar = tqdm(total=len(train_loader), ascii=True, desc=str(device))
    model.train()      # 设置为trainning模式
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data, target = Variable(data), Variable(target)  # 把数据转换成Variable
        optimizer.zero_grad()  # 优化器梯度初始化为零
        output = model(data)   # 把数据输入网络并得到输出，即进行前向传播
        loss = F.nll_loss(output, target)               # 计算损失函数  
        loss.backward()        # 反向传播梯度
        optimizer.step()       # 结束一次前传+反传之后，更新优化器参数
        #pbar.set_postfix({"Loss":loss.data.item()})
        #pbar.update()
    #pbar.close()
 
def test(model, test_loader, device):
    model.eval()      # 设置为test模式  
    test_loss = 0     # 初始化测试损失值为0
    correct = 0       # 初始化预测正确的数据个数为0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item() # sum up batch loss 把所有loss值进行累加
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()  # 对预测正确的数据个数进行累加
 
    test_loss /= len(test_loader.dataset)   # 因为把所有loss值进行过累加，所以最后要除以总得数据长度才得平均loss
    return 100. * correct / len(test_loader.dataset)

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
    net = Net()
    net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.5)
    train_loader = get_train_loader(batch_size)
    val_loader = get_val_loader(batch_size)
    train(net, train_loader, optimizer, device)
    acc = test(net, val_loader, device)
    # print("Device:{0} lr={1}, batch_size={2}, get {3:.2f}%".format(device_ID, lr, batch_size, acc))
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
    start = time.time()
    n_gpu = 2
    
    lr_list = [(i/10) for i in range(1, 10)]
    batch_size_list = [16, 32, 64, 128, 256]
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