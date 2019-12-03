import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import random

from mpi4py import MPI
import os 
import glob
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from hyperparams import *

DEBUG = False
bs_default = 256
mmt_default = 0.0
dr_default = 0.0
lr_default = 0.1
num_epochs = 1

# === Training CNN === #
class Net(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=5)
        self.conv2 = nn.Conv2d(2, 4, kernel_size=5)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(4*4*4, 10)
 
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 4*4*4)
        x = self.dropout(x)
        x = self.fc(x)
        return x
 
def train(model, train_loader, optimizer, criterion, device, 
    pbar_train=None, DEBUG=False):     
    model.train()
    correct, total = 0, 0
    loss_total = 0
    if pbar_train is not None:
        pbar_train.reset(total=len(train_loader))
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum().item() 
        loss_total += loss.item() 
        # Progress Bar
        if pbar_train is not None:
            pbar_train.set_postfix({"Loss":loss_total/(batch_idx+1), "Acc":correct/total})
            pbar_train.update()
        if DEBUG and batch_idx > 5:
            break
    return correct/total
 
def test(model, test_loader, criterion, device, 
    pbar_test=None, DEBUG=False):
    model.eval()      
    test_loss = 0    
    correct, total = 0, 0
    if pbar_test is not None:
        pbar_test.reset(total=len(test_loader))
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()
            # Progress Bar
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum().item()
            if pbar_test is not None:
                pbar_test.set_postfix({"Loss":test_loss/(batch_idx+1), "Acc":correct/total})
                pbar_test.update() 
            if DEBUG and batch_idx > 5:
                break
    return 100.* (correct/total)


def train_mnist(hparams, device=None, pbars=None, DEBUG=False):
    # Set up Hyper-parameters
    bs = bs_default 
    mmt = mmt_default
    dr = dr_default
    lr = lr_default

    for rv in hparams:
        if rv.name == LEARNING_RATE_NAME:
            lr = rv.value
        elif rv.name == DROPOUT_RATE_NAME:
            dr = rv.value
        elif rv.name == MOMENTUM_NAME:
            mmt = rv.value
        elif rv.name == BATCH_SIZE_NAME:
            bs = rv.value

    if pbars is not None:
        pbar_train = pbars['train']
        pbar_test = pbars['test']
    else:
        pbar_train = None
        pbar_test = None

    # Reproducibitity
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    # np.random.seed(0)
    # random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

     # Init Network
    net = Net(dropout_rate=dr)
    #net.load_state_dict(torch.load("MnistNet.pth"))
    net.to(device)

    # Data Loader
    train_loader = get_train_loader(bs)
    val_loader = get_val_loader(bs)

    # Hyper-Parameters
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=mmt)
    # optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, num_epochs+1):
        # Train
        if pbar_train is not None:
            pbar_train.set_description("[lr={:.4f}, dr={:.4f}] Train ({}/{})".format(
                lr, dr, epoch, num_epochs))
        train_acc = train(net, train_loader, optimizer, criterion, 
            device, pbar_train, DEBUG=DEBUG)
        # Test
        if pbar_test is not None:
            pbar_test.set_description( "[lr={:.4f}, dr={:.4f}] Test  ({}/{})".format(
                lr, dr, epoch, num_epochs))
        test_acc = test(net, val_loader, criterion, 
            device, pbar_test, DEBUG=DEBUG)

    return test_acc

# === Data Loader === #
def get_train_loader(batch_size):
    train_loader = torch.utils.data.DataLoader(       
        datasets.MNIST('./data', train=True, download=True,
               transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
               ])), batch_size=batch_size, shuffle=False)
    return train_loader

def get_val_loader(batch_size):
    test_loader = torch.utils.data.DataLoader(        
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
   ])), batch_size=batch_size, shuffle=True)
    return test_loader

# === MPI === #
def initMPI():
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    my_rank = comm.Get_rank()
    node_name = MPI.Get_processor_name()
    return MPIWorld(comm, world_size, my_rank, MASTER_RANK=0, node_name=node_name)

class MPIWorld:
    def __init__(self, comm, world_size, my_rank, MASTER_RANK=0, node_name=""):
        self.comm = comm
        self.world_size = world_size
        self.my_rank = my_rank
        self.MASTER_RANK = MASTER_RANK
        self.node_name = node_name
        self.log = "Process ({1}/{2}) : {0} ".format(node_name, my_rank+1, world_size)
        
    def isMaster(self):
        return self.my_rank == self.MASTER_RANK

def initPbars(mpiWorld, exp=False):

    if mpiWorld.isMaster():
        print("=== [Progress Bars] ===")
        if exp:
            return {"search":tqdm(), "train":None, "test":None}
        else:
            return {"search":tqdm(), "train":tqdm(), "test":tqdm()}
    else:
        return {"search":None, "train":None, "test":None}

def closePbars(pbars):
    if pbars["search"] is not None:
        pbars["search"].close()
    if pbars["train"] is not None:
        pbars["train"].close()
    if pbars["test"] is not None:
        pbars["test"].close()

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_best_acc(resultDict):
    best_acc = 0.0
    for key, acc in resultDict.items():
        best_acc = max(best_acc, acc)
    return best_acc

def syncData(dataDict, mpiWorld, blocking=False):
    my_rank = mpiWorld.my_rank
    world_size = mpiWorld.world_size
    MASTER_RANK = mpiWorld.MASTER_RANK
    comm = mpiWorld.comm

    # Non-Blocking
    if not blocking:
        if my_rank == MASTER_RANK:
            for i in range(world_size):
                if i == MASTER_RANK:
                    continue
                req = comm.irecv(source=i)
                flag, dataDictReceived = req.test()
                if(flag): 
                    for key, value in dataDictReceived.items():
                        if key not in dataDict:
                            # print(key, value)
                            dataDict[key] = value
        else:
            # print(my_rank, dataDict)
            comm.send(dataDict, dest=MASTER_RANK)
    # Blocking
    else:
        if my_rank == MASTER_RANK:
            for i in range(1, world_size):
                if i == MASTER_RANK:
                    continue
                dataDictReceived = comm.recv(source=i)
                for key, value in dataDictReceived.items():
                    dataDict[key] = value
        else:
            comm.send(dataDict, dest=MASTER_RANK)
    return dataDict

# === Visualization Search Result === #
def vis_search(hyperparams, result=None, save_name=""):
    
    if result is None:
        xs = [h[0] for h in hyperparams]
        ys = [h[1] for h in hyperparams]
        plt.scatter(xs, ys)
        if type(xs[0]) == type(0.0):
            plt.xticks([min(xs), max(xs)])
        elif type(xs[0]) == type(0):
            plt.xticks(list(set(xs)))
        if type(ys[0]) == type(0.0):
            plt.yticks([min(ys), max(ys)])
        elif type(ys[0]) == type(0):
            plt.yticks(list(set(ys)))
        plt.show()

    else:
        xs = [h[0] for h in hyperparams]
        ys = [h[1] for h in hyperparams]
        zs = [r for r in result]

        plt.scatter(xs, ys, c=zs, cmap='cool')
        if type(xs[0]) == type(0.0):
            plt.xticks([min(xs), max(xs)])
        elif type(xs[0]) == type(0):
            plt.xticks(list(set(xs)))
        if type(ys[0]) == type(0.0):
            plt.yticks([min(ys), max(ys)])
        elif type(ys[0]) == type(0):
            plt.yticks(list(set(ys)))
        # plt.clim(0, 100)
        plt.colorbar()
        plt.savefig('{}_2D_vis.png'.format(save_name))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for x, y, z in zip(xs, ys, zs):
            z_ = (z - min(zs)) / ( max(zs) - min(zs) )
            cmap = plt.cm.get_cmap('cool')
            ax.scatter([x], [y], [z], c=[cmap(z_)])
            ax.plot([x, x], [y,y], [z, 0], linestyle=":", c=cmap(z_))
        plt.savefig('{}_3D_vis.png'.format(save_name))