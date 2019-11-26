import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from mpi4py import MPI
import os 
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

DEBUG = True 

# === Training CNN === #
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
        return x
 
def train(model, train_loader, optimizer, criterion, device, pbar_train=None):     
    model.train()
    correct, total = 0, 0
    loss_total = 0
    if pbar_train is not None:
        pbar_train.reset(total=len(train_loader))
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        if DEBUG and batch_idx > 10:
            break
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
    return correct/total
 
def test(model, test_loader, criterion, device, pbar_test=None):
    model.eval()      
    test_loss = 0    
    correct, total = 0, 0
    if pbar_test is not None:
        pbar_test.reset(total=len(test_loader))
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            if DEBUG and batch_idx > 10:
                break
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
    return 100.* (correct/total)


def train_mnist(lr, batch_size, device, pbar_train=None, pbar_test=None):
    # Init Network
    net = Net()
    net.to(device)

    # process = psutil.Process(os.getpid())
    # print("\nPid:", process, "memory:", process.memory_info().rss,"\n")  # in bytes 

    # Data Loader
    train_loader = get_train_loader(batch_size)
    val_loader = get_val_loader(batch_size)

    # Hyper-Parameters
    num_epochs = 1
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, num_epochs+1):
        if pbar_train is not None:
            pbar_train.set_description("[lr={:.4f}, bs={}] Train ({}/{})".format(
                lr, batch_size, epoch, num_epochs))
        train_acc = train(net, train_loader, optimizer, criterion, device, pbar_train)
        if pbar_test is not None:
            pbar_test.set_description( "[lr={:.4f}, bs={}] Test  ({}/{})".format(
                lr, batch_size, epoch, num_epochs))
        test_acc = test(net, val_loader, criterion, device, pbar_test)

    return test_acc

# === Data Loader === #
def get_train_loader(batch_size):
    train_loader = torch.utils.data.DataLoader(       
        datasets.MNIST('./data', train=True, download=True,
               transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
               ])), batch_size=batch_size, shuffle=True)
    return train_loader

def get_val_loader(batch_size):
    test_loader = torch.utils.data.DataLoader(        
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
   ])), batch_size=batch_size, shuffle=True)
    return test_loader

# === Record === #
def save_record(lr, batch_size, acc, save_dir="record/minst"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = "{}_{}.txt".format(lr, batch_size)
    with open(os.path.join(save_dir, file_name), "w") as f:
        f.write("{},{},{}".format(lr, batch_size, acc))

def read_record(lr, batch_size, save_dir="record/minst"):
    file_name = "{}_{}.txt".format(lr, batch_size)
    with open(os.path.join(save_dir, file_name), "r") as f:
        lines = f.readlines()
        line = lines[0]
        lr, batch_size, acc = line.split(",")
        return float(acc)

def clear_records(save_dir="record/minst"):
    files = glob.glob("record/minst/*.txt")
    for file in files:
        os.remove(file)

def read_records(save_dir="record/minst"):
    file_paths = glob.glob(save_dir+"/*.txt")
    keys = []
    values = []
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        lr, batch_size = file_name.split(".txt")[0].split("_")
        acc = read_record(lr, batch_size)
        keys.append((lr, batch_size))
        values.append(acc)
    return dict(zip(keys, values))

# === MPI === #
def initMPI():
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    my_rank = comm.Get_rank()
    node_name = MPI.Get_processor_name()
    print("Process ({1}/{2}) : {0} ".format(node_name, my_rank+1, world_size))
    return comm, world_size, my_rank, node_name


def syncData(dataDict, comm, world_size, my_rank, MASTER_RANK, blocking=False):
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
def vis_search(hyperparams, result=None):
    
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
        plt.savefig('2D_vis.png')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for x, y, z in zip(xs, ys, zs):
            z_ = (z - min(zs)) / ( max(zs) - min(zs) )
            cmap = plt.cm.get_cmap('cool')
            ax.scatter([x], [y], [z], c=[cmap(z_)])
            ax.plot([x, x], [y,y], [z, 0], linestyle=":", c=cmap(z_))
        plt.savefig('3D_vis.png')