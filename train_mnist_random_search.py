from __future__ import print_function  
import torch     
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from tqdm import tqdm
from mpi4py import MPI
import time
import os 
import glob

from randomSearch import *
from mnist_utils import *

import psutil

# Global Variable 
MASTER_RANK = 0
device = torch.device("cpu") 

# Setting
num_search_total = 100
lr_low, lr_high = 0, 0.1
batch_size_list = [16, 32, 64, 128]



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
 
def train(model, train_loader, optimizer, criterion, pbar_train=None):     
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
    return correct/total
 
def test(model, test_loader, criterion, pbar_test=None):
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
    return 100.* (correct/total)


def train_mnist(lr, batch_size, pbar_train=None, pbar_test=None):
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
            pbar_test.set_description( "[lr={:.4f}, bs={}] Test  ({}/{})".format(
                lr, batch_size, epoch, num_epochs))
        train_acc = train(net, train_loader, optimizer, criterion, pbar_train)
        test_acc = test(net, val_loader, criterion, pbar_test)

    return test_acc

if __name__ == "__main__":
    start = time.time()

    # === Init MPI World === #
    comm, world_size, my_rank, node_name = initMPI()

    # === Init Search Grid === #
    var_lr = continuousRandomVariable(lr_low, lr_high )
    var_batch_size = discreteRandomVariable(batch_size_list)
    randomSearch = RandomSearch([var_lr, var_batch_size], ["learning rate", "batch size"])

    # === Init Progress Bar === #
    if my_rank == MASTER_RANK:
        clear_records()
        print("=== Random Search ===")
        print(randomSearch)
        pbar_search  = tqdm(total=num_search_total, desc="Random Search")
        pbar_train = tqdm(desc="Train (Master)")
        pbar_test  = tqdm(desc="Test  (Master)")
        num_search = 0
    else:
        pbar_grid  = None
        pbar_train = None
        pbar_test  = None

    # === Get Indexs === #
    start_idx = my_rank * (num_search_total/world_size) if my_rank != 0 else 0
    end_idx = (my_rank+1) * (num_search_total/world_size) if my_rank != world_size-1 else num_search_total
    idxes = [i for i in range(int(start_idx), int(end_idx))]

    # === Start Search === #
    for idx in idxes:
        # Get Hyper-Parameters
        lr, batch_size = randomSearch.get()
        # Train MNIST
        acc = train_mnist(lr, batch_size, pbar_train, pbar_test)
        # Save Record
        save_record(float("%.4f"%(lr)), batch_size, acc)
        # Update Grid Search Progress bar
        if my_rank == MASTER_RANK:
            files = glob.glob("record/minst/*.txt")
            cnt = len(files)
            pbar_search.update(cnt-num_search)
            num_search = cnt

    comm.Barrier()
    if my_rank == MASTER_RANK:
        # Close Progress Bar 
        pbar_search.close()
        pbar_train.close()
        pbar_test.close()

        # Read & Print Grid Search Result
        records = read_records()
        hyperparams = []
        print("\n=== Random Search Result ===\n")
        print("=" * len("{:^15} | {:^10} | {:^10} |".format("learning rate", "batch_size", "acc")))
        print("{:^15} | {:^10} | {:^10} |".format("learning rate", "batch_size", "acc"))
        print("=" * len("{:^15} | {:^10} | {:^10} |".format("learning rate", "batch_size", "acc")))
        for (lr, batch_size), acc in records.items():
            print("{:^15} | {:^10} | {:^10} |".format("%.4f"%float(lr), batch_size, "%.4f"%float(acc)))
            hyperparams.append((float(lr), int(batch_size), float(acc)))
        print("=" * len("{:^15} | {:^10} | {:^10} |".format("learning rate", "batch_size", "acc")))
        vis_random_search(hyperparams)

        # Print Execution Time
        end = time.time()
        print("Execution Time:", end-start)
        