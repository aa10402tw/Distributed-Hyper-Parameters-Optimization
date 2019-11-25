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

from gridSearch import *
from mnist_utils import *

MASTER_RANK = 0
device = torch.device("cpu")

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

    # Data Loader
    train_loader = get_train_loader(batch_size)
    val_loader = get_val_loader(batch_size)

    # Hyper-Parameters
    num_epochs = 1
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, num_epochs+1):
        if pbar_train is not None:
            pbar_train.set_description("[lr={}, bs={}] Train ({}/{})".format(
                lr, batch_size, epoch, num_epochs))
            pbar_test.set_description( "[lr={}, bs={}] Test  ({}/{})".format(
                lr, batch_size, epoch, num_epochs))
        train_acc = train(net, train_loader, optimizer, criterion, pbar_train)
        test_acc = test(net, val_loader, criterion, pbar_test)

    return test_acc

if __name__ == "__main__":
    start = time.time()

    # === Init MPI World === #
    comm, world_size, my_rank, node_name = initMPI()

    # === Init Search Grid === #
    lr_list = [(i/10) for i in range(1, 10)][:3]
    batch_size_list = [16, 32, 64, 128, 256][-3:]
    grid = Grid(lr_list, batch_size_list)

    # === Init Progress Bar === #
    if my_rank == MASTER_RANK:
        clear_records()
        print("=== Grid Search ===")
        print(grid)
        pbar_grid  = tqdm(total=len(grid), desc="Grid Search")
        pbar_train = tqdm(desc="Train (Master)")
        pbar_test  = tqdm(desc="Test  (Master)")
        num_search = 0
    else:
        pbar_grid  = None
        pbar_train = None
        pbar_test  = None

    # === Get Indexs === #
    start_idx = my_rank * (len(grid)/world_size) if my_rank != 0 else 0
    end_idx = (my_rank+1) * (len(grid)/world_size) if my_rank != world_size-1 else len(grid)
    idxes = [i for i in range(int(start_idx), int(end_idx))]

    # === Start Search === #
    for idx in idxes:
        # Get Hyper-Parameters
        lr, batch_size = grid.get(idx)
        # Train MNIST
        acc = train_mnist(lr, batch_size, pbar_train, pbar_test)
        # Save Record
        save_record(lr, batch_size, acc)
        # Update Grid Search Progress bar
        if my_rank == MASTER_RANK:
            files = glob.glob("record/minst/*.txt")
            cnt = len(files)
            pbar_grid.update(num_search-cbt)
            num_search = cnt

    comm.Barrier()
    if my_rank == MASTER_RANK:
        # Close Progress Bar 
        pbar_grid.close()
        pbar_train.close()
        pbar_test.close()

        # Read Grid Search Result
        grid.read_records()
        print("=== Grid Search Result ===")
        print(grid)
        clear_records()

        # Print Execution Time
        end = time.time()
        print("Execution Time:", end-start)
        