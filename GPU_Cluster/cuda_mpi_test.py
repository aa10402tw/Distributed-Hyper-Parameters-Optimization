import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import random

import mpi4py
from mpi4py import MPI

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class MPIWorld:
    def __init__(self, comm, world_size, my_rank, MASTER_RANK=0, node_name=""):
        self.comm = comm
        self.world_size = world_size
        self.my_rank = my_rank
        self.MASTER_RANK = MASTER_RANK
        self.node_name = node_name
        
    def isMaster(self):
        return self.my_rank == self.MASTER_RANK

def initMPI():
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    my_rank = comm.Get_rank()
    node_name = MPI.Get_processor_name()
    print("Process ({1}/{2}) : {0} ".format(node_name, my_rank+1, world_size))
    return MPIWorld(comm, world_size, my_rank, MASTER_RANK=0, node_name=node_name)

if __name__ == "__main__":
    mpiWorld = initMPI()
    print(torch.cuda.is_available())
    net = Net()
    net.to(device)
    print("Rank:{} Device:{} Network:\n{}".format(mpiWorld.my_rank, device, net))
    print("\n")
