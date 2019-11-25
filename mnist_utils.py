import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from mpi4py import MPI
import os 
import glob

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

