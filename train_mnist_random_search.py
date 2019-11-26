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
from argparse import ArgumentParser

# Global Variable 
MASTER_RANK = 0
device = torch.device("cpu") 

if __name__ == "__main__":
    start = time.time()

    # ArgParser (Setting)
    parser = ArgumentParser()
    parser.add_argument("-remote", default=False)
    parser.add_argument("-n", "--num_search", dest="n", default=8)
    parser.add_argument("-lr_low", "--lr_low",  default=0)
    parser.add_argument("-lr_high", "--lr_high", default=0.01)
    args = parser.parse_args()

    num_search_total = int(args.n)
    lr_low = float(args.lr_low)
    lr_high = float(args.lr_high)
    batch_size_list = [16, 32, 64, 128, 256]
    
    # === Init MPI World === #
    comm, world_size, my_rank, node_name = initMPI()

    # === Init Search Space === #
    var_lr = continuousRandomVariable(lr_low, lr_high )
    var_batch_size = discreteRandomVariable(batch_size_list)
    randomSearch = RandomSearch([var_lr, var_batch_size], ["learning rate", "batch size"])

    comm.Barrier()
    # === Init Progress Bar === #
    if my_rank == MASTER_RANK:
        print("=== Random Search ===")
        print(randomSearch)
        pbar_search  = tqdm(total=num_search_total, desc="Random Search")
        if not args.remote:
            pbar_train = tqdm(desc="Train (Master)")
            pbar_test  = tqdm(desc="Test  (Master)")
        else:
            pbar_train = None
            pbar_test  = None
        num_search = 0
    else:
        pbar_search  = None
        pbar_train = None
        pbar_test  = None

    # === Get Indexs === #
    start_idx = my_rank * (num_search_total/world_size) if my_rank != 0 else 0
    end_idx = (my_rank+1) * (num_search_total/world_size) if my_rank != world_size-1 else num_search_total
    idxes = [i for i in range(int(start_idx), int(end_idx))]
    resultDict = {}
    # === Start Search === #
    for idx in idxes:

        # Get Hyper-Parameters
        lr, batch_size = randomSearch.get()

        # Train MNIST
        acc = train_mnist(lr, batch_size, device, pbar_train, pbar_test)

        # Sync Data
        resultDict[(lr, batch_size)] = acc
        resultDict = syncData(resultDict, comm, world_size, my_rank, MASTER_RANK, blocking=True)

        # Update Grid Search Progress bar
        if my_rank == MASTER_RANK:
            pbar_search.update(len(resultDict)-pbar_search.n)
    resultDict = syncData(resultDict, comm, world_size, my_rank, MASTER_RANK, blocking=True)
    if my_rank == MASTER_RANK:
        pbar_search.update(len(resultDict)-pbar_search.n)
    comm.Barrier()
    if my_rank == MASTER_RANK:
        # Close Progress Bar 
        pbar_search.close()
        if not args.remote:
            pbar_train.close()
            pbar_test.close()

        # Read & Print Grid Search Result
        # records = read_records()
        records = resultDict
        hyperparams = []
        results = []
        print("\n=== Random Search Result ===\n")
        print("=" * len("{:^15} | {:^10} | {:^10} |".format("learning rate", "batch_size", "acc")))
        print("{:^15} | {:^10} | {:^10} |".format("learning rate", "batch_size", "acc"))
        print("=" * len("{:^15} | {:^10} | {:^10} |".format("learning rate", "batch_size", "acc")))
        for (lr, batch_size), acc in records.items():
            print("{:^15} | {:^10} | {:^10} |".format("%.4f"%float(lr), batch_size, "%.4f"%float(acc)))
            hyperparams.append((float(lr), int(batch_size)))
            results.append(float(acc))
        print("=" * len("{:^15} | {:^10} | {:^10} |".format("learning rate", "batch_size", "acc")))
        vis_search(hyperparams,results)

        # Print Execution Time
        end = time.time()
        print("Execution Time:", end-start)
        