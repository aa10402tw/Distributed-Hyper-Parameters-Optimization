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
from argparse import ArgumentParser
from gridSearch import *
from mnist_utils import *

MASTER_RANK = 0
device = torch.device("cpu")

if __name__ == "__main__":
    start = time.time()

    # === Init MPI World === #
    comm, world_size, my_rank, node_name = initMPI()

    parser = ArgumentParser()
    parser.add_argument("-remote", default=False)
    args = parser.parse_args()

    # === Init Search Grid === #
    lr_list = [(i/10) for i in range(1, 10+1)]
    batch_size_list = [16, 32, 64, 128, 256]
    grid = Grid(lr_list, batch_size_list)

    # === Init Progress Bar === #
    if my_rank == MASTER_RANK:
        print("=== Grid Search ===")
        print(grid)
        pbar_search  = tqdm(total=len(grid), desc="Grid Search")
        if not args.remote:
            pbar_train = tqdm(desc="Train (Master)")
            pbar_test  = tqdm(desc="Test  (Master)")
        else:
            pbar_train = None
            pbar_test  = None
    else:
        pbar_search  = None
        pbar_train = None
        pbar_test  = None

    # === Get Indexs === #
    start_idx = my_rank * (len(grid)/world_size) if my_rank != 0 else 0
    end_idx = (my_rank+1) * (len(grid)/world_size) if my_rank != world_size-1 else len(grid)
    idxes = [i for i in range(int(start_idx), int(end_idx))]
    resultDict = {}

    # === Start Search === #
    for idx in idxes:
        # Get Hyper-Parameters
        lr, batch_size = grid.get(idx)

        # Train MNIST
        acc = train_mnist(lr, batch_size, device, pbar_train, pbar_test)
        
        # Sync Data
        resultDict[(lr, batch_size)] = acc
        resultDict = syncData(resultDict, comm, world_size, my_rank, MASTER_RANK, blocking=True)

        # Update Grid Search Progress bar
        if my_rank == MASTER_RANK:
            pbar_search.update(len(resultDict)-pbar_search.n)

    comm.Barrier()
    if my_rank == MASTER_RANK:
        # Close Progress Bar 
        pbar_search.close()
        if not args.remote:
            pbar_train.close()
            pbar_test.close()

        # Read Grid Search Result
        grid.read_records(resultDict)
        print("=== Grid Search Result ===")
        print(grid)
        hyperparams = []
        results = []
        for i, lr in enumerate(lr_list):
            for j, batch_size in enumerate(batch_size_list):
                hyperparams.append((lr, batch_size))
                results.append(lr)
        vis_search(hyperparams, results)

        # Print Execution Time
        end = time.time()
        print("Execution Time:", end-start)
        