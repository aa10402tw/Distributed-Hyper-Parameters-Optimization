from __future__ import print_function  
import torch     
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from argparse import ArgumentParser
from tqdm import tqdm
from mpi4py import MPI
import numpy as np
import time
import os 
import glob

from grid_search_utils import *
from mnist_utils import *

MASTER_RANK = 0
device = torch.device("cpu")

def getIdxes(mpiWorld, gridSearch):
    world_size = mpiWorld.world_size
    my_rank = mpiWorld.my_rank
    total = len(gridSearch)
    start_idx = my_rank * (total/world_size) if my_rank != 0 else 0
    end_idx   = (my_rank+1) * (total/world_size) if my_rank != world_size-1 else total
    idxes = [i for i in range(int(start_idx), int(end_idx))]
    return idxes

if __name__ == "__main__":
    start = time.time()

    # === Init MPI World === #
    mpiWorld = initMPI()
    mpiWorld.comm.Barrier()
    logs = mpiWorld.comm.gather(mpiWorld.log, root=mpiWorld.MASTER_RANK)
    if mpiWorld.isMaster():
        print("\n=== MPI World ===")
        for log in logs:
            print(log)

    # === Argument === #
    parser = ArgumentParser()
    parser.add_argument("-remote", default=False)
    parser.add_argument("-DEBUG",  default=False)
    parser.add_argument("-exp",  default=False)
    args = parser.parse_args()
    args.DEBUG = str2bool(args.DEBUG)
    args.exp = str2bool(args.exp)

    # === Init Search Grid === #
    lr = DRV(choices=[i/10 for i in range(1, 5+1)], name=LEARNING_RATE_NAME)
    dr = DRV(choices=[i/10 for i in range(1, 5+1)], name=DROPOUT_RATE_NAME )
    hparams = HyperParams([lr, dr])
    gridSearch = GridSearch(hparams)

    # === Init Progress Bar === #
    if mpiWorld.isMaster():
        print("\n=== Args ===")
        print("Args:{}\n".format(args))
        print(gridSearch)
    pbars = initPbars(mpiWorld, args.remote)
    if mpiWorld.isMaster():
        pbars['search'].reset(total=len(gridSearch))
        pbars['search'].set_description("Grid Search")

    # === Get Indexs === #
    idxes = getIdxes(mpiWorld, gridSearch)
    resultDict = {}
    mpiWorld.comm.Barrier()

    # === Start Search === #
    for idx in idxes:
        # Get Hyper-Parameters
        hparams = gridSearch[idx]
        lr, dr = hparams.getValueTuple()

        # Train MNIST
        acc = train_mnist(hparams, device=device, pbars=pbars, DEBUG=args.DEBUG)
        
        # Sync Data
        resultDict[(lr, dr)] = acc
        if not args.exp:
            resultDict = syncData(resultDict, mpiWorld, blocking=True)

        # Update Grid Search Progress bar
        if mpiWorld.isMaster():
            pbars['search'].update(len(resultDict)-pbars['search'].n)

    mpiWorld.comm.Barrier()
    resultDict = syncData(resultDict, mpiWorld, blocking=True)
    mpiWorld.comm.Barrier()

    # Close Progress Bar 
    closePbars(pbars)
    
    if mpiWorld.isMaster():
        # Display Grid Search Result
        for i in range(len(gridSearch)):
            hparams = gridSearch[i]
            lr, dr = hparams.getValueTuple()
            acc = resultDict[(lr, dr)]
            gridSearch[i] = acc
        print("\n\n=== Grid Search Result ===")
        gridSearch.print2DGrid()
        hyperparams_list = []
        result_list = []
        for (lr, dr), acc in resultDict.items():
            hyperparams_list.append((lr, dr))
            result_list.append(acc)
        vis_search(hyperparams_list, result_list, "GridSearch")
        print("\nBest Accuracy:{:.4f}\n".format(get_best_acc(resultDict)))

        # Print Execution Time
        end = time.time()
        print("Execution Time:", end-start)
        