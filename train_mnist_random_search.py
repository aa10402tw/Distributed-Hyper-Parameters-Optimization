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

from argparse import ArgumentParser

def get_num_search_local(mpiWorld, num_search_global):
    world_size = mpiWorld.world_size
    my_rank = mpiWorld.my_rank
    total = num_search_global
    start_idx = my_rank * (total/world_size) if my_rank != 0 else 0
    end_idx   = (my_rank+1) * (total/world_size) if my_rank != world_size-1 else total
    return int(end_idx) - int(start_idx)

# Global Variable 
MASTER_RANK = 0
device = torch.device("cpu") 

if __name__ == "__main__":
    start = time.time()

    # === Init MPI World === #
    mpiWorld = initMPI()
    mpiWorld.comm.Barrier()
    
    # === Argument === #
    parser = ArgumentParser()
    parser.add_argument("-remote", default=False)
    parser.add_argument("-DEBUG",  default=False)
    args = parser.parse_args()
    args.DEBUG = str2bool(args.DEBUG)

    # === Init Search Space === #
    num_search_global = 100
    lr = CRV(low=0.0, high=1.0, name="lr")
    dr = CRV(low=0.0, high=1.0, name="dr")
    hparams = HyperParams([lr, dr])
    randomSearch = RandomSearch(hparams)

    mpiWorld.comm.Barrier()

    # === Init Progress Bar === #
    if mpiWorld.isMaster():
        print("\nArgs:{}\n".format(args))
        print(randomSearch)
    pbars = initPbars(mpiWorld, args.remote)
    if mpiWorld.isMaster():
        pbars['search'].reset(total=num_search_global)
        pbars['search'].set_description("Random Search")

    # === Get Indexs === #
    num_search_local  = get_num_search_local(mpiWorld, num_search_global)
    resultDict = {}

    # === Start Search === #
    for i in range(num_search_local):

        # Get Hyper-Parameters
        lr, dr = randomSearch.get()

        # Train MNIST
        acc = train_mnist(lr=lr, dr=dr, 
            device=device, pbars=pbars, DEBUG=args.DEBUG)

        # Sync Data
        resultDict[(lr, dr)] = acc
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

        # Read & Print Grid Search Result
        hyperparams_list = []
        result_list = []
        for (lr, dr), acc in resultDict.items():
            hyperparams_list.append((lr, dr))
            result_list.append(acc)
        vis_search(hyperparams_list, result_list, "RandomSearch")
        print("\n\nBest Accuracy:{:.4f}\n".format(get_best_acc(resultDict)))

        # Print Execution Time
        end = time.time()
        print("Execution Time:", end-start)
        