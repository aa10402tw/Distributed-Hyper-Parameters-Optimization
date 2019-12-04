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

from random_search_utils import *
from cifar10_utils import *

from argparse import ArgumentParser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_num_search_local(mpiWorld, num_search_global):
    world_size = mpiWorld.world_size
    my_rank = mpiWorld.my_rank
    total = num_search_global
    start_idx = my_rank * (total/world_size) if my_rank != 0 else 0
    end_idx   = (my_rank+1) * (total/world_size) if my_rank != world_size-1 else total
    return int(end_idx) - int(start_idx)

if __name__ == "__main__":
    start = time.time()

    # === Init MPI World === #
    mpiWorld = initMPI()
    mpiWorld.comm.Barrier()
    logs = mpiWorld.comm.gather(mpiWorld.log, root=mpiWorld.MASTER_RANK)
    if mpiWorld.isMaster():
        check_dataset()
        print("\n=== MPI World ===")
        for log in logs:
            print(log)
    mpiWorld.comm.Barrier()

    # === Argument === #
    parser = ArgumentParser()
    parser.add_argument("-DEBUG",  default=False)
    parser.add_argument("-exp",  default=False)
    parser.add_argument("-num_epochs",  default=20, type=int)
    args = parser.parse_args()
    args.DEBUG = str2bool(args.DEBUG)
    args.exp = str2bool(args.exp)

    # === Init Search Space === #
    num_search_global = 25
    lr = CRV(low=0.0, high=1.0, name=LEARNING_RATE_NAME)
    mmt = CRV(low=0.0, high=1.0, name=MOMENTUM_NAME)
    hparams = HyperParams([lr, mmt])
    randomSearch = RandomSearch(hparams)

    mpiWorld.comm.Barrier()

    # === Init Progress Bar === #
    if mpiWorld.isMaster():
        print("\n=== Args ===")
        print("Args:{}\n".format(args))
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
        hparams = randomSearch.get()
        lr, mmt = hparams.getValueTuple()

        # Train Cifar10
        acc = train_cifar10(hparams, num_epochs=args.num_epochs,
            device=device, pbars=pbars, DEBUG=args.DEBUG)

        # Sync Data
        resultDict[(lr, mmt)] = acc
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

        # Read & Print Grid Search Result
        hyperparams_list = []
        result_list = []
        for (lr, mmt), acc in resultDict.items():
            hyperparams_list.append((lr, mmt))
            result_list.append(acc)
        vis_search(hyperparams_list, result_list, "RandomSearch")
        print("\n\nBest Accuracy:{:.4f}\n".format(get_best_acc(resultDict)))

        print("Number of HyperParams evaluated : {}".format(len(hyperparams_list)))
        
        # Print Execution Time
        end = time.time()
        print("Execution Time:", end-start)
        