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


def getIdxes(mpiWorld, gridSearch):
    world_size = mpiWorld.world_size
    my_rank = mpiWorld.my_rank
    total = len(gridSearch)
    start_idx = my_rank * (total/world_size) if my_rank != 0 else 0
    end_idx   = (my_rank+1) * (total/world_size) if my_rank != world_size-1 else total
    idxes = [i for i in range(int(start_idx), int(end_idx))]
    return idxes

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    start = time.time()

    # === Init MPI World === #
    mpiWorld = initMPI()

    # === Argument === #
    parser = ArgumentParser()
    parser.add_argument("-remote", default=False)
    parser.add_argument("-DEBUG",  default=False)
    args = parser.parse_args()
    args.DEBUG = str2bool(args.DEBUG)
    

    # === Init Search Grid === #
    lr = DRV(choices=[i/10 for i in range(1, 10+1)], name="lr")
    dr = DRV(choices=[i/10 for i in range(1, 10+1)], name="dr")
    hparams = HyperParams([lr, dr])
    gridSearch = GridSearch(hparams)

    # === Init Progress Bar === #
    if mpiWorld.isMaster():
        print("\nArgs:{}\n".format(args))
        print("=== Grid Search ===")
        print(gridSearch)
        pbars = {
            "search":tqdm(total=len(gridSearch), desc="Grid Search"),
            "train" :tqdm(),
            "test"  :tqdm()
        }
        if args.remote:
            pbars = {
                "search":tqdm(total=len(gridSearch), desc="Grid Search"),
                "train" :None,
                "test"  :None
            }
    else:
        pbars = {
            "search":None,
            "train" :None,
            "test"  :None
        }

    # === Get Indexs === #
    idxes = getIdxes(mpiWorld, gridSearch)
    resultDict = {}
    mpiWorld.comm.Barrier()

    # === Start Search === #
    for idx in idxes:
        # Get Hyper-Parameters
        lr, dr = gridSearch[idx]

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
    if mpiWorld.isMaster():
        # Close Progress Bar 
        pbars['search'].close()
        if not args.remote:
            pbars['train'].close()
            pbars['test'].close()

        # Display Grid Search Result
        for i in range(len(gridSearch)):
            lr, dr = gridSearch[i]
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

        # Print Execution Time
        end = time.time()
        print("Execution Time:", end-start)
        