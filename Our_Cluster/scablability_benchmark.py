# Standard Library
from __future__ import print_function  
import sys
import numpy as np
import random
import time
import os 
import glob

# Third-Party Package
import torch     
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from argparse import ArgumentParser
from tqdm import tqdm
from mpi4py import MPI

# My Module
from grid_search_utils import *
from random_search_utils import *
from evolution_search_utils import *
from mnist_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DETAIL_LOG = False

def train_mnist_(hparams, device=None, pbars=None, DEBUG=False):
    # Set up Hyper-parameters
    bs = bs_default 
    mmt = mmt_default
    dr = dr_default
    lr = lr_default

    for rv in hparams:
        if rv.name == LEARNING_RATE_NAME:
            lr = rv.value
        elif rv.name == DROPOUT_RATE_NAME:
            dr = rv.value
        elif rv.name == MOMENTUM_NAME:
            mmt = rv.value
        elif rv.name == BATCH_SIZE_NAME:
            bs = rv.value
    # print(lr, dr, mmt, bs)
    if pbars is not None:
        pbar_train = pbars['train']
        pbar_test = pbars['test']
    else:
        pbar_train = None
        pbar_test = None

    # Reproducibitity
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

     # Init Network
    net = Net(dropout_rate=dr)
    net.to(device)

    # Data Loader
    train_loader = get_train_loader(bs)
    val_loader = get_val_loader(bs)

    # Hyper-Parameters
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=mmt)
    criterion = nn.CrossEntropyLoss()
    train_acc = train(net, train_loader, optimizer, criterion, 
        device, pbar_train, DEBUG=DEBUG)
    test_acc = test(net, val_loader, criterion, 
        device, pbar_test, DEBUG=DEBUG)

    return train_acc, test_acc

def write_time_log_file(time, acc, file_name="result/acc_evoluation_search.txt"):
    if os.path.isfile(file_name):
        with open(file_name, "a+") as f:
            f.write("{:.4f},{:.4f}\n".format(time, acc))
    else:
        with open(file_name, "w+") as f:
            f.write("{},{}\n".format("time", "acc"))
            f.write("{:.4f},{:.4f}\n".format(time, acc))

####################
### Grid Search  ###
####################
def getIdxes(mpiWorld, allIdx):
    world_size = mpiWorld.world_size
    my_rank = mpiWorld.my_rank
    total = len(allIdx)
    start_idx = my_rank * (total/world_size) if my_rank != 0 else 0
    end_idx   = (my_rank+1) * (total/world_size) if my_rank != world_size-1 else total
    idxes = allIdx[int(start_idx):int(end_idx)]
    return idxes

def grid_search(mpiWorld, args):
    start = time.time()
    # === Init Search Space === #
    lr  = CRV(low=0.0, high=1.0, name=LEARNING_RATE_NAME).to_DRV(args.grid_size)
    dr  = CRV(low=0.0, high=1.0, name=DROPOUT_RATE_NAME).to_DRV(args.grid_size)
    mmt = CRV(low=0.0, high=1.0, name=MOMENTUM_NAME).to_DRV(args.grid_size)
    bs_choices =[2**i for i in range(args.bs_choice_low, args.bs_choice_high+1)]
    bs  = DRV(choices=bs_choices, name=BATCH_SIZE_NAME)

    hparams = HyperParams([lr, dr, mmt, bs])
    gridSearch =  GridSearch(hparams)
    if mpiWorld.isMaster():
        if DETAIL_LOG:
            print(gridSearch)
        else:
            print("[Grid Search]")
    resultDict = {}
    pbars = {"search":None, "train":None, "test":None}

    if mpiWorld.isMaster():
        allIdx = [i for i in range(len(gridSearch))]
        random.shuffle(allIdx)
    else:
        allIdx = None
    allIdx = mpiWorld.comm.bcast(allIdx, root=mpiWorld.MASTER_RANK)
    idxes = getIdxes(mpiWorld, allIdx[:args.n_search])
    mpiWorld.comm.barrier()
    # === Start Search === #
    best_acc = 0.0
    for i, idx in enumerate(idxes):
        # Get Hyper-Parameters
        hparams = gridSearch[idx]

        # Train MNIST
        train_acc, test_acc = train_mnist_(
            hparams, device=device, pbars=pbars, DEBUG=args.DEBUG)
        resultDict[hparams] = test_acc

    mpiWorld.comm.barrier()
    end = time.time()
    time_elapsed = end-start
    if mpiWorld.isMaster():
        print("\tTime Elapsed:{:.2f}, Best Acc:{:.2f}, #hps Eval:{}".format(
            end-start, get_best_acc(resultDict), len(resultDict)))
    closePbars(pbars)
    return time_elapsed, get_best_acc(resultDict)


###################
### Ran Search  ###
###################
def get_num_search_local(mpiWorld, num_search_global):
    world_size = mpiWorld.world_size
    my_rank = mpiWorld.my_rank
    total = num_search_global
    start_idx = my_rank * (total/world_size) if my_rank != 0 else 0
    end_idx   = (my_rank+1) * (total/world_size) if my_rank != world_size-1 else total
    return int(end_idx) - int(start_idx)

def ran_search(mpiWorld, args):
    start = time.time()
    # === Init Search Space === #
    c = 1e-4
    lr  = CRV(low=0.0+c, high=1.0-c, name=LEARNING_RATE_NAME)
    dr  = CRV(low=0.0+c, high=1.0-c, name=DROPOUT_RATE_NAME)
    mmt = CRV(low=0.0+c, high=1.0-c, name=MOMENTUM_NAME)
    bs_choices =[2**i for i in range(args.bs_choice_low, args.bs_choice_high+1)]
    bs  = DRV(choices=bs_choices, name=BATCH_SIZE_NAME)

    hparams = HyperParams([lr, dr, mmt, bs])
    randomSearch = RandomSearch(hparams)
    if mpiWorld.isMaster():
        if DETAIL_LOG:
            print(randomSearch)
        else:
            print("[Random Search]")
        
    resultDict = {}
    result_log = []
    pbars = {"search":None, "train":None, "test":None}

    num_search_local = get_num_search_local(mpiWorld, args.n_search)
    mpiWorld.comm.barrier()

    # === Start Search === #
    best_acc = 0.0
    for i in range(num_search_local):

        # Get Hyper-Parameters
        hparams = randomSearch.get()

        # Train MNIST
        train_acc, test_acc = train_mnist_(
            hparams, device=device, pbars=pbars, DEBUG=args.DEBUG)

        # Sync Data
        resultDict[hparams] = test_acc
    mpiWorld.comm.barrier()
    end = time.time()
    time_elapsed = end-start
    if mpiWorld.isMaster():
        print("\tTime Elapsed:{:.2f}, Best Acc:{:.2f}, #hps Eval:{}".format(
            end-start, get_best_acc(resultDict), len(resultDict)))
    closePbars(pbars)
    return time_elapsed, get_best_acc(resultDict)

###################
### Evo Search  ###
###################
def flatten(list_2d):
    if list_2d is None:
        return None
    list_1d = []
    for l in list_2d:
        list_1d += l
    return list_1d

def evaluate_popuation(mpiWorld, population, pbars, DEBUG=False):
    # Scatter Population
    local_populations = []
    if mpiWorld.isMaster():
        world_size = mpiWorld.world_size
        total = len(population)
        for rank in range(world_size):
            s = (rank)   * (total/world_size) if rank != 0            else 0
            e = (rank+1) * (total/world_size) if rank != world_size-1 else total
            local_populations.append(population[int(s):int(e)])
    local_population = mpiWorld.comm.scatter(local_populations, root=mpiWorld.MASTER_RANK)

    # Evaluate local_population
    local_fitness = []
    logs = []
    for i, hparams in enumerate(local_population):
        # Train MNIST
        train_acc, acc = train_mnist_(hparams, device=device, pbars=pbars, DEBUG=DEBUG)
        local_fitness.append(acc)
    mpiWorld.comm.barrier()
    population_gather = mpiWorld.comm.gather(local_population, root=mpiWorld.MASTER_RANK)
    fitness_gather    = mpiWorld.comm.gather(local_fitness, root=mpiWorld.MASTER_RANK)
    return flatten(population_gather), flatten(fitness_gather)

def generation(mpiWorld, pop_start, resultDict, pbars, DEBUG=False):
    pop_dict = {}
    uneval_pop = None
    if mpiWorld.isMaster():
        pop_dict['start'] = pop_start
        population_size = len(pop_start)
        # Make Child (Crossover & Mutation)
        pop_child = make_child(pop_start, resultDict, population_size)
        pop_dict['child'] = pop_child
        population = pop_start + pop_child
        # Get unevaluated population
        uneval_pop = get_unevaluated_population(population, resultDict)

    # Evaluation
    uneval_pop, uneval_fitness = evaluate_popuation(mpiWorld, uneval_pop, pbars, DEBUG)
    
    if mpiWorld.isMaster():
        # Update resultDict
        for hps, acc in zip(uneval_pop, uneval_fitness):
            if hps not in resultDict:
                resultDict[hps] = acc
        # Select Best
        fitness = []
        for hps in population:
            fitness.append(resultDict[hps])
        sort_idx  = np.argsort(fitness)
        fitness = [fitness[i] for i in sort_idx]
        population = [population[i] for i in sort_idx]
        pop_dict['selection'] = population[-population_size:][::-1]
    return pop_dict


def evo_search(mpiWorld, args):
    start = time.time()
    # === Init Search Space === #
    c = 1e-4
    lr  = CRV(low=0.0+c, high=1.0-c, name=LEARNING_RATE_NAME)
    dr  = CRV(low=0.0+c, high=1.0-c, name=DROPOUT_RATE_NAME)
    mmt = CRV(low=0.0+c, high=1.0-c, name=MOMENTUM_NAME)
    bs_choices =[2**i for i in range(args.bs_choice_low, args.bs_choice_high+1)]
    bs  = DRV(choices=bs_choices, name=BATCH_SIZE_NAME)
    hparams = HyperParams([lr, dr, mmt, bs])
    
    # === Init Population === #
    num_generation  = args.n_gen
    population_size = args.pop_size
    population = [hparams.copy().initValue() for i in range(population_size)]
    if mpiWorld.isMaster(): 
        if DETAIL_LOG:
            print(get_evo_name(hparams, args.pop_size, args.n_gen))
        else:
            print("[Evo Search]")
    pbars = {"search":None, "train":None, "test":None}
    # === Start Search === #
    resultDict = {}
    pop_dicts = []
    best_acc = 0.0
    mpiWorld.comm.Barrier()
    for i in range(0, num_generation+1):
        if i==0:
            uneval_pop, uneval_fitness = evaluate_popuation(
                mpiWorld, population, pbars, args.DEBUG)
            if mpiWorld.isMaster():
                for hps, acc in zip(uneval_pop, uneval_fitness):
                    if hps not in resultDict:
                        resultDict[hps] = acc
        else:
            pop_dict = generation(mpiWorld, population, resultDict, pbars, args.DEBUG)
            if mpiWorld.isMaster():
                population = pop_dict['selection']
                pop_dicts.append(pop_dict)
        if mpiWorld.isMaster():
            acc_list = []
            for hps in population:
                acc_list.append(resultDict[hps])

    mpiWorld.comm.barrier()
    end = time.time()
    time_elapsed = end-start
    if mpiWorld.isMaster():
        print("\tTime Elapsed:{:.2f}, Best Acc:{:.2f}, #hps Eval:{}".format(
            end-start, get_best_acc(resultDict), len(resultDict)))
    closePbars(pbars)
    return time_elapsed, get_best_acc(resultDict)

if __name__ == "__main__":
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
    parser.add_argument("--DEBUG", default=False)
    parser.add_argument("--n_comparsion", default=1, type=int)
    parser.add_argument("--which_search", default=1, type=int)
    parser.add_argument("--bs_choice_low",  default=7, type=int)
    parser.add_argument("--bs_choice_high",  default=7, type=int)
    parser.add_argument("--grid_size", default=4, type=int)
    parser.add_argument("--n_search", default=64, type=int)
    parser.add_argument("--n_gen", default=16-1, type=int)
    parser.add_argument("--pop_size", default=4, type=int)
    args = parser.parse_args()
    args.DEBUG = str2bool(args.DEBUG)
    if args.DEBUG:
        DETAIL_LOG = True
    if mpiWorld.isMaster():
        print("\n=== Args ===")
        print("Args:{}\n".format(args))

    np.set_printoptions(precision=2)

    GRID_ACC_NAME = "result/acc_grid_search.txt" 
    RAN_ACC_NAME  = "result/acc_ran_search.txt"
    EVO_ACC_NAME  = "result/acc_evo_search.txt"

    for i in range(args.n_comparsion):
        if mpiWorld.isMaster():
            print("~~~ Scablability Comparsion {}/{} ~~~".format(i+1, args.n_comparsion))

        # Grid Search
        if args.which_search == 1:
            time_elapsed, best_acc = grid_search(mpiWorld, args)

        # Random Search 
        if args.which_search == 2:
            time_elapsed, best_acc = ran_search(mpiWorld, args)
            
        # Evoluation Search 
        if args.which_search == 3:
            time_elapsed, best_acc = evo_search(mpiWorld, args)
            
        