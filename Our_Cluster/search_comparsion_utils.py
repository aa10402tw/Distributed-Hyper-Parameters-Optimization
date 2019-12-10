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
import random
import time
import os 
import glob

from mnist_utils import *
from grid_search_utils      import *
from random_search_utils    import *
from evolution_search_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def initPbars_(mpiWorld, exp=False):
    if exp:
        return {"search":None, "train":None, "test":None}
    else:
        if mpiWorld.isMaster():
            print("=== [Progress Bars] ===")
            return {"search":tqdm(), "train":tqdm(), "test":tqdm()}
        else:
            return {"search":None, "train":None, "test":None}

#########################
##### Random Search ##### 
#########################
def get_num_search_local(mpiWorld, num_search_global):
    world_size = mpiWorld.world_size
    my_rank = mpiWorld.my_rank
    total = num_search_global
    start_idx = my_rank * (total/world_size) if my_rank != 0 else 0
    end_idx   = (my_rank+1) * (total/world_size) if my_rank != world_size-1 else total
    return int(end_idx) - int(start_idx)

def random_search(mpiWorld, args):
    start = time.time()
    # === Init Search Grid === #
    c = 1e-4
    lr  = CRV(low=0.0+c, high=1.0-c, name=LEARNING_RATE_NAME)
    dr  = CRV(low=0.0+c, high=1.0-c, name=DROPOUT_RATE_NAME)
    mmt = CRV(low=0.0+c, high=1.0-c, name=MOMENTUM_NAME)
    hparams = HyperParams([lr, dr, mmt])
    randomSearch = RandomSearch(hparams)
    if mpiWorld.isMaster() and not args.exp:
        print(randomSearch)

    # === Init Progress Bar === #
    pbars = initPbars_(mpiWorld, args.exp)
    if mpiWorld.isMaster() and not args.exp:
        pbars['search'].reset(total=args.n_random_search)
        pbars['search'].set_description("Random Search")

    # === Get Indexs === #
    num_search_local  = get_num_search_local(mpiWorld, args.n_random_search)
    resultDict = {}

    # === Start Search === #
    termination_time = float('inf')
    TERMINATION = False
    for i in range(num_search_local):
        # Get Hyper-Parameters
        hparams = randomSearch.get()

        # Train MNIST
        acc = train_mnist(hparams, device=device, pbars=pbars, DEBUG=args.DEBUG)

        # Sync Data
        resultDict[hparams] = acc
        resultDict = syncData(resultDict, mpiWorld, blocking=True)

        # Update Grid Search Progress bar
        if mpiWorld.isMaster() and not args.exp:
            pbars['search'].update(len(resultDict)-pbars['search'].n)

        # Test Temination Criteria and Do Synchronization
        if get_best_acc(resultDict) >= args.criteria:
            termination_time = time.time()
            TERMINATION = True
        TERMINATION = mpiWorld.comm.bcast(TERMINATION, root=mpiWorld.MASTER_RANK)
        if TERMINATION:
            break

        # Update Random Search Progress bar
        if mpiWorld.isMaster() and not args.exp:
            pbars['search'].update(len(resultDict)-pbars['search'].n)
    mpiWorld.comm.Barrier()
    resultDict = syncData(resultDict, mpiWorld, blocking=True)
    mpiWorld.comm.Barrier()
    closePbars(pbars)

    termination_time  = mpiWorld.comm.reduce(termination_time, op=MPI.MIN, root=mpiWorld.MASTER_RANK)
    end = time.time()
    if mpiWorld.isMaster():
        end = min(termination_time, end)
    time_elapsed = end-start
    return time_elapsed, get_best_acc(resultDict)

#######################
##### Grid Search ##### 
#######################
def getIdxes(mpiWorld, gridSearch):
    world_size = mpiWorld.world_size
    my_rank = mpiWorld.my_rank
    total = len(gridSearch)
    start_idx = my_rank * (total/world_size) if my_rank != 0 else 0
    end_idx   = (my_rank+1) * (total/world_size) if my_rank != world_size-1 else total
    idxes = [i for i in range(int(start_idx), int(end_idx))]
    return idxes

def grid_search(mpiWorld, args):
    start = time.time()
    # === Init Search Grid === #
    lr  = CRV(low=0.0, high=1.0, name=LEARNING_RATE_NAME).to_DRV(args.grid_size)
    dr  = CRV(low=0.0, high=1.0, name=DROPOUT_RATE_NAME ).to_DRV(args.grid_size)
    mmt = CRV(low=0.0, high=1.0, name=MOMENTUM_NAME).to_DRV(args.grid_size)
    hparams = HyperParams([lr, dr, mmt])
    gridSearch = GridSearch(hparams)
    if mpiWorld.isMaster() and not args.exp:
        print(gridSearch)

    # === Init Progress Bar === #
    pbars = initPbars_(mpiWorld, args.exp)
    if mpiWorld.isMaster() and not args.exp:
        pbars['search'].reset(total=len(gridSearch))
        pbars['search'].set_description("Grid Search")

    # === Get Indexs === #
    local_idxes = getIdxes(mpiWorld, gridSearch)
    random.shuffle (local_idxes)
    resultDict = {}
    mpiWorld.comm.Barrier()

    # === Start Search === #
    termination_time = float("inf")
    TERMINATION = False
    for idx in local_idxes:
        # Get Hyper-Parameters
        hparams = gridSearch[idx]

        # Train MNIST
        acc = train_mnist(hparams, device=device, pbars=pbars, DEBUG=args.DEBUG)
        
        # Sync Data
        resultDict[hparams] = acc
        resultDict = syncData(resultDict, mpiWorld, blocking=True)

        # Test Temination Criteria and Do Synchronization
        if get_best_acc(resultDict) >= args.criteria:
            termination_time = time.time()
            TERMINATION = True
        TERMINATION = mpiWorld.comm.bcast(TERMINATION, root=mpiWorld.MASTER_RANK)
        if TERMINATION:
            break

        # Update Grid Search Progress bar
        if mpiWorld.isMaster() and not args.exp:
            pbars['search'].update(len(resultDict)-pbars['search'].n)

    mpiWorld.comm.Barrier()
    resultDict = syncData(resultDict, mpiWorld, blocking=True)
    mpiWorld.comm.Barrier()
    closePbars(pbars)

    termination_time  = mpiWorld.comm.reduce(termination_time, op=MPI.MIN, root=mpiWorld.MASTER_RANK)
    end = time.time()
    if mpiWorld.isMaster():
        end = min(termination_time, end)
    time_elapsed = end-start
    return time_elapsed, get_best_acc(resultDict)


#############################
##### Evoluation Search ##### 
#############################
def evaluate_popuation(population, mpiWorld, pbars, args):
    termination_time = float("inf")
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
    for i, hparams in enumerate(local_population):
        if mpiWorld.isMaster() and not args.exp:
            pbars['search'].set_description(
                "Local:({}/{}), Global:{}".format(i+1, len(local_population), len(population))
            )
        # Train MNIST
        acc = train_mnist(hparams, device=device, pbars=pbars, DEBUG=args.DEBUG)
        local_fitness.append(acc)

        if acc >= args.criteria:
            termination_time = time.time()
    # Gather Fitness
    population_gather = mpiWorld.comm.gather(local_population, root=mpiWorld.MASTER_RANK)
    fitness_gather    = mpiWorld.comm.gather(local_fitness, root=mpiWorld.MASTER_RANK)
    termination_time  = mpiWorld.comm.reduce(termination_time, op=MPI.MIN, root=mpiWorld.MASTER_RANK)
    return flatten(population_gather), flatten(fitness_gather), termination_time

def evoluation_search(mpiWorld, args):
    start = time.time()
    # === Init Search Grid === #
    c = 1e-4
    lr  = CRV(low=0.0+c, high=1.0-c, name=LEARNING_RATE_NAME)
    dr  = CRV(low=0.0+c, high=1.0-c, name=DROPOUT_RATE_NAME)
    mmt = CRV(low=0.0+c, high=1.0-c, name=MOMENTUM_NAME)
    hparams = HyperParams([lr, dr, mmt])

    # === Init Population === #
    num_generation  = args.n_gen
    population_size = args.pop_size
    if mpiWorld.isMaster():
        population = [hparams.copy().initValue() for i in range(population_size)]
        if not args.exp:
            print(get_evo_name(hparams, args.pop_size, args.n_gen))
    else:
        population = None

    # === Init Progress Bar === #
    pbars = initPbars_(mpiWorld, args.exp)
    if mpiWorld.isMaster() and not args.exp:
        pbars['search'].reset(total=num_generation)
        pbars['search'].set_description("Evolution Search")

    resultDict = {}
    # === Start Search === #
    termination_time = float("inf")
    TERMINATION = False
    for i in range(num_generation+1):
        # At First generation, evaluate population directly
        if i==0:
            uneval_pop = population
        # After First generation, need to Make child first
        else:
            if mpiWorld.isMaster():
                pop_child = make_child(population, args.pop_size)
                uneval_pop = get_unevaluated_population(pop_child, resultDict)
        pop, fit, ter_time = evaluate_popuation(uneval_pop, mpiWorld, pbars, args)
        # Update Result Dict
        if mpiWorld.isMaster():
            for hps, acc in zip(pop, fit):
                resultDict[hps] = acc
            if get_best_acc(resultDict) >= args.criteria:
                termination_time = ter_time
                TERMINATION = True
        TERMINATION = mpiWorld.comm.bcast(TERMINATION, root=mpiWorld.MASTER_RANK)
        if TERMINATION:
            break
        if i>0 and mpiWorld.isMaster() and not args.exp:
            pbars['search'].update()
    closePbars(pbars)

    termination_time  = mpiWorld.comm.reduce(termination_time, op=MPI.MIN, root=mpiWorld.MASTER_RANK)
    end = time.time()
    if mpiWorld.isMaster():
        end = min(termination_time, end)
    time_elapsed = end-start
    return time_elapsed, get_best_acc(resultDict)