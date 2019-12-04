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

from evolution_search_utils import *
from mnist_utils import *

MASTER_RANK = 0
device = torch.device("cpu")

def flatten(list_2d):
    if list_2d is None:
        return None
    list_1d = []
    for l in list_2d:
        list_1d += l
    return list_1d

def evaluate_popuation(population, mpiWorld, pbars, DEBUG=False):
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
        if mpiWorld.isMaster():
            pbars['search'].set_description(
                "Local:({}/{}), Global:{}".format(i+1, len(local_population), len(population))
            )
        # Train MNIST
        acc = train_mnist(hparams, device=device, pbars=pbars, DEBUG=DEBUG)
        local_fitness.append(acc)

    # Gather Fitness
    population_gather = mpiWorld.comm.gather(local_population, root=mpiWorld.MASTER_RANK)
    fitness_gather    = mpiWorld.comm.gather(local_fitness, root=mpiWorld.MASTER_RANK)

    return flatten(population_gather), flatten(fitness_gather)

def generation(pop_start, resultDict, mpiWorld, pbars, DEBUG=False):
    pop_dict = {}
    uneval_pop = None
    if mpiWorld.isMaster():
        pop_dict['start'] = pop_start
        population_size = len(pop_start)

        # Make Child
        pop_child = make_child(pop_start, population_size)
        pop_dict['child'] = pop_child

        population = pop_start + pop_child

        # # Crossover
        # pop_crossover = crossover(pop_start)
        # pop_dict['crossover'] = pop_crossover
        # population = pop_start + pop_crossover 

        # # Mutation
        # pop_mutation = mutation(population)
        # pop_dict['mutation'] = pop_mutation
        # population = population + pop_mutation

        # Get unevaluated population
        uneval_pop = get_unevaluated_population(population, resultDict)

    # Evaluation
    uneval_pop, uneval_fitness = evaluate_popuation(uneval_pop, mpiWorld, pbars, DEBUG)
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
    parser.add_argument("-DEBUG",  default=False)
    parser.add_argument("-exp",  default=False)
    args = parser.parse_args()
    args.DEBUG = str2bool(args.DEBUG)
    args.exp = str2bool(args.exp)

    # === Init Search Space === #
    lr = CRV(low=0.0, high=1.0, name=LEARNING_RATE_NAME)
    dr = CRV(low=0.0, high=1.0, name=DROPOUT_RATE_NAME )
    hparams = HyperParams([lr, dr])
    
    
    population_size = 9
    num_generation  = 9
    if mpiWorld.isMaster():
        population = [hparams.copy().initValue() for i in range(population_size)] 
    else:
        population = None

    # === Init Progress Bar === #
    if mpiWorld.isMaster():
        print("\n=== Args ===")
        print("Args:{}\n".format(args))
        print(hparams)
    pbars = initPbars(mpiWorld, args.exp)
    if mpiWorld.isMaster():
        pbars['search'].reset(total=num_generation)
        pbars['search'].set_description("Evolution Search")

    resultDict = {}
    mpiWorld.comm.Barrier()

    # === Start Search === #
    pop_dicts = []
    for i in range(num_generation):
        pop_dict = generation(population, resultDict, mpiWorld, pbars, args.DEBUG)
        if mpiWorld.isMaster():
            population = pop_dict['selection']
            pop_dicts.append(pop_dict)
            pbars['search'].update()
        else:
            population = None

    end = time.time()
    closePbars(pbars)

    if mpiWorld.isMaster():
        # Display Grid Search Result
        hyperparams_list = []
        result_list = []
        for hparams, acc in resultDict.items():
            hyperparams_list.append(hparams.getValueTuple())
            result_list.append(acc)
        vis_search(hyperparams_list, result_list, "Evolution Search")
        print("\n\nBest Accuracy:{:.4f}\n".format(get_best_acc(resultDict)))

        print("Number of HyperParams evaluated : {}".format(len(hyperparams_list)))

        # Print Execution Time
        print("Execution Time:", end-start)

        vis_generation(pop_dicts, save_name="es_same", same_limit=True)
        vis_generation(pop_dicts, save_name="es", same_limit=False)

        # for pop_dict in pop_dicts:
        #     print(pop_dict['selection'])