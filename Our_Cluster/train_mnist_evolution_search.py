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

device = torch.device("cpu")
save_name = "result/EvolutionSearch"

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

        # Make Child (Crossover & Mutation)
        pop_child = make_child(pop_start, population_size)
        pop_dict['child'] = pop_child

        population = pop_start + pop_child

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
        check_dataset()
        print("\n=== MPI World ===")
        for log in logs:
            print(log)
    mpiWorld.comm.Barrier()

    # === Argument === #
    parser = ArgumentParser()
    parser.add_argument("--DEBUG",  default=False)
    parser.add_argument("--exp",  default=False)
    parser.add_argument("--n_gen",  default=9, type=int)
    parser.add_argument("--pop_size",  default=10, type=int)
    args = parser.parse_args()
    args.DEBUG = str2bool(args.DEBUG)
    args.exp = str2bool(args.exp)

    # === Init Search Space === #
    c = 1e-8
    lr = CRV(low=0.0+c, high=1.0-c, name=LEARNING_RATE_NAME)
    dr = CRV(low=0.0+c, high=1.0-c, name=DROPOUT_RATE_NAME)
    hparams = HyperParams([lr, dr])
    
    # === Init Population === #
    num_generation  = args.n_gen
    population_size = args.pop_size
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
    closePbars(pbars)

    # === Display Result === #
    end = time.time()
    if mpiWorld.isMaster():
        print("\n\nBest Accuracy:{:.4f}\n".format(get_best_acc(resultDict)))
        print("Number of HyperParams evaluated : {}".format(len(resultDict)))
        print("Execution Time:", end-start)

    # === Write Result Logs === #
    if mpiWorld.isMaster():
        pop_save_name = "{}_pop".format(save_name)
        acc_save_name = "{}_acc".format(save_name)
        write_log(pop_dicts, save_name=pop_save_name)
        pop_logs = read_log(save_name=pop_save_name)
        write_log(resultDict, save_name=acc_save_name)
        acc_logs = read_log(save_name=acc_save_name)

    # === Write Result Figures === #
    if mpiWorld.isMaster():
        hyperparams_list = []
        result_list = []
        for hparams, acc in resultDict.items():
            hyperparams_list.append(hparams.getValueTuple())
            result_list.append(acc)
        vis_search(hyperparams_list, result_list, save_name)
        vis_generation(pop_dicts, save_name="result/es_same", same_limit=True)
        vis_generation(pop_dicts, save_name="result/es", same_limit=False)