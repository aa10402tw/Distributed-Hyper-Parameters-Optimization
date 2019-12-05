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
import pickle
import time
import os 
import glob

from evolution_search_utils import *
from cifar10_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_name = "result/EvolutionSearch"

def evaluate_popuation(population, mpiWorld, pbars, args):
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
        # Train Cifar10
        acc = train_cifar10(hparams, num_epochs=args.num_epochs,
            device=device, pbars=pbars, DEBUG=args.DEBUG)
        local_fitness.append(acc)

    # Gather Fitness
    population_gather = mpiWorld.comm.gather(local_population, root=mpiWorld.MASTER_RANK)
    fitness_gather    = mpiWorld.comm.gather(local_fitness, root=mpiWorld.MASTER_RANK)

    return flatten(population_gather), flatten(fitness_gather)

def generation(pop_start, resultDict, mpiWorld, pbars, args):
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
    uneval_pop, uneval_fitness = evaluate_popuation(uneval_pop, mpiWorld, pbars, args)
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
    parser.add_argument("-DEBUG",  default=False)
    parser.add_argument("-exp",  default=False)
    parser.add_argument("-n_gen",  default=9, type=int)
    parser.add_argument("-pop_size",  default=8, type=int)
    parser.add_argument("-num_epochs",  default=20, type=int)
    args = parser.parse_args()
    args.DEBUG = str2bool(args.DEBUG)
    args.exp = str2bool(args.exp)

    # === Init Search Space === #
    c = 1e-8
    lr  = CRV(low=0.0+c, high=1.0-c, name=LEARNING_RATE_NAME)
    mmt = CRV(low=0.0+c, high=1.0-c, name=MOMENTUM_NAME)
    hparams = HyperParams([lr, mmt])
    
    population_size = args.pop_size
    num_generation  = args.n_gen
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
        pop_dict = generation(population, resultDict, mpiWorld, pbars, args)
        if mpiWorld.isMaster():
            population = pop_dict['selection']
            pop_dicts.append(pop_dict)
            pbars['search'].update()
        else:
            population = None

    end = time.time()
    closePbars(pbars)

    # Write Logs
    if mpiWorld.isMaster():
        pop_save_name = "{}_pop".format(save_name)
        acc_save_name = "{}_acc".format(save_name)
        write_log(pop_dicts, save_name=pop_save_name)
        pop_logs = load_log(save_name=pop_save_name)
        #print(pop_logs)
        write_log(resultDict, save_name=acc_save_name)
        acc_logs = load_log(save_name=acc_save_name)
        #print(acc_logs)

    # Save Figure
    if mpiWorld.isMaster():
        # Display Grid Search Result
        hyperparams_list = []
        result_list = []
        for hparams, acc in resultDict.items():
            hyperparams_list.append(hparams.getValueTuple())
            result_list.append(acc)
        vis_search(hyperparams_list, result_list, save_name)
        print("\n\nBest Accuracy:{:.4f}\n".format(get_best_acc(resultDict)))

        print("Number of HyperParams evaluated : {}".format(len(hyperparams_list)))

        # Print Execution Time
        print("Execution Time:", end-start)

        vis_generation(pop_dicts, save_name="result/res_same", same_limit=True)
        vis_generation(pop_dicts, save_name="result/es", same_limit=False)

        # for pop_dict in pop_dicts:
        #     print(pop_dict['selection'])