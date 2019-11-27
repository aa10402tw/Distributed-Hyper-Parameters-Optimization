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
from evolutionSearch import *
from mnist_utils import *
import numpy as np

MASTER_RANK = 0
device = torch.device("cpu")

def evaluate_popuation(population, mpiWorld, pbars):
    # Divide Data
    local_population = []
    local_fitness =  []
    if mpiWorld.isMaster():
        n = len(population) / (mpiWorld.world_size)
        for i in range(mpiWorld.world_size):
            s = int(i*n) if i != 0 else 0
            e = int((i+1)*n) if i != mpiWorld.world_size-1 else len(population)
            local_population.append(population[s:e])
    else:
        local_population = None
    local_population = mpiWorld.comm.scatter(local_population, root=mpiWorld.MASTER_RANK)
    mpiWorld.comm.barrier()

    

    # Evaluate HyperParameters
    for i, hparams in enumerate(local_population):
        if mpiWorld.isMaster():
            pbars['search'].set_postfix({
                "Best":"{:.2f}%".format(get_best_acc(resultDict)), 
                "Progress":"({}/{})".format(i+1, len(local_population))
            })

        # Get Hyper-Parameters
        lr, batch_size = hparams[0].value, hparams[1].value
        # Train MNIST
        acc = train_mnist(lr, batch_size, device, pbars['train'], pbars['test'])
        local_fitness.append(acc)
        

    mpiWorld.comm.barrier()

    populations = mpiWorld.comm.gather(local_population, root=mpiWorld.MASTER_RANK)
    scores      = mpiWorld.comm.gather(local_fitness, root=mpiWorld.MASTER_RANK)
    if mpiWorld.isMaster():
        population = []
        fitness = []
        for p in populations:
            population += p
        for s in scores:
            fitness += s
        return population, fitness
    else:
        return None, None

def get_unevaluated_population(population, resultDict):
    if mpiWorld.isMaster():
        unevaluated_population = []
        for hyperparams in population:
            key = tuple(hyperparams.rvs)
            if key not in resultDict:
                unevaluated_population.append(hyperparams)
        return unevaluated_population
    else:
        return None

def generation(population, mpiWorld, resultDict, pbars):
    population_dict = {}
    if mpiWorld.isMaster():
        population_dict['start'] = population
        population_size = len(population)

        # === 1. Crossover === #
        population_crossover = crossover(population)
        population_dict['crossover'] = population_crossover
        population += population_crossover
        # print("\n", population_size, len(population), "\n")

        # ===  2. Mutation === #
        population_mutation = mutation(population)
        population_dict['mutation'] = population_mutation
        population += population_mutation 
        # print("\n", population_size, len(population), "\n")

    # === 3. Selcetion === #
    # Evaluate Unevaluated Hypermeters  
    unevaluated_population = get_unevaluated_population(population, resultDict)
    unevaluated_population, fitness = evaluate_popuation(unevaluated_population, mpiWorld, pbars)
    if mpiWorld.isMaster():
        # Update resultDict
        for hparams, acc in zip(unevaluated_population, fitness):
            if hparams not in resultDict:
                key = hparams
                resultDict[key] = acc
                # resultDict[key] = hparams[0].value
        # Select best
        scores = []
        for hparams in population:
            key = hparams
            scores.append(resultDict[key])
        sort_idx = np.argsort(scores)
        scores = [scores[i.item()] for i in sort_idx][-population_size:][::-1]
        population = [population[i.item()] for i in sort_idx][-population_size:][::-1]
        population_dict['selcetion'] = population
    return population_dict

def get_best_acc(resultDict):
    best_acc = 0.0
    for hparams, acc in resultDict.items():
        if acc > best_acc:
            best_acc = acc
    return best_acc

if __name__ == "__main__":
    start = time.time()

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # === Init MPI World === #
    mpiWorld = initMPI()
    pbars = initPbars(mpiWorld)

    # === Arg Parse === #
    parser = ArgumentParser()
    parser.add_argument("-remote", default=False)
    args = parser.parse_args()

    # === Init Population === #
    population = []
    resultDict = {}
    if mpiWorld.isMaster():
        population_size = 32
        for i in range(population_size):
            lr = CRV(low=0, high=1, name="lr")
            #bs = DRV(choices=[16, 32, 64, 128, 256], name="bs")
            bs = DRV(choices=[16, 64, 256], name="bs")
            hparams = HyperParams([lr, bs])
            population.append(hparams)

    num_generation = 8
    if mpiWorld.isMaster():
        pbars['search'].__init__(total=num_generation, unit='generation')
        pbars['search'].set_description("Evolutionary Search")
        population_each_generation = []
    for g in range(1, num_generation+1):
        population_dict = generation(population, mpiWorld, resultDict, pbars)
        if mpiWorld.isMaster():
            population = population_dict["selcetion"]
            pbars['search'].update()
            population_each_generation.append(population_dict)

    mpiWorld.comm.Barrier()
    if mpiWorld.isMaster():
        # # Close Progress Bar 
        # pbars['search'].close()
        # if not args.remote:
        #     pbar_train.close()
        #     pbar_test.close()

        # Display Evolutionary Search Result
        records = resultDict
        hyperparams = []
        results = []
        print("\n=== Evolutionary Search Result ===\n")
        print("=" * len("{:^15} | {:^10} | {:^10} |".format("learning rate", "batch_size", "acc")))
        print("{:^15} | {:^10} | {:^10} |".format("learning rate", "batch_size", "acc"))
        print("=" * len("{:^15} | {:^10} | {:^10} |".format("learning rate", "batch_size", "acc")))
        for hparams, acc in resultDict.items():
            (lr, batch_size) = hparams[0].value, hparams[1].value
            print("{:^15} | {:^10} | {:^10} |".format("%.4f"%float(lr), batch_size, "%.4f"%float(acc)))
            hyperparams.append((float(lr), int(batch_size)))
            results.append(float(acc))
        print("=" * len("{:^15} | {:^10} | {:^10} |".format("learning rate", "batch_size", "acc")))
        # vis_search(hyperparams, results)

        # Print Execution Time
        end = time.time()
        print("Execution Time:", end-start)

        import matplotlib.pyplot as plt

        for i in range(num_generation):
            if i >= 9:
                break
            plt.subplot(3, 3, i+1)
            vis_population(population_each_generation[i]['start'], label='start')
            vis_population(population_each_generation[i]['crossover'], label='crossover')
            vis_population(population_each_generation[i]['mutation'], label='mutation')
            vis_population(population_each_generation[i]['selcetion'], label='selcetion')
            # plt.legend(loc='best')
            plt.xlim(-0.2, 1.2)
        plt.show()

        