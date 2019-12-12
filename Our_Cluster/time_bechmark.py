from __future__ import print_function  
import torch     
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import sys
from argparse import ArgumentParser
from tqdm import tqdm
#from mpi4py import MPI
import numpy as np
import time
import os 
import glob

from grid_search_utils import *
from random_search_utils import *
from evolution_search_utils import *
from mnist_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TIME_LIMIT = 3600 # 1 hour
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
            f.write("[{:.4f}],[{:.4f}]\n".format(time, acc))
    else:
        with open(file_name, "w+") as f:
            f.write("[{:.4f}],[{:.4f}]\n".format(time, acc))


####################
### Grid Search  ###
####################
def grid_search(args):
    start = time.time()
    # === Init Search Space === #
    lr  = CRV(low=0.0, high=1.0, name=LEARNING_RATE_NAME).to_DRV(args.grid_size)
    dr  = CRV(low=0.0, high=1.0, name=DROPOUT_RATE_NAME).to_DRV(args.grid_size)
    mmt = CRV(low=0.0, high=1.0, name=MOMENTUM_NAME).to_DRV(args.grid_size)
    bs_choices =[2**i for i in range(6, 9+1)]
    bs  = DRV(choices=bs_choices, name=BATCH_SIZE_NAME)

    hparams = HyperParams([lr, dr, mmt, bs])
    gridSearch =  GridSearch(hparams)
    if DETAIL_LOG:
        print(gridSearch)
    resultDict = {}
    pbars = {"search":None, "train":None, "test":None}

    # Print Table
    if DETAIL_LOG:
        title = "|{:^5}|{:^6}|{:^6}|{:^6}|{:^5}|{:^10}|".format(
            "cnt", "lr", "dr", "mmt", "bs", "acc", )
        print("="*(len(title)))
        print(title)
        print("="*(len(title)))
    idxes = [i for i in range(len(gridSearch))]
    import random
    random.shuffle(idxes)
    best_acc = 0.0
    # === Start Search === #
    for i, idx in enumerate(idxes[:args.n_search]):
        # Get Hyper-Parameters
        hparams = gridSearch[idx]
        lr, dr, mmt, bs = hparams.getValueTuple()

        # Train MNIST
        train_acc, test_acc = train_mnist_(
            hparams, device=device, pbars=pbars, DEBUG=args.DEBUG)
        resultDict[hparams] = test_acc
        if get_best_acc(resultDict) > best_acc:
            best_acc = get_best_acc(resultDict)
            if not DETAIL_LOG:
                print("{:.2f}({})".format(best_acc, len(resultDict)), end=',')
        if DETAIL_LOG:
            print("|{:^5}|{:^6}|{:^6}|{:^6}|{:^5}|{:^10}|".format(
                i+1, "%.2f"%lr, "%.2f"%dr, "%.2f"%mmt, bs, "%.4f"%test_acc))
        if test_acc > args.criteria:
            break

    end = time.time()
    time_elapsed = end-start
    print("==={}===".format("Grid"))
    print("Time:{}\nBest Acc:{}\n".format(time_elapsed, get_best_acc(resultDict)))
    closePbars(pbars)
    write_time_log_file(time_elapsed, get_best_acc(resultDict), file_name="result/time_grid_search.txt")
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

def ran_search(args):
    start = time.time()
    # === Init Search Space === #
    c = 1e-4
    lr  = CRV(low=0.0+c, high=1.0-c, name=LEARNING_RATE_NAME)
    dr  = CRV(low=0.0+c, high=1.0-c, name=DROPOUT_RATE_NAME)
    mmt = CRV(low=0.0+c, high=1.0-c, name=MOMENTUM_NAME)
    bs_choices =[2**i for i in range(6, 9+1)]
    bs  = DRV(choices=bs_choices, name=BATCH_SIZE_NAME)

    hparams = HyperParams([lr, dr, mmt, bs])
    randomSearch = RandomSearch(hparams)
    if DETAIL_LOG:
        print(randomSearch)
    resultDict = {}
    result_log = []
    pbars = {"search":None, "train":None, "test":None}

    # Print Table
    if DETAIL_LOG:
        title = "|{:^5}|{:^6}|{:^6}|{:^6}|{:^5}|{:^10}|".format(
            "cnt", "lr", "dr", "mmt", "bs", "acc", )
        print("="*(len(title)))
        print(title)
        print("="*(len(title)))
    # === Start Search === #
    best_acc = 0.0
    for i in range(args.n_search):

        # Get Hyper-Parameters
        hparams = randomSearch.get()
        lr, dr, mmt, bs = hparams.getValueTuple()

        # Train MNIST
        train_acc, test_acc = train_mnist_(
            hparams, device=device, pbars=pbars, DEBUG=args.DEBUG)
        resultDict[hparams] = test_acc
        if get_best_acc(resultDict) > best_acc :
            best_acc = get_best_acc(resultDict)
            if not DETAIL_LOG:
                print("{:.2f}({})".format(best_acc, len(resultDict)), end=',')
        if DETAIL_LOG:
            print("|{:^5}|{:^6}|{:^6}|{:^6}|{:^5}|{:^10}|".format(
                i+1, "%.2f"%lr, "%.2f"%dr, "%.2f"%mmt, bs, "%.4f"%test_acc))
        if test_acc > args.criteria:
            break
    end = time.time()
    time_elapsed = end-start
    print("==={}===".format("Ran"))
    print("Time:{}\nBest Acc:{}\n".format(time_elapsed, get_best_acc(resultDict)))
    closePbars(pbars)
    write_time_log_file(time_elapsed, get_best_acc(resultDict), file_name="result/time_random_search.txt")
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

def evaluate_popuation(args, population, pbars, DEBUG=False):
    # Evaluate local_population
    fitness = []
    for i, hparams in enumerate(population):
        # Train MNIST
        train_acc, acc = train_mnist_(hparams, device=device, pbars=pbars, DEBUG=DEBUG)
        fitness.append(acc)
        for rv in hparams:
            if rv.name == LEARNING_RATE_NAME:
                lr = rv.value
            elif rv.name == DROPOUT_RATE_NAME:
                dr = rv.value
            elif rv.name == MOMENTUM_NAME:
                mmt = rv.value
            elif rv.name == BATCH_SIZE_NAME:
                bs = rv.value
        if DETAIL_LOG:
            print("|{:^8}|{:^8}|{:^8}|{:^5}||{:^8}|".format(
                "%.4f"%lr, "%.4f"%dr, "%.4f"%mmt, bs, "%.2f"%acc))
        if acc > args.criteria:
            return population[:len(fitness)], fitness

    return population, fitness

def generation(args, pop_start, resultDict, pbars, DEBUG=False):
    pop_dict = {}
    uneval_pop = None
    pop_dict['start'] = pop_start
    population_size = len(pop_start)

    # Make Child (Crossover & Mutation)
    pop_child = make_child(pop_start, population_size)
    pop_dict['child'] = pop_child

    population = pop_start + pop_child

    # Get unevaluated population
    uneval_pop = get_unevaluated_population(population, resultDict)

    # Evaluation
    uneval_pop, uneval_fitness = evaluate_popuation(args, uneval_pop, pbars, DEBUG)
    # Update resultDict
    for hps, acc in zip(uneval_pop, uneval_fitness):
        if hps not in resultDict:
            resultDict[hps] = acc
    if get_best_acc(resultDict) > args.criteria:
        return None


    # Select Best
    fitness = []
    for hps in population:
        fitness.append(resultDict[hps])
    sort_idx  = np.argsort(fitness)
    fitness = [fitness[i] for i in sort_idx]
    population = [population[i] for i in sort_idx]
    pop_dict['selection'] = population[-population_size:][::-1]
    return pop_dict


def evo_search(args):
    global TERMINATION
    TERMINATION = False
    start = time.time()
    # === Init Search Space === #
    c = 1e-4
    lr  = CRV(low=0.0+c, high=1.0-c, name=LEARNING_RATE_NAME)
    dr  = CRV(low=0.0+c, high=1.0-c, name=DROPOUT_RATE_NAME)
    mmt = CRV(low=0.0+c, high=1.0-c, name=MOMENTUM_NAME)
    bs_choices =[2**i for i in range(6, 9+1)]
    bs  = DRV(choices=bs_choices, name=BATCH_SIZE_NAME)
    hparams = HyperParams([lr, dr, mmt, bs])
    
    # === Init Population === #
    num_generation  = args.n_gen
    population_size = args.pop_size
    population = [hparams.copy().initValue() for i in range(population_size)] 
    if DETAIL_LOG:
        print(get_evo_name(hparams, args.pop_size, args.n_gen))
    pbars = {"search":None, "train":None, "test":None}
    resultDict = {}

    # Print Table   
    if DETAIL_LOG:
        title = "|{:^8}|{:^8}|{:^8}|{:^5}||{:^8}|".format(
            "lr", "dr", "mmt", "bs", "acc")
        print("="*(len(title)))
        print(title)
        print("="*(len(title)))
    # === Start Search === #
    pop_dicts = []
    best_acc = 0.0
    for i in range(0, num_generation+1):
        if DETAIL_LOG:
            gen_text = "(Generation : {})".format(i)
            gen_info = " "*int(len(title)/2-len(gen_text)/2) + gen_text +  " "*int(len(title)/2-len(gen_text)/2)
            print(gen_info)
            print("="*(len(title)))
        if i==0:
            uneval_pop, uneval_fitness = evaluate_popuation(
                args, population, pbars, args.DEBUG)
            for hps, acc in zip(uneval_pop, uneval_fitness):
                if hps not in resultDict:
                    resultDict[hps] = acc
            if get_best_acc(resultDict) > args.criteria:
                break
        else:
            pop_dict = generation(args, population, resultDict, pbars, args.DEBUG)
            if get_best_acc(resultDict) > args.criteria:
                break
            population = pop_dict['selection']
            pop_dicts.append(pop_dict)
        acc_list = []
        for hps in population:
            acc_list.append(resultDict[hps])
        #if not DETAIL_LOG:
        print(np.array(sorted(acc_list)[::-1]))
    closePbars(pbars)
    end = time.time()
    time_elapsed = end-start
    print("==={}===".format("Evo"))
    print("Time:{}\nBest Acc:{}\n".format(time_elapsed, get_best_acc(resultDict)))
    closePbars(pbars)
    write_time_log_file(time_elapsed, get_best_acc(resultDict), file_name="result/time_evoluation_search.txt")
    return time_elapsed, get_best_acc(resultDict)


if __name__ == "__main__":
    # === Argument === #
    parser = ArgumentParser()
    parser.add_argument("--DEBUG",  default=False)
    parser.add_argument("--criteria", default=95, type=float)
    parser.add_argument("--grid_size",  default=10, type=int)
    parser.add_argument("--n_search",  default=1000, type=int)
    parser.add_argument("--n_gen",  default=200-1, type=int)
    parser.add_argument("--pop_size",  default=5, type=int)
    args = parser.parse_args()
    args.DEBUG = str2bool(args.DEBUG)
    print("\n=== Args ===")
    print("Args:{}\n".format(args))
    np.set_printoptions(precision=2)

    for i in range(10):
        print("\n==Grid==")
        time_elapsed, best_acc = grid_search(args)
        print("\n==Ran==")
        time_elapsed, best_acc = ran_search(args)
        print("\n===EVO===")
        time_elapsed, best_acc = evo_search(args)