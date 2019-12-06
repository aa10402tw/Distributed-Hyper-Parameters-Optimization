import os
import glob
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from mnist_utils import *
import copy

from hyperparams import *


def get_unevaluated_population(population, resultDict):
    unevaluated_population = []
    for hyperparams in population:
        key = hyperparams
        if key not in resultDict:
            unevaluated_population.append(hyperparams)
    return unevaluated_population

def make_child(population, n_child):
    population = list(set(population))
    child = []
    new_pop = population + child
    while(len(new_pop) - len(population) < n_child):
        child += crossover(population)
        child += mutation(population)
        new_pop = new_pop + child
        new_pop = list(set(new_pop))
    child = list(set(new_pop)-set(population))
    return random.sample(child, n_child)

def crossover(population, prob_crossover=0.8):
    new_population = []
    for i, hparams_1 in enumerate(population):
        if random.uniform(0, 1) <= prob_crossover:
            idx_2 = random.choice([j for j in range(len(population)) if j != i])
            hparams_2 = population[idx_2]
            num_rv_from_h1 = random.randint(1, len(hparams_1))
            rvs = []
            for i in range(len(hparams_1)):
                if i < num_rv_from_h1:
                    rvs.append(hparams_1[i].copy())
                else:
                    rvs.append(hparams_2[i].copy())
            child = HyperParams(rvs)
            new_population.append(child)
    return new_population

def mutation(population, prob_mutation=0.2):
    new_population = []
    for hparams in population:
        new_rvs = []
        for i, rv in enumerate(hparams.rvs):
            if random.uniform(0, 1) <= prob_mutation:
                if isinstance(rv, CRV):
                    mean = rv.value
                    std = np.std( [h.rvs[i].value for h in population] )
                    rv_new = rv.copy()
                    rv_new.value = np.random.normal(mean, std, 1).clip(
                        rv.low, rv.high).item()
                    new_rvs.append(rv_new)
                if isinstance(rv, DRV):
                    rv_new = rv.copy()
                    rv_new.value = random.choices(rv.choices)[0]
                    new_rvs.append(rv_new)
            else:
                new_rvs.append(rv.copy())
        hparams_new = HyperParams(new_rvs)
        if hparams_new != hparams:
            new_population.append(hparams_new)
    return new_population

def vis_population(population, label="", c=None, s=None, marker=None):
    if len(population) == 0:
        return 
    xs = [hparams.rvs[0].value for hparams in population]
    ys = [hparams.rvs[1].value for hparams in population]
    plt.scatter(xs, ys, label=label, c=c, s=s, marker=marker)

    if isinstance(population[0].rvs[0], DRV):
        plt.xticks(list(set(xs)))
    if isinstance(population[0].rvs[1], DRV):
        plt.yticks(list(set(ys)))

def vis_generation(pop_dicts, save_name="es" ,same_limit=True):
    offset = max(1, len(pop_dicts)/9)
    display_idx = [int(i*offset) for i in range(9)]
    hparams = pop_dicts[0]['start'][0]
    plt.clf()
    plt.figure(figsize=(10,10))
    for cnt, gen in enumerate(display_idx):
        if gen >= len(pop_dicts):
            break
        plt.subplot(3, 3, cnt+1)
        vis_population(pop_dicts[gen]['start'], marker='x')
        vis_population(pop_dicts[gen]['child'], marker='x')
        # vis_population(pop_dicts[gen]['crossover'], c='red', marker='x')
        # vis_population(pop_dicts[gen]['mutation'],  c='purple', marker='x')
        vis_population(pop_dicts[gen]['selection'], c='green', marker='x')
        offset = 0.05
        if same_limit:
            plt.xlim(hparams[0].low-offset, hparams[0].high+offset)
            plt.ylim(hparams[1].low-offset, hparams[1].high+offset)
        plt.title("Generation : {}".format(gen+1))
    plt.savefig('{}_generation.png'.format(save_name))


def get_evo_name(hparams, pop_size, n_gen):
    s = "=== Evolution Search ===\n"
    s += "pop_size:{}\n".format(pop_size)
    s += "num_gen:{}\n".format(n_gen)
    for rv in hparams:
        if isinstance(rv, DRV):
            s += "{}:{}\n".format(rv.name, rv.choices)
        if isinstance(rv, CRV):
            s += "{}:[{} ~ {}]\n".format(rv.name, rv.low, rv.high)
    return s

def test_evolution():
    lr = CRV(low=0.0, high=1.0, name=LEARNING_RATE_NAME)
    dr = CRV(low=0.0, high=1.0, name=DROPOUT_RATE_NAME)
    hparams = HyperParams([lr, dr])

    population_size = 5
    num_generation = 5

    population = [hparams.copy().initValue() for i in range(population_size)]
    print(len(population))
    child = make_child(population, 10)
    print(len(child))
    print(population)
    print(child)

if __name__ == "__main__":
    test_evolution()

