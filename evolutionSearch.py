import os
import glob
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from mnist_utils import *

import copy

# === Random Variable === #
class CRV():
    def __init__(self, low, high, value=None, name=""):
        self.low  = low
        self.high = high
        self.name = name
        self.value = value
        if self.value is None:
            self.initValue()

    def initValue(self):
        self.value = random.uniform(self.low, self.high)

    def __str__(self):
        s = "{} = {} (Continuous Random Variable, low={}, high={})".format(
            self.name, self.value, self.low, self.high
        )
        return s

    def __eq__(self, other):
        return (self.__class__ == other.__class__
            and self.value == other.value)

    def __repr__(self):
        s = "{}={:.4f}".format(
            self.name, self.value
        )
        return s
    def __hash__(self):
        return hash(self.value)

    def copy(self):
        return copy.deepcopy(self)

class DRV:
    def __init__(self, choices, value=None, name=""):
        self.choices = choices
        self.name = name
        self.value = value
        if self.value is None:
            self.initValue()

    def initValue(self):
        self.value = random.choices(self.choices, k=1)[0]

    def __str__(self):
        s = "{} = {} (Discrete Random Variable, choices={})".format(
            self.name, self.value, self.choices
        )
        return s

    def __repr__(self):
        s = "{}={}".format(
            self.name, self.value
        )
        return s

    def __eq__(self, other):
        return (self.__class__ == other.__class__
            and self.value == other.value)

    def __hash__(self):
        return hash(self.value)

    def copy(self):
        return copy.deepcopy(self)

class HyperParams:
    def __init__(self, rvs):
        self.rvs = rvs

    def initValue(self):
        for i in range(len(self.rvs)):
            self.rvs[i].initValue()

    def __str__(self):
        s = "[Hyper-Parameters]\n"
        for i, rv in enumerate(self.rvs):
            s += "  var {} : {}\n".format(i+1, rv)
        return s

    def __repr__(self):
        s = "(HyperParams:"
        for i, rv in enumerate(self.rvs):
            s += "{},".format(rv.__repr__())
        s += ")\n"
        return s

    def __len__(self):
        return len(self.rvs)

    def __getitem__(self, key):
        return self.rvs[key]

    def __setitem__(self, key, value):
        self.rvs[key] = value

    def __hash__(self):
        return hash( self.__repr__() )

    def __eq__(self, other):
        if not self.__class__ == other.__class__:
            return False
        if not len(self) == len(other):
            return False
        for (rv_1, rv_2) in zip(self.rvs, other.rvs):
            if rv_1 != rv_2:
                return False
        return True

    def copy(self):
        return copy.deepcopy(self)

def crossover(population, prob_crossover=0.5):
    new_population = []
    for hparams_1 in population:
        if random.uniform(0, 1) <= prob_crossover:
            hparams_2 = random.choice(population)
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

def mutation(population, prob_mutation=0.1):
    new_population = []
    for hparams in population:
        if random.uniform(0, 1) <= prob_mutation:
            rvs = []
            for i, rv in enumerate(hparams.rvs):
                if isinstance(rv, CRV):
                    mean = rv.value
                    std = np.std( [h.rvs[i].value for h in population] )
                    rv_new = rv.copy()
                    rv_new.value = np.random.normal(mean, std, 1).clip(rv.low, rv.high).item()
                    rvs.append(rv_new)
                if isinstance(rv, DRV):
                    rv_new = rv.copy()
                    rv_new.value = random.choices(rv.choices)[0]
                    rvs.append(rv_new)
            hparams_new = HyperParams(rvs)
            new_population.append(hparams_new)
    return new_population

def vis_population(population, label=""):
    if len(population) == 0:
        return 
    xs = [hparams.rvs[0].value for hparams in population]
    ys = [hparams.rvs[1].value for hparams in population]
    plt.scatter(xs, ys, label=label)
    if isinstance(population[0].rvs[0], DRV):
        plt.xticks(list(set(xs)))
    if isinstance(population[0].rvs[1], DRV):
        plt.yticks(list(set(ys)))

def fitness(hparams):
    e1, e2 = 0, 0
    for i, rv in enumerate(hparams):
        if isinstance(rv, CRV):
            e1 += (0.75-rv.value)**2
            e2 += (0.25-rv.value)**2
        if isinstance(rv, DRV):
            if rv.value == 32:
                e1 -= 0.5
            if rv.value == 64:
                e2 -= 0.5
    return min(e1, e2)

def selection(population, k):
    fitness_list = list(map(fitness, population))
    sort_idx = np.argsort(fitness_list)
    good_population = [population[i.item()] for i in sort_idx][:k]
    return good_population

def test_evolution():
    population_size = 30
    num_generation = 100

    population = []
    for i in range(population_size):
        lr = CRV(low=0, high=1, name="lr")
        bs = DRV(choices=[16, 32, 64, 128, 256], name="bs")
        hparams = HyperParams([lr, bs])
        population.append(hparams)

    plt.subplot(331)
    vis_population(population, label="init")
    plt.legend(loc='best')

    for gen in range(1, num_generation+1):
        if num_generation > 8: 
            population_crossover = crossover(population)
            population = population + population_crossover
            population_mutation = mutation(population)
            population = population + population_mutation
            population_selection = selection(population, population_size)
            population = population_selection
        else:
            plt.subplot(3, 3, gen+1)
            # Crossover
            population_crossover = crossover(population)
            vis_population(population_crossover, label="crossover")
            population = population + population_crossover
            #        
            population_mutation = mutation(population)
            vis_population(population_mutation, label="mutation")
            population = population + population_mutation

            population_selection = selection(population, population_size)
            vis_population(population_selection, label="selection")
            plt.legend(loc='best')
            plt.ylim(-1,257)
            plt.xlim(0, 1)
            population = population_selection

    plt.subplot(3, 3, 9)
    plt.ylim(-1,257)
    plt.xlim(0, 1)
    vis_population(population, label="final")
    for h in population:
        print(h)
    plt.legend(loc='best')
    plt.show()

if __name__ == "__main__":
    # test_evolution()
    lr = CRV(low=0, high=1, name="lr")
    bs = DRV(choices=[16, 32, 64, 128, 256], name="bs")
    hparams = HyperParams([lr, bs])
    hparams_2 = HyperParams([lr, bs])
    mydict = {hparams:0}
    print(mydict[hparams_2])



