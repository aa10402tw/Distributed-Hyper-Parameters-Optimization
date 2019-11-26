import os
import glob
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from mnist_utils import *

class RandomVariable:
    def __init__(self):
        pass

    def get(self):
        pass

class continuousRandomVariable(RandomVariable):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def get(self):
        return random.uniform(self.low, self.high)

    def __str__(self):
        return "Continuous [{}, {}]".format(self.low, self.high) + "\n"

class discreteRandomVariable(RandomVariable):
    def __init__(self, choices):
        self.choices = choices

    def get(self):
        return random.choice(self.choices)

    def __str__(self):
        return "Discrete " + str(self.choices) + "\n"


class RandomSearch:
    def __init__(self, randomVariable_list, name_list=None):
        self.randomVariable_list = randomVariable_list
        self.name_list = name_list

    def get(self):
        items = []
        for rv in self.randomVariable_list:
            item = rv.get()
            items.append(item)
        return tuple(items)

    def __str__(self):
        s = ""
        for i, rv in enumerate(self.randomVariable_list):
            if self.name_list is not None:
                s += "{:15} : {}".format(self.name_list[i], str(rv))
            else:
                s += "Var_%d : "%(i+1) + str(rv)
        return s


def test_random():
    var_lr = continuousRandomVariable(0, 1)
    var_batch_size = discreteRandomVariable([16, 32, 64, 128])
    randomSearch = RandomSearch([var_lr, var_batch_size], ["learning_rate", "batch_size"])
    print(randomSearch)
    hyperparams = []
    results = []
    for i in range(50):
        lr, batch_size = randomSearch.get()
        acc = random.uniform(0,100)
        hyperparams.append((lr, batch_size))
        results.append(acc)
    vis_search(hyperparams, results)

if __name__ == "__main__":
    test_random()


