import os
import glob
import random
import matplotlib.pyplot as plt

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

def vis_random_search(hyperparams):
    xs = [h[0] for h in hyperparams]
    ys = [h[1] for h in hyperparams]
    plt.scatter(xs, ys)
    if type(xs[0]) == type(0.0):
        plt.xticks([min(xs), max(xs)])
    elif type(xs[0]) == type(0):
        plt.xticks(list(set(xs)))
    if type(ys[0]) == type(0.0):
        plt.yticks([min(ys), max(ys)])
    elif type(ys[0]) == type(0):
        plt.yticks(list(set(ys)))
    plt.show()


def test_random():
    var_lr = continuousRandomVariable(0, 1)
    var_batch_size = discreteRandomVariable([16, 32, 64, 128])
    randomSearch = RandomSearch([var_lr, var_batch_size], ["learning_rate", "batch_size"])
    print(randomSearch)
    hyperparams = []
    for i in range(50):
        lr, batch_size = randomSearch.get()
        hyperparams.append((lr, batch_size))
    # vis_random_search(hyperparams)

if __name__ == "__main__":
    test_random()


