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

def vis_random_search(hyperparams):
    
    if len(hyperparams[0]) == 2:
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

    elif len(hyperparams[0]) == 3:
        xs = [h[0] for h in hyperparams]
        ys = [h[1] for h in hyperparams]
        zs = [h[2] for h in hyperparams]

        plt.scatter(xs, ys, c=zs, cmap='cool')
        if type(xs[0]) == type(0.0):
            plt.xticks([min(xs), max(xs)])
        elif type(xs[0]) == type(0):
            plt.xticks(list(set(xs)))
        if type(ys[0]) == type(0.0):
            plt.yticks([min(ys), max(ys)])
        elif type(ys[0]) == type(0):
            plt.yticks(list(set(ys)))
        # plt.clim(0, 100)
        plt.colorbar()
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for x, y, z in zip(xs, ys, zs):
            z_ = (z - min(zs)) / max(zs)
            cmap = plt.cm.get_cmap('cool')
            ax.scatter([x], [y], [z], c=[cmap(z_)])
            ax.plot([x, x], [y,y], [z, 0], linestyle=":", c=cmap(z_))
        plt.show()


def test_random():
    var_lr = continuousRandomVariable(0, 1)
    var_batch_size = discreteRandomVariable([16, 32, 64, 128])
    randomSearch = RandomSearch([var_lr, var_batch_size], ["learning_rate", "batch_size"])
    print(randomSearch)
    hyperparams = []
    for i in range(50):
        lr, batch_size = randomSearch.get()
        acc = random.uniform(0,100)
        hyperparams.append((lr, batch_size, acc))
    vis_random_search(hyperparams)

if __name__ == "__main__":
    test_random()


