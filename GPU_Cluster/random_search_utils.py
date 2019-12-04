import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random

from cifar10_utils import *
from hyperparams import *

class RandomSearch:
    def __init__(self, hyperparams):
        self.hyperparams = hyperparams

    def get(self):
        hparams = self.hyperparams.copy().initValue()
        return hparams

    def __str__(self):
        s = "=== Random Search ===\n"
        for rv in self.hyperparams:
            if isinstance(rv, DRV):
                s += "{}:{}\n".format(rv.name, rv.choices)
            if isinstance(rv, CRV):
                s += "{}:[{} ~ {}]\n".format(rv.name, rv.low, rv.high)
        return s 


def test_random():
    lr = CRV(low=0.0, high=1.0, name=LEARNING_RATE_NAME)
    dr = CRV(low=0.0, high=1.0, name=DROPOUT_RATE_NAME)
    hparams = HyperParams([lr, dr])
    randomSearch = RandomSearch(hparams)
    print(randomSearch)
    hyperparams_list = []
    result_list = []
    for i in range(50):
        hparams = randomSearch.get()
        lr, dr = hparams.getValueTuple()
        acc = random.uniform(0,100)
        hyperparams_list.append((lr, dr))
        result_list.append(acc)
    vis_search(hyperparams_list, result_list, save_name="random")

if __name__ == "__main__":
    test_random()


