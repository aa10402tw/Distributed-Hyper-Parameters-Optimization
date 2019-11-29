import os
import glob
import matplotlib.pyplot as plt

from hyperparams import *
from mnist_utils import *

import numpy as np

def product(tuple1):
    """Calculates the product of a tuple"""
    prod = 1
    for x in tuple1:
        prod = prod * x
    return prod

class Grid:
    def __init__(self, hyperparams):
        # Assume all hparam is DRV
        self.hyperparams = hyperparams
        self.grid_shape = tuple([len(h.choices) for h in hyperparams])
        self.param_grid = []
        self.result_grid = []
        for idx in range(len(self)):
            entry = self.idx2entry(idx)
            values = []
            for i, e in enumerate(entry):
                values.append( self.hyperparams[i].choices[e] )
            self.param_grid.append(tuple(values))
            self.result_grid.append(float('nan'))

    def get_hparams(self, key):
        if isinstance(key, tuple):
            return self.param_grid[self.entry2idx(key)]
        else:
            return self.param_grid[key]

    def get_result(self, key):
        if isinstance(key, tuple):
            return self.result_grid[self.entry2idx(key)]
        else:
            return self.result_grid[key]

    def entry2idx(self, entry):
        idx = 0
        for i, e in enumerate(entry):
            idx += e*product(self.grid_shape[i+1:])
        return idx

    def idx2entry(self, idx):
        entry = []
        for i, s in enumerate(self.grid_shape):
            entry.append( idx // product(self.grid_shape[i+1:]))
            idx = idx % product(self.grid_shape[i+1:])
        return tuple(entry)

    def shape(self):
        return self.grid_shape

    def print2DGrid(self, row=0, col=1):
        col_choices = self.hyperparams[col].choices
        row_choices = self.hyperparams[row].choices

        s = ""
        for j in range(len(col_choices)):
            if j == 0:
                s += "{:5} ||".format("")
            s += " {0:^6} |".format("%.2f"%col_choices[j])
            if j == len(col_choices)-1:
                s += "\n"
        s = s + "="*len(s.split("\n")[-2]) + "\n"
        s = "="*len(s.split("\n")[-2]) + "\n" + s
        for i in range(len(row_choices)):
            for j in range(len(col_choices)):
                if j == 0:
                    s += "{:5} ||".format("%.2f"%row_choices[i])
                # Compute Avg
                value, cnt = 0, 0
                for idx in range(len(self)):
                    entry = self.idx2entry(idx)

                    if (self.hyperparams[row].choices[entry[row]] == row_choices[i] and 
                        self.hyperparams[col].choices[entry[col]] == col_choices[j]):
                        value += self.get_result(entry)
                        cnt += 1
                if cnt == 0:
                    value = float('nan')
                else:
                    value = value / cnt
                s += " {0:^6} |".format("%.2f"%(value))
                if j == len(col_choices)-1:
                    s += "\n"
        s += "="*len(s.split("\n")[-2]) + "\n"
        print(s)

    def __str__(self):
        s = "=== Grid ===\n"
        for drv in self.hyperparams:
            s += "{}:{}\n".format(drv.name, drv.choices)
        return s 

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self.param_grid[self.entry2idx(key)]
        else:
            return self.param_grid[key]

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            self.result_grid[self.entry2idx(key)] = value
        else: 
            self.result_grid[key] = value

    def __len__(self):
        return product(self.grid_shape)

def test_grid():
    lr = DRV(choices=[i/10 for i in range(1, 10+1)], name="lr")
    dr = DRV(choices=[i/10 for i in range(1, 10+1)], name="dr")
    bs = DRV(choices=[16, 32, 64, 128, 256], name="bs")
    hparams = HyperParams([lr, dr, bs])
    print(hparams)

    grid = Grid(hparams)
    print(grid)
    grid.print2DGrid()
    for i in range(len(grid)):
        grid[i] = i/1000
    grid.print2DGrid()

if __name__ == "__main__":
    test_grid()
