import matplotlib.pyplot as plt
import numpy as np

from hyperparams import *
from cifar10_utils import *



def product(tuple1):
    """Calculates the product of a tuple"""
    prod = 1
    for x in tuple1:
        prod = prod * x
    return prod

class GridSearch:
    def __init__(self, hyperparams):
        # Assume all hparam is DRV
        self.hyperparams = hyperparams
        self.grid_shape = tuple([len(h.choices) for h in hyperparams])
        self.param_grid = []
        self.result_grid = []
        for idx in range(len(self)):
            entry = self.idx2entry(idx)
            hps = hyperparams.copy()
            for i, e in enumerate(entry):
                hps[i].value = hyperparams[i].choices[e]
            self.param_grid.append(hps)
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
        s = "# === Grid Search === #\n"
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
    lr = DRV(choices=[i/10 for i in range(1, 10+1)], name=LEARNING_RATE_NAME)
    dr = DRV(choices=[i/10 for i in range(1, 10+1)], name=DROPOUT_RATE_NAME )
    #bs = DRV(choices=[16, 32, 64, 128, 256], name="bs")
    hparams = HyperParams([lr, dr])
    print(hparams)

    gridSearch = GridSearch(hparams)
    print(gridSearch)
    gridSearch.print2DGrid()
    for i in range(len(gridSearch)):
        gridSearch[i] = i/100
    gridSearch.print2DGrid()

    hyperparams_list = []
    result_list = []
    for i in range(len(gridSearch)):
        hparams = gridSearch[i]
        lr, dr = hparams.getValueTuple()
        hyperparams_list.append((lr, dr))
        result_list.append(random.uniform(0,100))
    vis_search(hyperparams_list, result_list, save_name="grid")


if __name__ == "__main__":
    test_grid()
