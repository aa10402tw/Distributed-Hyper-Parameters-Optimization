import os
import glob
import matplotlib.pyplot as plt

from mnist_utils import *

class Grid:
    def __init__(self, list_1, list_2):
        self.list_1 = list_1
        self.list_2 = list_2
        self.params_grid = []
        for i in range(len(list_1)):
            self.params_grid.append( [(list_1[i], list_2[j]) for j in range(len(self.list_2))] )
        self.result_grid = [ [0.0 for j in range(len(self.list_2))] for i in range(len(self.list_1)) ]

    def __len__(self):
        return len(self.list_1) * len(self.list_2)

    def get(self, idx):
        h = len(self.list_1)
        w = len(self.list_2)
        i, j = int(idx / w), int(idx % w)
        return self.params_grid[i][j]

    def read_records(self, resultDict):
        for key, value in resultDict.items():
            lr, batch_size = key
            i = self.list_1.index(lr)
            j = self.list_2.index(batch_size)
            self.result_grid[i][j] = "{0:.2f}".format(value)

    def __str__(self):
        s = "\n"
        for j in range(len(self.list_2)):
            if j == 0:
                s += "{:5} ||".format("")
            s += " {0:^5} |".format(self.list_2[j])
            if j == len(self.list_2)-1:
                s += "\n"
        s+= "="*len(s.split("\n")[-2]) + "\n"
        for i in range(len(self.list_1)):
            for j in range(len(self.list_2)):
                if j == 0:
                    s += "{:5} ||".format(self.list_1[i] if self.list_1[i] != 0.0 else "NaN")
                s += " {0:^5} |".format(self.result_grid[i][j])
                if j == len(self.list_2)-1:
                    s += "\n"
        s += "="*len(s.split("\n")[-2]) + "\n"
        s = "\n" + "="*len(s.split("\n")[-2]) + s
        return s

def vis_grid_search(grid):
    hyperparams = []
    for i in range(len(grid)):
        h = grid.get(i)
        hyperparams.append(h)
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

def test_grid():
    lr_list = [(i/10) for i in range(1, 10+1)]
    batch_size_list = [16, 32, 64, 128, 256]
    grid = Grid(lr_list, batch_size_list)
    hyperparams = []
    results = []
    for i, lr in enumerate(lr_list):
        for j, batch_size in enumerate(batch_size_list):
            hyperparams.append((lr, batch_size))
            results.append(lr)
    print(grid)
    vis_search(hyperparams, results)

if __name__ == "__main__":
    test_grid()


