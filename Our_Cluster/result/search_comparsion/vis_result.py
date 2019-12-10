import numpy as np
import matplotlib.pyplot as plt

def read_time_txt(file_name):
    time_list = []
    with open(file_name, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i == 0:
                continue
            time, acc = line.split(",")
            time_list.append(float(time))
    return np.mean(time_list), np.std(time_list)

if __name__ == "__main__":
    grid_mean, grid_std = read_time_txt("time_grid.txt")
    ran_mean, ran_std = read_time_txt("time_random.txt")
    evo_mean, evo_std = read_time_txt("time_evoluation.txt")

    title = "|{:^15}|{:^15}|{:^15}|".format("", "mean", "std")
    print("="*len(title))
    print(title)
    print("="*len(title))
    print("|{:^15}|{:^15}|{:^15}|".format("Grid", "%.2f"%grid_mean, "%.2f"%grid_std))
    print("|{:^15}|{:^15}|{:^15}|".format("Random", "%.2f"%ran_mean, "%.2f"%ran_std))
    print("|{:^15}|{:^15}|{:^15}|".format("Evoluation", "%.2f"%evo_mean, "%.2f"%evo_std))
    print("="*len(title))
    positions = [0,1,2]
    plt.bar(positions, [grid_mean, ran_mean, evo_mean],
        yerr=[grid_std, ran_std, evo_std])
    plt.savefig("search_comparsion.png")

