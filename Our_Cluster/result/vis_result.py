import os
import numpy as np
import matplotlib.pyplot as plt

def read_txt(file_name):
	time_list = []
	acc_list  = []
	if not os.path.isfile(file_name):
		return (0, 0), (0, 0)
	with open(file_name, "r") as f:
		lines = f.readlines()
		for i, line in enumerate(lines):
			if i == 0:
				continue
			time = line.split(',')[0]
			acc  = line.split(',')[1]
			time_list.append(float(time))
			acc_list.append(float(acc))
	return (np.mean(time_list), np.std(time_list)), (np.mean(acc_list), np.std(acc_list))


GRID_ACC_NAME = "acc_grid_search.txt" 
RAN_ACC_NAME  = "acc_ran_search.txt"
EVO_ACC_NAME  = "acc_evo_search.txt"

GRID_TIME_NAME = "time_grid_search.txt" 
RAN_TIME_NAME  = "time_ran_search.txt"
EVO_TIME_NAME  = "time_evo_search.txt"

names = ['Grid', 'Random', 'Evoluation']
positions = [0, 1, 2]

grid_acc_mean, grid_acc_std = read_txt(GRID_ACC_NAME)[1]
ran_acc_mean, ran_acc_std = read_txt(RAN_ACC_NAME)[1]
evo_acc_mean, evo_acc_std = read_txt(EVO_ACC_NAME)[1]

acc_means = [grid_acc_mean, ran_acc_mean, evo_acc_mean]
acc_stds  = [grid_acc_std, ran_acc_std, evo_acc_std]

grid_time_mean, grid_time_std = read_txt(GRID_TIME_NAME)[0]
ran_time_mean, ran_time_std = read_txt(RAN_TIME_NAME)[0]
evo_time_mean, evo_time_std = read_txt(EVO_TIME_NAME)[0]

time_means = [grid_time_mean, ran_time_mean, evo_time_mean]
time_stds  = [grid_time_std, ran_time_std, evo_time_std]

plt.subplot(1,2,1)
plt.title("Accuracy")
plt.bar(positions, acc_means, yerr=acc_stds)
plt.xticks(positions, names)

plt.subplot(1,2,2)
plt.title("Time")
plt.bar(positions, time_means, yerr=time_stds)
plt.xticks(positions, names)
# plt.show()
plt.savefig("Comparsion_Acc_Time.png")


title = "|{:^20}|{:^20}|{:^20}|{:^20}|".format(
	"Accuracy", names[0], names[1], names[2])
row_1 = "|{:^20}|{:^20}|{:^20}|{:^20}|".format(
	"Mean", "%.2f"%acc_means[0], "%.2f"%acc_means[1], "%.2f"%acc_means[2])
row_2 = "|{:^20}|{:^20}|{:^20}|{:^20}|".format(
	"Std", "%.2f"%acc_stds[0], "%.2f"%acc_stds[1], "%.2f"%acc_stds[2])

print("="*len(title))
print(title)
print("="*len(title))
print(row_1)
print("-"*len(title))
print(row_2)
print("="*len(title))
print()

title = "|{:^20}|{:^20}|{:^20}|{:^20}|".format(
	"Minimum Time", names[0], names[1], names[2])
row_1 = "|{:^20}|{:^20}|{:^20}|{:^20}|".format(
	"Mean", "%.2f"%time_means[0], "%.2f"%time_means[1], "%.2f"%time_means[2])
row_2 = "|{:^20}|{:^20}|{:^20}|{:^20}|".format(
	"Std", "%.2f"%time_stds[0], "%.2f"%time_stds[1], "%.2f"%time_stds[2])

print("="*len(title))
print(title)
print("="*len(title))
print(row_1)
print("-"*len(title))
print(row_2)
print("="*len(title))
print()

