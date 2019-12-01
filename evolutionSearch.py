import os
import glob
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from mnist_utils import *
import copy

from hyperparams import *

class EvoluationSearch:
    def __init__(self, hparams, population_size=10, prob_crossover=0.8, prob_mutation=0.2):
        self.hparams = hparams
        self.population = [self.hparams.copy().initValue() for i in range(population_size)]
        self.prob_crossover = prob_crossover
        self.prob_mutation  = prob_mutation
        self.resultDict = {}

    def crossover(self, population):
        population_crossover = []
        for idx1, hp1 in enumerate(population):
            if random.uniform(0, 1) <= self.prob_crossover:
                idx2 = random.choices([i2 for i2 in range(len(population)) if i2 != idx1])[0]
                hp2 = population[idx2]
                n_rv_hp1 = random.randint(1, len(hp1)-1)
                rvs_hp1 = hp1[:n_rv_hp1]
                rvs_hp2 = hp2[n_rv_hp1:]
                child = HyperParams(rvs_hp1+rvs_hp2)
                population_crossover.append(child)
        return population_crossover

    def mutation(self, population):
        population_mutation = []
        mean = 0
        std = 0
        for hps in population:
            for i, rv in enumerate(hps):
                hps_new = hps.copy()
                if random.uniform(0, 1) <= self.prob_mutation:
                    if isinstance(rv, CRV):
                        mean = rv.value
                        std = np.std( [hps.rvs[i].value for hps in population] )
                        hps_new[i].value = np.random.normal(mean, std, 1).clip(
                            rv.low, rv.high).item()
                    if isinstance(rv, DRV):
                        hps_new[i].value = random.choices(rv.choices)[0]
                if hps_new != hps:
                    population_mutation.append(hps_new)
        return population_mutation

    def get_unevaluated_population(self, population):
        unevaluated_population = []
        for hyperparams in population:
            key = hyperparams
            if key not in self.resultDict:
                unevaluated_population.append(hyperparams)
        return unevaluated_population

    def evaluate_population(self, population, mpiWorld, pbars, DEUBG):
        # Sactter Population
        local_population = [] 
        local_fitness =  []
        if mpiWorld.isMaster():
            n = len(population) / (mpiWorld.world_size)
            for i in range(mpiWorld.world_size):
                s = int(i*n) if i != 0 else 0
                e = int((i+1)*n) if i != mpiWorld.world_size-1 else len(population)
                local_population.append(population[s:e])
        local_population = mpiWorld.comm.scatter(local_population, root=mpiWorld.MASTER_RANK)
        mpiWorld.comm.barrier()

        # Evaluate local population
        for i, hps in enumerate(local_population):
            if mpiWorld.isMaster():
                pbars['search'].set_postfix({
                    "Best":"{:.2f}%".format(get_best_acc(resultDict)), 
                    "Progress":"({}/{})".format(i+1, len(local_population))
                })

            
            # Train MNIST
            acc = train_mnist(hps, device=device, pbars=pbars, DEBUG=DEUBG)
            local_fitness.append(acc)

        # Gather Population and Fitness
        local_population_gather = mpiWorld.comm.gather(local_population, root=mpiWorld.MASTER_RANK)
        local_fitness_gather    = mpiWorld.comm.gather(local_fitness, root=mpiWorld.MASTER_RANK)
        
        if mpiWorld.isMaster():
            population = []
            fitness = []
            for (pop, fit) in zip(local_population_gather, local_fitness_gather):
                population += pop
                fitness    += fit
            return population, fitness
        else:
            return None, None


    def generation(self, mpiWorld, pbars, DEUBG=False):
        population_dict = {}
        if mpiWorld.isMaster():
            # Crossover and Mutation
            population = self.population
            population_crossover = self.crossover(population)
            population_mutation  = self.mutation(population+population_crossover)
            population_dict['start'] = population
            population_dict['crossover'] = population_crossover
            population_dict['mutation']  = population_mutation
            population = list(set(population+population_crossover+population_mutation))
            unevaluated_population = self.get_unevaluated_population(population)
        else:
            unevaluated_population = None
        unevaluated_population, fitness = evaluate_population(unevaluated_population, mpiWorld, pbars, DEUBG)
        if mpiWorld.isMaster():
            # Selection
            # Update resultDict
            for hparams, acc in zip(unevaluated_population, fitness):
                if hparams not in resultDict:
                    key = hparams
                    resultDict[key] = acc
            # Select best
            scores = []
            for hparams in population:
                key = hparams
                scores.append(resultDict[key])
            sort_idx = np.argsort(scores)
            scores = [scores[i.item()] for i in sort_idx][-self.population_size:][::-1]
            population = [population[i.item()] for i in sort_idx][-self.population_size:][::-1]
            population_dict['selcetion'] = population

        return population_dict

mpiWorld = initMPI()
lr = CRV(low=0.0, high=1.0, name="lr")
dr = CRV(low=0.0, high=1.0, name="dr")
hparams = HyperParams([lr, dr])
evoluationSearch = EvoluationSearch(hparams, population_size=5)




# def crossover(population, prob_crossover=0.8):
#     new_population = []
#     for i, hparams_1 in enumerate(population):
#         if random.uniform(0, 1) <= prob_crossover:
#             idx_2 = random.choice([j for j in range(len(population)) if j != i])
#             hparams_2 = population[idx_2]
#             num_rv_from_h1 = random.randint(1, len(hparams_1))
#             rvs = []
#             for i in range(len(hparams_1)):
#                 if i < num_rv_from_h1:
#                     rvs.append(hparams_1[i].copy())
#                 else:
#                     rvs.append(hparams_2[i].copy())
#             child = HyperParams(rvs)
#             new_population.append(child)
#     return new_population

# def mutation(population, prob_mutation=0.2):
#     new_population = []
#     for hparams in population:
#         new_rvs = []
#         for i, rv in enumerate(hparams.rvs):
#             if random.uniform(0, 1) <= prob_mutation:
#                 if isinstance(rv, CRV):
#                     mean = rv.value
#                     std = np.std( [h.rvs[i].value for h in population] )
#                     rv_new = rv.copy()
#                     rv_new.value = np.random.normal(mean, std, 1).clip(
#                         rv.low, rv.high).item()
#                     new_rvs.append(rv_new)
#                 if isinstance(rv, DRV):
#                     rv_new = rv.copy()
#                     rv_new.value = random.choices(rv.choices)[0]
#                     new_rvs.append(rv_new)
#             else:
#                 new_rvs.append(rv.copy())
#         hparams_new = HyperParams(new_rvs)
#         if hparams_new != hparams:
#             new_population.append(hparams_new)
#     return new_population

def vis_population(population, label="", c=None):
    if len(population) == 0:
        return 
    xs = [hparams.rvs[0].value for hparams in population]
    ys = [hparams.rvs[1].value for hparams in population]
    if c is not None:
        plt.scatter(xs, ys, label=label, c=c)
    else:
         plt.scatter(xs, ys, label=label)
    if isinstance(population[0].rvs[0], DRV):
        plt.xticks(list(set(xs)))
    if isinstance(population[0].rvs[1], DRV):
        plt.yticks(list(set(ys)))

# def fitness_test(hparams):
#     e1, e2 = 0, 0
#     for i, rv in enumerate(hparams):
#         if isinstance(rv, CRV):
#             e1 += (0.75-rv.value)**2
#             e2 += (0.25-rv.value)**2
#         if isinstance(rv, DRV):
#             if rv.value == 32:
#                 e1 -= 0.5
#             if rv.value == 64:
#                 e2 -= 0.5
#     return min(e1, e2)

# def selection_test(population, k):
#     fitness_list = list(map(fitness_test, population))
#     sort_idx = np.argsort(fitness_list)
#     good_population = [population[i.item()] for i in sort_idx][:k]
#     return good_population

# def test_evolution():
#     population_size = 30
#     num_generation = 100

#     population = []
#     for i in range(population_size):
#         lr = CRV(low=0, high=1, name="lr")
#         bs = DRV(choices=[16, 32, 64, 128, 256], name="bs")
#         hparams = HyperParams([lr, bs])
#         population.append(hparams)

#     plt.subplot(331)
#     vis_population(population, label="init")
#     plt.legend(loc='best')

#     for gen in range(1, num_generation+1):
        
#         if  gen > 8: 
#             population_crossover = crossover(population)
#             population = population + population_crossover
#             population_mutation = mutation(population)
#             population = population + population_mutation
#             population_selection = selection_test(population, population_size)
#             population = population_selection
#         else:
#             plt.subplot(3, 3, gen+1)
#             # Crossover
#             population_crossover = crossover(population)
#             vis_population(population_crossover, label="crossover", c="red")
#             population = population + population_crossover
#             #        
#             population_mutation = mutation(population)
#             vis_population(population_mutation, label="mutation", c="purple")
#             population = population + population_mutation

#             population_selection = selection(population, population_size)
#             vis_population(population_selection, label="selection", c="green")
#             plt.legend(loc='best')
#             plt.ylim(-1,257)
#             plt.xlim(0, 1)
#             population = population_selection

#     plt.subplot(3, 3, 9)
#     plt.ylim(-1,257)
#     plt.xlim(0, 1)
#     vis_population(population, label="final")
#     plt.legend(loc='best')
#     plt.show()

# if __name__ == "__main__":
#     # test_evolution()
#     lr = CRV(low=0, high=1, name="lr")
#     bs = DRV(choices=[16, 32, 64, 128, 256], name="bs")
#     hparams = HyperParams([lr, bs])
#     hparams_2 = HyperParams([lr, bs])
#     mydict = {hparams:0}
#     # print(mydict[hparams_2])
#     test_evolution()



