### Grid Search
mpiexec.mpich -hostfile hostfile_1 python3 train_mnist_grid_search.py --grid_size=10 
mpiexec.mpich -hostfile hostfile_3 python3 train_mnist_grid_search.py --grid_size=10 

## Random Search
mpiexec.mpich -hostfile hostfile_1 python3 train_mnist_random_search.py --n_search=100
mpiexec.mpich -hostfile hostfile_3 python3 train_mnist_random_search.py --n_search=100

## Evoluation Search
mpiexec.mpich -hostfile hostfile_1 python3 train_mnist_evolution_search.py --pop_size=10 --n_gen=9 
mpiexec.mpich -hostfile hostfile_3 python3 train_mnist_evolution_search.py --pop_size=10 --n_gen=9 
