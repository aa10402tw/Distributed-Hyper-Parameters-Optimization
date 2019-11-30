# Usage Local
	mpiexec -n 4 python train_mnist_grid_search.py
	mpiexec -n 4 python train_mnist_random_search.py 

# On Cluster 
	mpiexec.mpich -hostfile hostfile train_mnist_grid_search.py -DEBUG True
