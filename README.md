# Usage Local
	mpiexec -n 4 python train_mnist_grid_search.py
	mpiexec -n 4 python train_mnist_random_search.py 

# On Cluster 
	mpiexec.mpich -hostfile hostfile_3 python3 train_mnist_grid_search.py -DEBUG True
	mpiexec.mpich -hostfile hostfile_9 python3 train_mnist_grid_search.py -DEBUG True
	mpiexec.mpich -hostfile hostfile_3 python3 train_mnist_random_search.py -DEBUG True
	mpiexec.mpich -hostfile hostfile_9 python3 train_mnist_random_search.py -DEBUG True
