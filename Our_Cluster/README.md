## Connect to Server
	ssh pp2019@140.113.216.71

## Update Git
	mv ~/nfs/Distributed-Hyper-Parameters-Optimization/Our_Cluster/data ~/nfs
	cd ~/nfs
	rm -rf Distributed-Hyper-Parameters-Optimization
	git clone https://github.com/aa10402tw/Distributed-Hyper-Parameters-Optimization.git
	mv ~/nfs/data ~/nfs/Distributed-Hyper-Parameters-Optimization/Our_Cluster
	cd ~/nfs/Distributed-Hyper-Parameters-Optimization/Our_Cluster

### Alias 
	alias mpi_py_1="mpiexec.mpich -hostfile hostfiles/hostfile_1 python3"
	alias mpi_py_3="mpiexec.mpich -hostfile hostfiles/hostfile_3 python3"

# Comparsion Between Search Methods
### Accuracy (Same number of hyper-parameters evaluated)
	mpi_py_3 acc_benchmark.py --n_comparsion=1 --grid_size=5 --n_search=90 --pop_size=6 --n_gen=14
### Termination Time (Reach given criteria)
	mpi_py_3 time_benchmark.py --TIME_LIMIT=1200 --n_comparsion=1 --grid_size=10 --n_search=999 --pop_size=6 --n_gen=165
### See the result 
	cd result/
	python3 vis_result.py

# Comparsion Between Different Number of Nodes
### Grid Search
	mpi_py_1 train_mnist_grid_search.py --grid_size=10 
	mpi_py_3 train_mnist_grid_search.py --grid_size=10 

### Random Search
	mpi_py_1 train_mnist_random_search.py --n_search=100
	mpi_py_3 train_mnist_random_search.py --n_search=100

### Evoluation Search
	mpi_py_1 train_mnist_evolution_search.py --pop_size=10 --n_gen=9 
	mpi_py_3 train_mnist_evolution_search.py --pop_size=10 --n_gen=9 

other flags: --DEBUG=True for debug, --exp=True for experiment
