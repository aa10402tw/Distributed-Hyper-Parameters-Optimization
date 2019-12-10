## Connect to Server
	ssh pp2019@140.113.216.71

## Update Git
	mv ~/nfs/Distributed-Hyper-Parameters-Optimization/Our_Cluster/data ~/nfs
	cd ~/nfs
	rm -rf Distributed-Hyper-Parameters-Optimization
	git clone https://github.com/aa10402tw/Distributed-Hyper-Parameters-Optimization.git
	mv ~/nfs/data ~/nfs/Distributed-Hyper-Parameters-Optimization/Our_Cluster
	cd ~/nfs/Distributed-Hyper-Parameters-Optimization/Our_Cluster

## Comparsion Between Search Methods
	mpiexec.mpich -hostfile hostfile_3 python3 search_comparsion.py 
--grid_size=0 to skip grid_search, 
--n_random_search=0 to skip random_search,
--n_gen=0 to skip evoluation_search,

## Comparsion Between Different Number of Nodes

### Grid Search
	mpiexec.mpich -hostfile hostfile_1 python3 train_mnist_grid_search.py --grid_size=10 
	mpiexec.mpich -hostfile hostfile_3 python3 train_mnist_grid_search.py --grid_size=10 

### Random Search
	mpiexec.mpich -hostfile hostfile_1 python3 train_mnist_random_search.py --n_search=100
	mpiexec.mpich -hostfile hostfile_3 python3 train_mnist_random_search.py --n_search=100

### Evoluation Search
	mpiexec.mpich -hostfile hostfile_1 python3 train_mnist_evolution_search.py --pop_size=10 --n_gen=9 
	mpiexec.mpich -hostfile hostfile_3 python3 train_mnist_evolution_search.py --pop_size=10 --n_gen=9 

other flags: --DEBUG=True for debug, --exp=True for experiment
