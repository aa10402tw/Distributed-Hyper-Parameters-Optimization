# Jupyter notebook on remote cluster

## Remote Activate Anaconda environment
	cd ~/anaconda3/bin
	source activate base

## Remote command
	ssh 0756079@140.113.215.195 -p 37031
	cd ~/PP19_Final/Distributed-Hyper-Parameters-Optimization/GPU_Cluster
	jupyter notebook --no-browser --port=8888
	
## Local Command
	ssh -N -L localhost:8888:localhost:8888 0756079@140.113.215.195 -p 37031

# Execute .py file
	alias cuda_mpiexec="/home/students/0756079/openmpi/v3.0.0/bin/mpiexec -hostfile hostfile /home/students/0756079/anaconda3/bin/python"
	cuda_mpiexec cuda_mpi_test.py
	cuda_mpiexec train_cifar10_grid_search.py --grid_size=2 --num_epochs=1 --DEBUG=True
	cuda_mpiexec train_cifar10_random_search.py --n_search=4 --num_epochs=1 --DEBUG True
	cuda_mpiexec train_cifar10_evolution_search.py --n_gen=1 --pop_size=4 --num_epochs=1 --DEBUG=True
	
