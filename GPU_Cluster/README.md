# Update Git
	mv ~/PP19_Final/Distributed-Hyper-Parameters-Optimization/GPU_Cluster/data ~/PP19_Final/
	cd ~/PP19_Final/
	rm -rf Distributed-Hyper-Parameters-Optimization 
	git clone https://github.com/aa10402tw/Distributed-Hyper-Parameters-Optimization.git
	mv ~/PP19_Final/data ~/PP19_Final/Distributed-Hyper-Parameters-Optimization/GPU_Cluster/
	cd ~/PP19_Final/Distributed-Hyper-Parameters-Optimization/GPU_Cluster/
	mkdir result

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

### Alias
	alias cuda_mpiexec_1="/home/students/0756079/openmpi/v3.0.0/bin/mpiexec -hostfile hostfiles/gpu_hostfile_1 /home/students/0756079/anaconda3/bin/python"
	alias cuda_mpiexec_2="/home/students/0756079/openmpi/v3.0.0/bin/mpiexec -hostfile hostfiles/gpu_hostfile_2 /home/students/0756079/anaconda3/bin/python"
	alias cuda_mpiexec_3="/home/students/0756079/openmpi/v3.0.0/bin/mpiexec -hostfile hostfiles/gpu_hostfile_3 /home/students/0756079/anaconda3/bin/python"
	alias cuda_mpiexec_4="/home/students/0756079/openmpi/v3.0.0/bin/mpiexec -hostfile hostfiles/gpu_hostfile_4 /home/students/0756079/anaconda3/bin/python"
# Execute .py file

	cuda_mpiexec_4 cuda_mpi_test.py
	cuda_mpiexec_4 train_cifar10_grid_search.py --grid_size=2 --num_epochs=1 --DEBUG=True
	cuda_mpiexec_4 train_cifar10_random_search.py --n_search=4 --num_epochs=1 --DEBUG True
	cuda_mpiexec_4 train_cifar10_evolution_search.py --n_gen=1 --pop_size=4 --num_epochs=1 --DEBUG=True
	
