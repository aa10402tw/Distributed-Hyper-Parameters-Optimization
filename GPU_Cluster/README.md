# Jupyter notebook to cluster
Remote Activate Anaconda environment
	cd ~/anaconda3/bin
	source activate base

Remote 
	ssh 0756079@140.113.215.195 -p 37031
	cd ~/PP19_Final/Distributed-Hyper-Parameters-Optimization/GPU_Cluster
	jupyter notebook --no-browser --port=8888
	
Local
	ssh -N -L localhost:8888:localhost:8888 0756079@140.113.215.195 -p 37031

# Execute 
	/home/students/0756079/openmpi/v3.0.0/bin/mpiexec -hostfile hostfile /home/students/0756079/anaconda3/bin/python mpi_test.py
	
