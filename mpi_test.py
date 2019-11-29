from mpi4py import MPI
import time
import random
from tqdm import tqdm
from mnist_utils import *

MASTER_RANK = 0

class MPIWorld:
    def __init__(self, comm, world_size, my_rank, MASTER_RANK=0, node_name=""):
        self.comm = comm
        self.world_size = world_size
        self.my_rank = my_rank
        self.MASTER_RANK = MASTER_RANK
        self.node_name = node_name
        
    def isMaster(self):
        return self.my_rank == self.MASTER_RANK

def initMPI():
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    my_rank = comm.Get_rank()
    node_name = MPI.Get_processor_name()
    print("Process ({1}/{2}) : {0} ".format(node_name, my_rank+1, world_size))
    return MPIWorld(comm, world_size, my_rank, MASTER_RANK=0, node_name=node_name)



mpiWorld = initMPI()
