from mpi4py import MPI
import time
import random
from tqdm import tqdm
from mnist_utils import *

MASTER_RANK = 0

if __name__ == "__main__":
	comm, world_size, my_rank, node_name = initMPI()

	idxs = [i for i in range(my_rank*10,  my_rank*10+10)]
	dataDict = {}
	comm.barrier()
	if my_rank == MASTER_RANK:
		pbar = tqdm(total = 10*world_size)

	for idx in idxs:
		dataDict[idx] = "OUO"
		time.sleep(random.uniform(0,1))
		syncData(dataDict, comm, world_size, my_rank, MASTER_RANK, blocking=False)
		if my_rank == MASTER_RANK:
			pbar.update(len(dataDict) - pbar.n)
	syncData(dataDict, comm, world_size, my_rank, MASTER_RANK, blocking=True)
	if my_rank == MASTER_RANK:
		pbar.update(len(dataDict) - pbar.n)
	comm.barrier()
