from search_comparsion_utils import *

if __name__ == "__main__":
    start = time.time()

    # === Init MPI World === #
    mpiWorld = initMPI()
    mpiWorld.comm.Barrier()
    logs = mpiWorld.comm.gather(mpiWorld.log, root=mpiWorld.MASTER_RANK)
    if mpiWorld.isMaster():
        check_dataset()
        print("\n=== MPI World ===")
        for log in logs:
            print(log)
    mpiWorld.comm.Barrier()

    # === Argument === #
    parser = ArgumentParser()
    parser = ArgumentParser()
    parser.add_argument("--DEBUG",  default=False)
    parser.add_argument("--exp",  default=False)
    parser.add_argument("--criteria",  default=95.0, type=float)
    parser.add_argument("--n_comparsion", default=10, type=int)

    # Grid Search: 5*5*5 = 125
    parser.add_argument("--grid_size",  default=5, type=int)  
    # Random Search: 100 
    parser.add_argument("--n_random_search",  default=100, type=int) 
    # Evoluation Search: 10+10*9 = 100
    parser.add_argument("--pop_size",  default=10, type=int)         
    parser.add_argument("--n_gen",  default=10, type=int) 
    args = parser.parse_args()
    args.DEBUG = str2bool(args.DEBUG)
    args.exp = str2bool(args.exp)
    if mpiWorld.isMaster():
        print("\n=== Args ===")
        print("Args:{}\n".format(args))

    SAVE_NAME_GRID       = "result/search_comparsion/time_grid.txt"
    SAVE_NAME_RANDOM     = "result/search_comparsion/time_random.txt"
    SAVE_NAME_EVOLUATION = "result/search_comparsion/time_evoluation.txt"
    
    for i in range(args.n_comparsion):
        # Grid Search 
        start = time.time()
        acc_grid = grid_search(mpiWorld, args)
        end = time.time()
        time_elapsed_grid = end-start
        if mpiWorld.isMaster() and args.exp:
            print("time_elapsed_grid:", time_elapsed_grid, "acc:", acc_grid)

        # Random Search
        start = time.time()
        acc_ran = random_search(mpiWorld, args)
        end = time.time()
        time_elapsed_ran = end-start
        if mpiWorld.isMaster() and args.exp:
            print("time_elapsed_random:", time_elapsed_ran, "acc:", acc_ran)

        # Evoluation Search
        start = time.time()
        acc_evo = evoluation_search(mpiWorld, args)
        end = time.time()
        time_elapsed_evo = end-start
        if mpiWorld.isMaster() and args.exp:
            print("time_elapsed_evoluation:", time_elapsed_evo, "acc:", acc_evo)

        if mpiWorld.isMaster():
            write_time_log(time_elapsed_grid, acc_grid, SAVE_NAME_GRID)
            write_time_log(time_elapsed_ran,  acc_ran,  SAVE_NAME_RANDOM)
            write_time_log(time_elapsed_evo,  acc_evo,  SAVE_NAME_EVOLUATION)
