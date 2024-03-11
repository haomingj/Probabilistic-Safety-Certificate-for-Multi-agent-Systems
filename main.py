import numpy as np
from agent import Agent
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.curandom as curandom
from pycuda.compiler import SourceModule
import time
import sys

# TODO: figure out other arguments
def main(save_directory):
    # max simulation time
    # this is in time step, not in seconds
    tmax = 5000
    playground = [-50,50,-50,50]
    dt = 0.01
    control_limit = [-30,30,-6,6]

    # time horizon for safety certificate
    T = 80

    # number of samples used to estimate safe probability
    # due to limit in CUDA, maximum K is 1024
    # for now, K must be a multiple of 8
    K = 800

    # dx for finite difference estimation of gradient
    DX = 2*[0.5, 0.5, 0.5, 0.4]

    # minimum allowed distance between any 2 agents
    safety_threshold = 2

    # gain for nominal controller
    control_gain = 1

    # max speed for nominal controller
    max_speed = 10

    # noise variance in the system
    sigma = [10, 10]
    
    # safe probability threshold
    epsilon = 0.05

    # how far an agent can communicate
    communication_range = 15

    # if the agent reaches within this distance to the goal, it is considered as reaching the goal
    goal_range = 5
    start_locations = np.zeros((0,4))
    goals = np.zeros((0,2))
    rng = curandom.XORWOWRandomNumberGenerator()


    n_agent = 30
    min_start_separation = 10
    too_close = True
    agent_list = []
    current_id = 0
    for i in range(n_agent):
        while too_close:
            this_start = [np.random.uniform(playground[0],playground[1]),np.random.uniform(playground[2],playground[3]),0,0]
            this_goal = [np.random.uniform(playground[0],playground[1]),np.random.uniform(playground[2],playground[3])]
            if (this_start[0]-this_goal[0])**2 + (this_start[1]-this_goal[1])**2 > min_start_separation**2:
                too_close = False
                for j in range(i):
                    if (this_start[0]-start_locations[j,0])**2 + (this_start[1]-start_locations[j,1])**2 < min_start_separation**2 or (this_start[0]-goals[j,0])**2 + (this_start[1]-goals[j,1])**2 < min_start_separation**2:
                        too_close = True
                        break
        agent_list.append(Agent(this_start,control_limit,0,this_goal,T,dt,K,current_id,DX,safety_threshold,control_gain,max_speed,sigma,epsilon,communication_range,goal_range,save_directory))
        start_locations = np.vstack((start_locations, this_start))
        goals = np.vstack((goals, this_goal))
        current_id += 1
        too_close = True
    
    for t in range(tmax):
        t0 = time.time()
        for i in range(n_agent):
            if agent_list[i].check_if_on():
                agent_list[i].start_round()
        for i in range(n_agent):
            if agent_list[i].check_if_on():
                agent_list[i].run_step(agent_list,rng)
                # print(t)
        t1 = time.time()
        print(t)
        print("Time per cycle: ", t1-t0)
    for i in range(n_agent):
        agent_list[i].finish_round()
    for i in range(n_agent):
        trajectory, control, safe_probability, safe_probability_left, safe_probability_right, goal = agent_list[i].snapshot()
        with open(save_directory + "/trajectory"+str(i)+".npy", "wb") as f:
            np.savez(f, trajectory, control, safe_probability, safe_probability_left, safe_probability_right, goal)

if __name__ == "__main__":
    main(sys.argv[1])