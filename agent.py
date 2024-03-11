import numpy as np
import random
# pycuda is the current GPU interface
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.curandom as curandom
from pycuda.compiler import SourceModule
from pathlib import Path
import time
import os

class Agent:
    def __init__(self, start_location, control_limit, timestamp, goal, T, dt, K, id, DX, safety_threshold, control_gain, max_speed, sigma, epsilon, communication_range, goal_range, save_directory):
        self.control_limit = control_limit
        # timestamp not used yet
        self.timestamp = timestamp

        # state is [x, y, v, theta]
        # [x position, y position, speed, heading angle]
        self.state = np.array(start_location)
        self.trajectory = np.array(start_location)
        self.control_trajectory = np.zeros((0,2))
        self.safe_probability_trajectory = np.zeros((0,1))
        self.safe_probability_left_trajectory = np.zeros((0,4))
        self.safe_probability_right_trajectory = np.zeros((0,4))
        self.goal = goal
        self.vmax = max_speed
        self.gain = control_gain
        self.safety_threshold = safety_threshold
        self.sigma = sigma

        # the maximum distance that one agent can get other agents' states
        self.communication_range = communication_range

        # if the agent reaches the goal within this distance, it is considered to have reached the goal
        self.goal_range = goal_range

        # sets control limits
        # u1 is acceleration
        # u2 is turning rate
        self.u1min = control_limit[0]
        self.u1max = control_limit[1]
        self.u2min = control_limit[2]
        self.u2max = control_limit[3]

        # turn agent on
        self.on = True
        self.T = T
        self.dt = dt
        self.K = K

        # delta x used in finite difference approximation for gradient
        # there are 4 states in the system
        self.DX1 = DX[0]
        self.DX2 = DX[1]
        self.DX3 = DX[2]
        self.DX4 = DX[3]

        # assign id to agent
        self.id = id

        # 
        self.control_available = False

        # safety threshold
        self.epsilon = epsilon

        # parameters to be passed into GPU
        self.parameters = np.array([self.dt, self.DX1, self.DX2, self.DX3, self.DX4, self.safety_threshold, self.T, self.K, self.gain, self.vmax, self.sigma[0], self.sigma[1]], dtype=np.float32)
        self.parameters_cuda = cuda.mem_alloc(self.parameters.nbytes)
        cuda.memcpy_htod(self.parameters_cuda, self.parameters)

        # grab cuda functions
        self.cuda_source_code = Path("psi_gradient.cu").read_text()
        self.cuda_module = SourceModule(self.cuda_source_code)
        self.dynamic_evolution = self.cuda_module.get_function("dynamic_evolution")
        self.identify_collision = self.cuda_module.get_function("identify_collision")
        self.identify_safety = self.cuda_module.get_function("identify_safety")
        self.dynamic_evolution_no_gradient = self.cuda_module.get_function("dynamic_evolution_no_gradient")
        self.identify_collision_no_gradient = self.cuda_module.get_function("identify_collision_no_gradient")
        self.identify_safety_along_n = self.cuda_module.get_function("identify_safety_along_n")
        self.identify_safety_along_T = self.cuda_module.get_function("identify_safety_along_T")

        # save directory, used for saving data in batch runs
        self.save_directory = save_directory

    def start_round(self):
        # at start of a time step, make control unavailable until control action is generated
        self.control_available = False

    def get_id(self):
        return self.id
    
    def set_goal(self, goal):
        self.goal = goal

    def get_goal(self):
        return self.goal

    def check_if_on(self):
        return self.on
    
    def check_if_accessible(self, state, communication_range):
        # accessibility is determined based on the distance between two agents and the communication range of the agent asking for state
        if np.linalg.norm(state[0:2]-self.state[0:2]) < communication_range:
            # accessible only if the agent is on and the agent has already made its control decision in this time step
            return self.on and self.control_available
        else:
            return False
    
    def get_state(self):
        return self.state
    
    def get_trajectory(self):
        return self.trajectory
    
    def observe_state(self, state):
        # TODO
        # temporary: full state
        return self.state
    
    def nominal_control(self):
        u1 = self.gain*(self.vmax - self.state[2])
        theta_ref = np.arctan2(self.goal[1]-self.state[1], self.goal[0]-self.state[0])
        if np.abs(theta_ref - self.state[3]) < np.pi:
            u2 = self.gain*(theta_ref - self.state[3])
        else:
            u2 = self.gain*(self.state[3] - theta_ref)
        u1 = np.clip(u1, self.u1min, self.u1max)
        u2 = np.clip(u2, self.u2min, self.u2max)
        return np.array([u1, u2])
    
    def observe_control(self, state):
        return self.control
    
    def evolve_dynamics(self, state, control):
        # this is the f(x) + g(x)u term in continuous time, without dt or noise
        return_fg = np.zeros(4)
        return_fg[0] = np.cos(state[3])*state[2]
        return_fg[1] = np.sin(state[3])*state[2]
        return_fg[2] = control[0]
        return_fg[3] = control[1]
        return return_fg

    def finish_round(self):
        # write the agent trajectory to file when one round of simulation is finished
        with open(self.save_directory + "/agent"+str(self.id)+".npy", "wb") as f:
            np.save(f, self.trajectory)

    def snapshot(self):
        # take a snapshot of the simulation data
        return self.trajectory, self.control_trajectory, self.safe_probability_trajectory, self.safe_probability_left_trajectory, self.safe_probability_right_trajectory, self.goal

    def run_step(self, agent_list, rng):
        # gamma function as defined in equation (18)
        # this is a rather arbitrary choice of gamma function
        gamma = lambda x: x-1

        def find_feasible_control(k0,k1,k2,u1ref,u2ref,u1min,u1max,u2min,u2max):
            # finds the feasible control action given the control limit and safety condition
            # it is possible that the safety condition cannot be satisfied, given the control limit
            eps = 0.000001
            if (k1 < eps) and (k1 > -eps) and (k2 < eps) and (k2 > -eps):
                return np.array([u1ref, u2ref])
            elif (k2 < eps) and (k2 > -eps):
                # print("Coeficient not zero")
                # print("Safe probability: ", safety_self_mean)
                if k1 > 0:
                    return np.array([min(u1max, -k0/k1), u2ref])
                else:
                    return np.array([max(u1min, -k0/k1), u2ref])
            elif (k1 < eps) and (k1 > -eps):
                # print("Coeficient not zero")
                # print("Safe probability: ", safety_self_mean)
                if k2 > 0:
                    return np.array([u1ref, min(u2max, -k0/k2)])
                else:
                    return np.array([u1ref, max(u2min, -k0/k2)])
            # print("Coeficient not zero")
            # print("Safe probability: ", safety_self_mean)
            no_intersection = True
            min_dist = np.inf
            u1_candidate = []
            u2_candidate = []
            temp = (-k1*u1min-k0)/k2
            if (temp >= u2min) and (temp <= u2max):
                u1_candidate.append(u1min)
                u2_candidate.append(temp)
                no_intersection = False
            temp = (-k1*u1max-k0)/k2
            if (temp >= u2min) and (temp <= u2max):
                u1_candidate.append(u1max)
                u2_candidate.append(temp)
                no_intersection = False
            temp = (-k2*u2min-k0)/k1
            if (temp >= u1min) and (temp <= u1max):
                u1_candidate.append(temp)
                u2_candidate.append(u2min)
                no_intersection = False
            temp = (-k2*u2max-k0)/k1
            if (temp >= u1min) and (temp <= u1max):
                u1_candidate.append(temp)
                u2_candidate.append(u2max)
                no_intersection = False
            if no_intersection:
                dist = np.abs(k0+k1*u1min+k2*u2min)/np.sqrt(k1**2+k2**2)
                if dist < min_dist:
                    min_dist = dist
                    u1return = u1min
                    u2return = u2min
                dist = np.abs(k0+k1*u1min+k2*u2max)/np.sqrt(k1**2+k2**2)
                if dist < min_dist:
                    min_dist = dist
                    u1return = u1min
                    u2return = u2max
                dist = np.abs(k0+k1*u1max+k2*u2min)/np.sqrt(k1**2+k2**2)
                if dist < min_dist:
                    min_dist = dist
                    u1return = u1max
                    u2return = u2min
                dist = np.abs(k0+k1*u1max+k2*u2max)/np.sqrt(k1**2+k2**2)
                if dist < min_dist:
                    min_dist = dist
                    u1return = u1max
                    u2return = u2max
                return np.array([u1return, u2return])
            else:
                u2intersect = (k1**2*u2ref-k1*k2*u1ref-k0*k2)/((k1**2+k2**2))
                u1intersect = (-k2*u2intersect-k0)/k1
                if (u2intersect >= u2min) and (u2intersect <= u2max) and (u1intersect >= u1min) and (u1intersect <= u1max):
                    return np.array([u1intersect, u2intersect])
                else:
                    for i in range(len(u1_candidate)):
                        dist = np.abs(k0+k1*u1_candidate[i]+k2*u2_candidate[i])/np.sqrt(k1**2+k2**2)
                        if dist < min_dist:
                            min_dist = dist
                            u1return = u1_candidate[i]
                            u2return = u2_candidate[i]
                    return np.array([u1return, u2return])

        number_accessible = 1

        # intialize observable state
        sys_state = np.array([])
        sys_state = np.append(sys_state,self.state)
        sys_goal = np.array([])
        sys_goal = np.append(sys_goal,self.goal)

        # initialize f(x) + g(x)u for observable states
        dynamics_for_computation = np.array([])
        dynamics_for_computation = np.append(dynamics_for_computation,self.evolve_dynamics(self.state, [0,0]))

        # initialize dx for finite difference approximation
        dx_for_computation = np.array([])
        dx_for_computation = np.append(dx_for_computation,np.array([self.DX1,self.DX2,self.DX3,self.DX4]))

        # gather the above information from observable agents for computation
        for agents in agent_list:
            if agents.check_if_accessible(self.state, self.communication_range) and (agents.get_id() != self.id):
                sys_state = np.append(sys_state,agents.observe_state(self.state))
                sys_goal = np.append(sys_goal,agents.get_goal())
                dynamics_for_computation = np.append(dynamics_for_computation,agents.evolve_dynamics(agents.get_state(), agents.observe_control(self.state)))
                dx_for_computation = np.append(dx_for_computation,np.array([self.DX1,self.DX2,self.DX3,self.DX4]))
                number_accessible += 1
        sys_state = sys_state.astype(np.float32)
        sys_goal = sys_goal.astype(np.float32)

        # gpu variable initialization
        t0 = time.time()
        n_agent_cuda = cuda.mem_alloc(4)
        cuda.memcpy_htod(n_agent_cuda,np.array([number_accessible],dtype=np.float32))
        state_cuda = cuda.mem_alloc(sys_state.nbytes)
        goal_cuda = cuda.mem_alloc(sys_goal.nbytes)
        cuda.memcpy_htod(state_cuda, sys_state)
        cuda.memcpy_htod(goal_cuda, sys_goal)
        all_state_cuda = cuda.mem_alloc(number_accessible*2*self.K*self.T*number_accessible*4*2*4)
        
        collision = cuda.mem_alloc(self.K*self.T*number_accessible*4*2*4)
        safety_cuda = cuda.mem_alloc(self.K*number_accessible*4*2*4)

        all_state_cuda_no_gradient = cuda.mem_alloc(number_accessible*2*self.K*self.T*4)
        
        collision_no_gradient = cuda.mem_alloc(number_accessible*self.K*self.T*4)
        collision_n_no_gradient = cuda.mem_alloc(self.K*self.T*4)
        safety_no_gradient = cuda.mem_alloc(self.K*4)
        t1 = time.time()

        # pre-generate noise
        t0 = time.time()
        noise = rng.gen_normal((number_accessible*2*self.K*self.T*number_accessible*4*2*4,1), dtype=np.float32)
        noise_no_gradient = rng.gen_normal((number_accessible*2*self.K*self.T*4,1), dtype=np.float32)
        t1 = time.time()

        temp_state_cuda = cuda.mem_alloc(number_accessible*2*4)


        # gpu computation
        t0 = time.time()

        # this block computes APsi
        cuda.Context.synchronize()
        self.dynamic_evolution(state_cuda, goal_cuda, noise, all_state_cuda, self.parameters_cuda, block=(self.K,1,1), grid=(2*number_accessible*4,number_accessible,1))
        cuda.Context.synchronize()
        self.identify_collision(all_state_cuda, collision, temp_state_cuda, self.parameters_cuda, block=(self.K,1,1), grid=(2*number_accessible*4,self.T,1))
        cuda.Context.synchronize()
        self.identify_safety(collision, safety_cuda, self.parameters_cuda, block=(self.K,1,1), grid=(2*number_accessible*4,1,1))
        cuda.Context.synchronize()

        # this block computes Psi
        self.dynamic_evolution_no_gradient(state_cuda, goal_cuda, noise_no_gradient, all_state_cuda_no_gradient, self.parameters_cuda, block=(self.K,1,1), grid=(1,number_accessible,1))
        cuda.Context.synchronize()
        self.identify_collision_no_gradient(all_state_cuda_no_gradient, collision_no_gradient, self.parameters_cuda, block=(self.K,1,1), grid=(number_accessible,self.T,1))
        cuda.Context.synchronize()
        self.identify_safety_along_n(n_agent_cuda, collision_no_gradient, collision_n_no_gradient, self.parameters_cuda, block=(self.K,1,1), grid=(self.T,1,1))
        cuda.Context.synchronize()
        self.identify_safety_along_T(collision_n_no_gradient, safety_no_gradient, self.parameters_cuda, block=(self.K,1,1), grid=(1,1,1))
        cuda.Context.synchronize()
        t1 = time.time()
        # print("Computation time")
        # print(t1-t0)

        temp_state = np.zeros((number_accessible*2), dtype=np.float32)
        cuda.memcpy_dtoh(temp_state, temp_state_cuda)

        # gpu data transfer
        safety = np.zeros((2*number_accessible*4,self.K), dtype=np.float32)
        cuda.memcpy_dtoh(safety, safety_cuda)
        safety_self = np.zeros((self.K,1), dtype=np.float32)
        cuda.memcpy_dtoh(safety_self, safety_no_gradient)
        t0 = time.time()
        state_cuda.free()
        goal_cuda.free()
        all_state_cuda.free()
        collision.free()
        safety_cuda.free()
        all_state_cuda_no_gradient.free()
        collision_no_gradient.free()
        collision_n_no_gradient.free()
        safety_no_gradient.free()
        t1 = time.time()

        # gpu data processing
        # in the current configuration, the gradient is estimated by collecting 8 data points, fit a line, and use the slope as the estimated gradient
        safety_1 = safety[:,0:self.K//4]
        safety_2 = safety[:,self.K//4:2*self.K//4]
        safety_3 = safety[:,2*self.K//4:3*self.K//4]
        safety_4 = safety[:,3*self.K//4:self.K]
        safety_mean_1 = np.mean(safety_1, axis=1)
        safety_mean_2 = np.mean(safety_2, axis=1)
        safety_mean_3 = np.mean(safety_3, axis=1)
        safety_mean_4 = np.mean(safety_4, axis=1)
        gradient_left_1 = safety_mean_1[0:number_accessible*4]
        gradient_right_1 = safety_mean_1[number_accessible*4:2*number_accessible*4]
        gradient_left_2 = safety_mean_2[0:number_accessible*4]
        gradient_right_2 = safety_mean_2[number_accessible*4:2*number_accessible*4]
        gradient_left_3 = safety_mean_3[0:number_accessible*4]
        gradient_right_3 = safety_mean_3[number_accessible*4:2*number_accessible*4]
        gradient_left_4 = safety_mean_4[0:number_accessible*4]
        gradient_right_4 = safety_mean_4[number_accessible*4:2*number_accessible*4]
        gradient = np.zeros((number_accessible*4), dtype=np.float32)
        for i in range(number_accessible*4):
            x = np.array([-4*dx_for_computation[i],-3*dx_for_computation[i],-2*dx_for_computation[i],-dx_for_computation[i],dx_for_computation[i],2*dx_for_computation[i],3*dx_for_computation[i],4*dx_for_computation[i]])
            y = np.array([gradient_left_4[i],gradient_left_3[i],gradient_left_2[i],gradient_left_1[i],gradient_right_1[i],gradient_right_2[i],gradient_right_3[i],gradient_right_4[i]])
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]
            gradient[i] = m
        gradient_left = gradient
        gradient_right = gradient
        safety_self_mean = np.mean(safety_self)
        # gradient_left = safety_mean[0:number_accessible*4]
        # gradient_right = safety_mean[number_accessible*4:2*number_accessible*4]
        # print("Safe probability left: ", gradient_left[0:4])
        # print("Safe probability right: ", gradient_right[0:4])
        # gradient = (gradient_right - gradient_left)/2/dx_for_computation
        # print("Safe probability: ", safety_self_mean)

        # construct safety certificate, as defined in equation (18)
        # this certificate can be constructed as the form of a linear function of the control action, k_0 + k_1*u1 + k_2*u2 >= 0
        coefficient_const = np.dot(gradient, dynamics_for_computation) + gamma(safety_self_mean-(1-self.epsilon))
        coefficient_u1 = gradient[2]
        coefficient_u2 = gradient[3]
        nominal_control = self.nominal_control()
        if coefficient_const + coefficient_u1*nominal_control[0] + coefficient_u2*nominal_control[1] >= 0:
            self.control = nominal_control
        else:
            self.control = find_feasible_control(coefficient_const, coefficient_u1, coefficient_u2, nominal_control[0], nominal_control[1], self.u1min, self.u1max, self.u2min, self.u2max)
        self.control_available = True       
        self.control_trajectory = np.vstack((self.control_trajectory, self.control))

        # evolve dynamics
        self.state = self.state + self.evolve_dynamics(self.state, self.control)*self.dt
        # inject noise in speed and heading angle
        self.state[2] = self.state[2] + np.random.normal(0,self.sigma[0]*self.dt,1)
        self.state[3] = self.state[3] + np.random.normal(0,self.sigma[1]*self.dt,1)

        # put heading angle in [-pi, pi], if needed
        if self.state[2] < 0:
            self.state[2] = -self.state[2]
            self.state[3] = self.state[3] - np.pi
        while self.state[3] > np.pi:
            self.state[3] = self.state[3] - 2*np.pi
        while self.state[3] < -np.pi:
            self.state[3] = self.state[3] + 2*np.pi
        # record information
        self.trajectory = np.vstack((self.trajectory, self.state))
        self.safe_probability_trajectory = np.vstack((self.safe_probability_trajectory, safety_self_mean))
        # both left and right are the gradient itself
        # please ignore these 2 lines
        self.safe_probability_left_trajectory = np.vstack((self.safe_probability_left_trajectory, gradient_left[0:4]))
        self.safe_probability_right_trajectory = np.vstack((self.safe_probability_right_trajectory, gradient_right[0:4]))
        # if goal is reached, turn agent off
        if np.linalg.norm(self.state[0:2]-self.goal) < self.goal_range:
            self.on = False

    