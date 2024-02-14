import numpy as np
from sim import *

sim_params = {'end_t': 120,
              'dt': 0.02,
              'random_init_Y': True,
              'is_animate': True,
              'is_save_gif': True}

robot_params = {}

MPC_params = {'Q': np.array([[100,    0,  0,  0], 
                             [0,  40000,  0,  0],
                             [0,    0, 1000,  0],
                             [0,    0,  0, 1000]]),
              'S': np.array([[100,    0,  0,  0], 
                             [0,  40000,  0,  0],
                             [0,    0, 1000,  0],
                             [0,    0,  0, 1000]]),
              'R': np.array([[100., 0], [0, 1.]]),
              'h_windows': 10, 
              'du1_limit': np.pi/300., #delta steering angle control input
              'du2_limit': 0.1,  #delta applied acceleration
              'x_dot_min': 1.,   #min longitudinal velocity relative to body frame
              'x_dot_max': 30.,  #max longitudinal velocity relative to body frame
              'y_dot_min': -3.,  #min lateral velocity relative to body frame
              'y_dot_max':  3.,  #max lateral velocity relative to body frame
              'x_dot_dot_min': -1., #min applied longitudinal acc relative to body frame
              'x_dot_dot_max':  4., #max applied longitudinal acc relative to body frame
              'delta_limit': np.pi/6., #steering angle limit => respect the linearization assumption
              } 

ref_gen_params = {
              'X_waypts': np.array([ 0, 60, 110, 140, 160, 110,  40, -20,  0]), #relative to global frame
              'Y_waypts': np.array([40, 20,  20,  60, 100, 140, 140,  80, 40]), #relative to global frame
              'X_dot_waypts': np.array([2,1,1,1,0,-1,-1, 0,1])*3, #relative to global frame
              'Y_dot_waypts': np.array([0,0,0,1,1, 0, 0,-1,0])*3, #relative to global frame
}

sim(sim_params, 
    robot_params,
    MPC_params, 
    ref_gen_params)