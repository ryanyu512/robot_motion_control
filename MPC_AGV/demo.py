import numpy as np
from sim import *

sim_params = {'end_t': 154,
              'dt': 0.02,
              'random_init_Y': True,
              'is_animate': True,
              'is_save_gif': False}

robot_params = {'init_x_dot': 0.1,
                'init_y_dot': 0.,
                'init_psi': 0.,
                'init_psi_dot': 0.,
                'init_X': 0,
                'init_Y': 10,
                'steering_ang_lim': np.math.pi/6.
                }

MPC_params = {'Q': np.array([[100,    0,  0,  0], 
                             [0,  20000,  0,  0],
                             [0,    0, 1000,  0],
                             [0,    0,  0, 1000]]),
              'S': np.array([[100,    0,  0,  0], 
                             [0,  20000,  0,  0],
                             [0,    0, 1000,  0],
                             [0,    0,  0, 1000]]),
              'R': np.array([[100., 0], [0, 1.]]),
              'h_windows': 10}

ref_gen_params = {
              'X_waypts': np.array([ 0,60,110,140,160,110, 40,10,40,70,110,150]),
              'Y_waypts': np.array([40,20, 20, 60,100,140,140,80,60,60, 90, 90]),
              'X_dot_waypts': np.array([2,1,1,1,0,-1,-1, 0,1,1,1,1])*3,
              'Y_dot_waypts': np.array([0,0,0,1,1, 0, 0,-1,0,0,0,0])*3,
}

sim(sim_params, 
    robot_params,
    MPC_params, 
    ref_gen_params)