import numpy as np
from sim import *

sim_params = {'end_t': 8.,
              'dt': 0.02,
              'y_target': [-5, 0, 5, -5],
              'random_init_Y': True,
              'is_animate': True,
              'is_save_gif': False}

robot_params = {'N_out': 4,  
                'init_x_dot': 0.1,
                'init_y_dot': 0.,
                'init_psi': 0.,
                'init_psi_dot': 0.,
                'init_X': 0,
                'init_Y': 10,
                'steering_ang_lim': np.math.pi/6.
                }

MPC_params = {'Q': np.array([[1,    0,  0,  0], 
                             [0,  200,  0,  0],
                             [0,    0, 50,  0],
                             [0,    0,  0, 50]]),
              'S': np.array([[1,    0,  0,  0], 
                             [0,  200,  0,  0],
                             [0,    0, 50,  0],
                             [0,    0,  0, 50]]),
              'R': np.array([[100., 0], [0, 1.]]),
              'h_windows': 10}

sim(sim_params, 
    robot_params,
    MPC_params)