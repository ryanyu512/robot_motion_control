import copy as copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

from ref_gen import *
from robot import *
from MPC import *

def sim(sim_params, 
        robot_params,
        MPC_params): 

    if not type(sim_params['y_target']) is list:
        print("== error == please ensure the y_target type is LIST")
        return 
    
    #initialise simulation setting
    end_t = sim_params['end_t']
    dt    = sim_params['dt']
    y_target = sim_params['y_target']
    sim_steps = int(end_t/dt)

    is_animate  = sim_params['is_animate']
    is_save_gif = sim_params['is_save_gif']

    #initialise control input
    u_control = np.array([0., 0.]) #steering angle of the front wheel
    u_control.shape = (2, 1)
    u_hist = [u_control]

    #initialise robot 
    robot = Robot()

    #initialise delta control input
    N_in = 2
    du = np.zeros((N_in*MPC_params['h_windows'], 1))

    #initialise trajectory generator
    ref_gen = Ref_Gen()

    x_dot_body_ref, y_dot_body_ref, X_ref, Y_ref, psi_ref = ref_gen.get_cubic_path(dt, 0, end_t, 
                                                                                   s_x = 0.,  s_y = 0.,
                                                                                   e_x = 10., e_y = 10.,
                                                                                   s_vx = 1., s_vy = 1.,
                                                                                   e_vx = 1., e_vy = 1.)
    # plt.plot(X_ref, Y_ref, '-')
    # plt.show()

    # plt.plot(x_dot_body_ref)
    # plt.show()

    # plt.plot(y_dot_body_ref)
    # plt.show()

    # plt.plot(psi_ref)
    # plt.show()

    #initialise robot parameters
    N_out = robot_params['N_out']
    init_x_dot = x_dot_body_ref[0]   #longitudinal velocity relative to body frame
    init_y_dot = y_dot_body_ref[0]   #lateral velocity relative to body frame
    init_psi   = psi_ref[0]          #angle at the MoC relative to global x
    init_psi_dot = robot_params['init_psi_dot']   #angular velocity at the MoC relative to global x
    
    if sim_params['random_init_Y']:
        init_Y = np.random.uniform(low = y_target[0] - 10., high = y_target[0] + 10.)  #Y position in global frame
    else:
        init_Y = robot_params['init_Y']
    init_X = robot_params['init_X']
    steering_ang_lim = robot_params['steering_ang_lim']

    #initialise current state
    c_state = np.array([init_x_dot, init_y_dot, init_psi, init_psi_dot, init_X, init_Y])
    c_state.shape = (len(c_state), 1)
    state_hist = [c_state]

    #initialise MPC controller
    mpc = MPC()
    mpc.Q = MPC_params['Q']
    mpc.S = MPC_params['S']
    mpc.R = MPC_params['R']
    mpc.h_windows = MPC_params['h_windows']

    mpc.get_state_space(c_state,
                        u_control[0, 0], 
                        u_control[1, 0],
                        dt)

    for i in range(sim_steps + 1):
        c_aug_state = np.concatenate((copy.copy(c_state), u_control), axis = 0)
        
        #obtain reference signal windows
        aug_r = np.zeros((N_out*mpc.h_windows, 1)) 

        ref_cnt = i
        for j in range(0, aug_r.shape[0], N_out):
            if ref_cnt < len(psi_ref):
                aug_r[j    , 0] = x_dot_body_ref[ref_cnt]
                aug_r[j + 1, 0] = psi_ref[ref_cnt]
                aug_r[j + 2, 0] = X_ref[ref_cnt]
                aug_r[j + 3, 0] = Y_ref[ref_cnt] 
                ref_cnt += 1
            else:
                aug_r[j    , 0] = x_dot_body_ref[-1]
                aug_r[j + 1, 0] = psi_ref[-1]
                aug_r[j + 2, 0] = X_ref[-1]
                aug_r[j + 3, 0] = Y_ref[-1] 

        H, Ft, C_AB, A_pow, g, h = mpc.compute_aug_constraints(c_aug_state, du, dt)
        du = mpc.compute_control_input(H, Ft, g, h, c_aug_state, aug_r)
        break
    #     # u_control += du

    #     # if u_control[0,0] < -steering_ang_lim:
    #     #     u_control[0,0] = -steering_ang_lim
    #     # elif u_control[0,0] > steering_ang_lim:
    #     #     u_control[0,0] =  steering_ang_lim

    #     #update current states
    #     c_state = robot.update(c_state, u_control[0,0], dt)
    #     c_state.shape = (len(c_state), 1)

    #     #stort history
    #     u_hist.append(u_control[0, 0])
    #     state_hist.append(copy.copy(c_state))

    # fig = plt.figure()
    # ax  = plt.subplot(1, 1, 1)

    # def init_func():
    #     ax.clear()
    #     plt.xlabel('x position (m)')
    #     plt.ylabel('y position (m)')

    # def update_plot(i):
    #     #clear plot
    #     ax.clear()

    #     psi = state_hist[i][1, 0]
    #     y   = state_hist[i][3, 0]
    #     steer_ang = u_hist[i]

    #     #plot 
    #     ax.plot(X_ref, Y_ref, '-r')

    #     #==== plot wheels ====#

    #     R = np.array([[np.cos(psi), -np.sin(psi)], 
    #                 [np.sin(psi),  np.cos(psi)]])

    #     #plot heading
    #     ax.plot([X_ref[i], 
    #              X_ref[i] + robot.Lf*np.cos(psi) * 2], 
    #             [y, 
    #             y + robot.Lf*np.sin(psi) * 2], 
    #             '-g')

    #     #plot body left line
    #     L_pt1 = np.matmul(R, np.array([[-robot.Lr], [-2.]]))
    #     L_pt2 = np.matmul(R, np.array([[ robot.Lf], [-2.]]))
    #     ax.plot([X_ref[i] + L_pt1[0,0], 
    #              X_ref[i] + L_pt2[0,0]], 
    #             [y + L_pt1[1,0], 
    #              y + L_pt2[1,0]], 
    #             '-g')

    #     #plot body right line
    #     R_pt1 = np.matmul(R, np.array([[-robot.Lr], [2.]]))
    #     R_pt2 = np.matmul(R, np.array([[ robot.Lf], [2.]]))
    #     ax.plot([X_ref[i] + R_pt1[0,0], 
    #              X_ref[i] + R_pt2[0,0]], 
    #             [y + R_pt1[1,0], 
    #              y + R_pt2[1,0]], 
    #             '-g')
        
    #     #plot body front line
    #     ax.plot([X_ref[i] + R_pt2[0,0], 
    #              X_ref[i] + L_pt2[0,0]], 
    #             [y + R_pt2[1,0], 
    #             y + L_pt2[1,0]], 
    #             '-g')

    #     #plot body back line
    #     ax.plot([X_ref[i] + R_pt1[0,0], 
    #              X_ref[i] + L_pt1[0,0]], 
    #             [y + R_pt1[1,0], 
    #             y + L_pt1[1,0]], 
    #             '-g')

    #     #==== plot wheels ====#
    #     Rw = np.array([[np.cos(psi + steer_ang), -np.sin(psi + steer_ang)], 
    #                    [np.sin(psi + steer_ang),  np.cos(psi + steer_ang)]])
        
    #     #plot front left wheel
    #     Lfw_pt1 = L_pt2 + np.matmul(Rw, np.array([[-robot.R_wheel], [0.]]))
    #     Lfw_pt2 = L_pt2 + np.matmul(Rw, np.array([[ robot.R_wheel], [0.]]))
    #     ax.plot([X_ref[i] + Lfw_pt1[0, 0], 
    #              X_ref[i] + Lfw_pt2[0, 0]], 
    #             [y + Lfw_pt1[1, 0], 
    #             y + Lfw_pt2[1, 0]], 
    #             '-b', linewidth = '3')
        
    #     #plot front right wheel
    #     Rfw_pt1 = R_pt2 + np.matmul(Rw, np.array([[-robot.R_wheel], [0.]]))
    #     Rfw_pt2 = R_pt2 + np.matmul(Rw, np.array([[ robot.R_wheel], [0.]]))
    #     ax.plot([X_ref[i] + Rfw_pt1[0, 0], 
    #              X_ref[i] + Rfw_pt2[0, 0]], 
    #             [y + Rfw_pt1[1, 0], 
    #             y + Rfw_pt2[1, 0]], 
    #             '-b', linewidth = '3')

    #     #plot back left wheel
    #     Lrw_pt1 = L_pt1 + np.matmul(R, np.array([[-robot.R_wheel], [0.]]))
    #     Lrw_pt2 = L_pt1 + np.matmul(R, np.array([[ robot.R_wheel], [0.]]))
    #     ax.plot([X_ref[i] + Lrw_pt1[0, 0], 
    #              X_ref[i] + Lrw_pt2[0, 0]], 
    #             [y + Lrw_pt1[1, 0], 
    #             y + Lrw_pt2[1, 0]], 
    #             '-b', linewidth = '3')
        
    #     #plot back right wheel
    #     Rrw_pt1 = R_pt1 + np.matmul(R, np.array([[-robot.R_wheel], [0.]]))
    #     Rrw_pt2 = R_pt1 + np.matmul(R, np.array([[ robot.R_wheel], [0.]]))
    #     ax.plot([X_ref[i] + Rrw_pt1[0, 0], 
    #              X_ref[i] + Rrw_pt2[0, 0]], 
    #             [y + Rrw_pt1[1, 0], 
    #             y + Rrw_pt2[1, 0]], 
    #             '-b', linewidth = '3')

    #     ax.set_xlim([min(X_ref), max(X_ref)])
    #     ax.set_ylim([min(init_Y , min(y_target) - 10),  max(init_Y, max(y_target) + 10)])
    #     ax.set_aspect('equal', adjustable='box')

    #     plt.xlabel('x position (m)')
    #     plt.ylabel('y position (m)')

    # anim = FuncAnimation(fig, 
    #             update_plot,
    #             frames = np.arange(0, sim_steps), 
    #             init_func = init_func,
    #             interval = 1,
    #             repeat = False)

    # if is_animate:
    #     plt.show()

    # if is_save_gif:
    #     writer = animation.PillowWriter(fps=15,
    #                                     metadata=dict(artist='Me'),
    #                                     bitrate=1800)

    #     anim.save('demo.gif', writer=writer)