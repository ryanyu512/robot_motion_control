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
        MPC_params,
        ref_gen_params): 
    
    #initialise simulation setting
    end_t = sim_params['end_t']
    dt    = sim_params['dt']
    sim_steps = int(end_t/dt)

    is_animate  = sim_params['is_animate']
    is_save_gif = sim_params['is_save_gif']

    #initialise control input
    u_control = np.array([[0.], [0.]]) #steering angle of the front wheel
    u_hist = [u_control]

    #initialise robot 
    robot = Robot()

    #initialise MPC controller
    mpc = MPC()
    mpc.Q = MPC_params['Q']
    mpc.S = MPC_params['S']
    mpc.R = MPC_params['R']
    mpc.h_windows = MPC_params['h_windows']
    mpc.du1_limit = MPC_params['du1_limit']
    mpc.du2_limit = MPC_params['du2_limit']
    mpc.delta_limit = MPC_params['delta_limit']
    mpc.y_dot_min = MPC_params['y_dot_min']
    mpc.y_dot_max = MPC_params['y_dot_max']
    mpc.x_dot_min = MPC_params['x_dot_min']
    mpc.x_dot_max = MPC_params['x_dot_max']
    mpc.x_dot_dot_min = MPC_params['x_dot_dot_min']
    mpc.x_dot_dot_max = MPC_params['x_dot_dot_max']

    #initialise delta control input
    N_in = mpc.N_in
    du = np.zeros((N_in*MPC_params['h_windows'], 1))

    #initialise trajectory generator
    ref_gen = Ref_Gen()
    ref_gen.X_waypts = ref_gen_params['X_waypts']
    ref_gen.Y_waypts = ref_gen_params['Y_waypts']
    ref_gen.X_dot_waypts = ref_gen_params['X_dot_waypts']
    ref_gen.Y_dot_waypts = ref_gen_params['Y_dot_waypts']
    x_dot_body_ref, y_dot_body_ref, psi_ref, X_ref, Y_ref = ref_gen.get_cubic_path(0, end_t, dt)

    #initialise robot parameters
    init_x_dot = x_dot_body_ref[0]   #longitudinal velocity relative to body frame
    init_y_dot = y_dot_body_ref[0]   #lateral velocity relative to body frame
    init_psi   = psi_ref[0]          #angle at the MoC relative to global x
    init_psi_dot = 0.                #angular velocity at the MoC relative to global x
    init_X     = X_ref[0]            #initial x position relative to global frame
    init_Y     = Y_ref[0]            #initial y position relative to global frame

    #initialise current state
    c_state = np.array([[init_x_dot], 
                        [init_y_dot], 
                        [init_psi], 
                        [init_psi_dot], 
                        [init_X],
                        [init_Y]])
    
    state_hist = [c_state]

    x_dot_dot_hist = []

    for i in range(sim_steps + 1):
        #augment state
        c_aug_state = np.concatenate((copy.copy(c_state), u_control), axis = 0)
        
        #obtain reference signal windows
        aug_r = []

        ref_cnt = i
        for j in range(0, mpc.h_windows):
            if ref_cnt < len(psi_ref):
                aug_r.append(x_dot_body_ref[ref_cnt])
                aug_r.append(psi_ref[ref_cnt])
                aug_r.append(X_ref[ref_cnt])
                aug_r.append(Y_ref[ref_cnt]) 
                ref_cnt += 1
            else:
                mpc.h_windows -= 1
                break

        aug_r = np.array(aug_r)
        aug_r.shape = (len(aug_r), 1)

        mpc.get_state_space(c_state, u_control[0, 0], u_control[1, 0], dt)
        H, Ft, C_AB, A_pow, g, h = mpc.compute_aug_constraints(c_aug_state, du, dt)
        du = mpc.compute_control_input(H, Ft, g, h, c_aug_state, aug_r)
        
        u_control[0, 0] += du[0, 0]
        u_control[1, 0] += du[1, 0]

        #update current states
        c_state, x_dot_dot, y_dot_dot, psi_dot_dot = robot.update(c_state, u_control[0,0], u_control[1,0], dt)
        c_state.shape = (len(c_state), 1)

        #stort history
        u_hist.append(u_control)
        state_hist.append(copy.copy(c_state))
        x_dot_dot_hist.append(x_dot_dot)

    fig = plt.figure()
    ax  = plt.subplot(1, 1, 1)

    def init_func():
        ax.clear()
        plt.xlabel('x position (m)')
        plt.ylabel('y position (m)')

    def update_plot(i):
        #clear plot
        ax.clear()

        # aug_state = [x_dot, y_dot, psi, psi_dot, X, Y, delta, acc_x]
        psi = state_hist[i][2, 0]
        x   = state_hist[i][4, 0]
        y   = state_hist[i][5, 0]
        steer_ang = u_hist[i][0, 0]

        #plot 
        ax.plot(X_ref, Y_ref, '-r')

        #plot robot history trajectory
        robot_trajectory_hist = np.array([[sh[4, 0], sh[5, 0]] for sh in state_hist[0:i+1]])
        ax.plot(robot_trajectory_hist[:, 0], 
                robot_trajectory_hist[:, 1], '--g')
        #==== plot wheels ====#

        R = np.array([[np.cos(psi), -np.sin(psi)], 
                      [np.sin(psi),  np.cos(psi)]])

        #plot heading
        ax.plot([x, 
                 x + robot.Lf*np.cos(psi) * 2], 
                [y, 
                 y + robot.Lf*np.sin(psi) * 2], 
                '-g')

        #plot body left line
        L_pt1 = np.matmul(R, np.array([[-robot.Lr], [-2.]]))
        L_pt2 = np.matmul(R, np.array([[ robot.Lf], [-2.]]))
        ax.plot([x + L_pt1[0,0], 
                 x + L_pt2[0,0]], 
                [y + L_pt1[1,0], 
                 y + L_pt2[1,0]], 
                '-g')

        #plot body right line
        R_pt1 = np.matmul(R, np.array([[-robot.Lr], [2.]]))
        R_pt2 = np.matmul(R, np.array([[ robot.Lf], [2.]]))
        ax.plot([x + R_pt1[0,0], 
                 x + R_pt2[0,0]], 
                [y + R_pt1[1,0], 
                 y + R_pt2[1,0]], 
                '-g')
        
        #plot body front line
        ax.plot([x + R_pt2[0,0], 
                 x + L_pt2[0,0]], 
                [y + R_pt2[1,0], 
                 y + L_pt2[1,0]], 
                '-g')

        #plot body back line
        ax.plot([x + R_pt1[0,0], 
                 x + L_pt1[0,0]], 
                [y + R_pt1[1,0], 
                 y + L_pt1[1,0]], 
                '-g')

        #==== plot wheels ====#
        Rw = np.array([[np.cos(psi + steer_ang), -np.sin(psi + steer_ang)], 
                       [np.sin(psi + steer_ang),  np.cos(psi + steer_ang)]])
        
        #plot front left wheel
        Lfw_pt1 = L_pt2 + np.matmul(Rw, np.array([[-robot.R_wheel], [0.]]))
        Lfw_pt2 = L_pt2 + np.matmul(Rw, np.array([[ robot.R_wheel], [0.]]))
        ax.plot([x + Lfw_pt1[0, 0], 
                 x + Lfw_pt2[0, 0]], 
                [y + Lfw_pt1[1, 0], 
                 y + Lfw_pt2[1, 0]], 
                '-b', linewidth = '3')
        
        #plot front right wheel
        Rfw_pt1 = R_pt2 + np.matmul(Rw, np.array([[-robot.R_wheel], [0.]]))
        Rfw_pt2 = R_pt2 + np.matmul(Rw, np.array([[ robot.R_wheel], [0.]]))
        ax.plot([x + Rfw_pt1[0, 0], 
                 x + Rfw_pt2[0, 0]], 
                [y + Rfw_pt1[1, 0], 
                y + Rfw_pt2[1, 0]], 
                '-b', linewidth = '3')

        #plot back left wheel
        Lrw_pt1 = L_pt1 + np.matmul(R, np.array([[-robot.R_wheel], [0.]]))
        Lrw_pt2 = L_pt1 + np.matmul(R, np.array([[ robot.R_wheel], [0.]]))
        ax.plot([x + Lrw_pt1[0, 0], 
                 x + Lrw_pt2[0, 0]], 
                [y + Lrw_pt1[1, 0], 
                 y + Lrw_pt2[1, 0]], 
                '-b', linewidth = '3')
        
        #plot back right wheel
        Rrw_pt1 = R_pt1 + np.matmul(R, np.array([[-robot.R_wheel], [0.]]))
        Rrw_pt2 = R_pt1 + np.matmul(R, np.array([[ robot.R_wheel], [0.]]))
        ax.plot([x + Rrw_pt1[0, 0], 
                 x + Rrw_pt2[0, 0]], 
                [y + Rrw_pt1[1, 0], 
                 y + Rrw_pt2[1, 0]], 
                '-b', linewidth = '3')

        ax.set_xlim([min(X_ref - 20), max(X_ref + 20)])
        ax.set_ylim([min(Y_ref - 20), max(Y_ref + 20)])
        ax.set_aspect('equal', adjustable='box')

        plt.xlabel('x position (m)')
        plt.ylabel('y position (m)')

    anim = FuncAnimation(fig, 
                update_plot,
                frames = np.arange(0, sim_steps, 5), 
                init_func = init_func,
                interval = 1,
                repeat = False)

    if is_animate:
        plt.show()

    if is_save_gif:
        writer = animation.PillowWriter(fps=15,
                                        metadata=dict(artist='Me'),
                                        bitrate=1800)

        anim.save('demo.gif', writer=writer)