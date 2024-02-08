import copy as copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

from ref_gen import *
from robot import *
from MPC import *

def sim(sim_params, 
        robot_params): 

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

    #initialise robot parameters
    N_out = robot_params['N_out']
    init_y_dot = robot_params['init_y_dot']   #lateral velocity relative to body frame
    init_psi   = robot_params['init_psi']     #angle at the MoC relative to global x
    init_psi_dot = robot_params['init_psi_dot']   #angular velocity at the MoC relative to global x
    
    if sim_params['random_init_Y']:
        init_Y = np.random.uniform(low = y_target[0] - 10., high = y_target[0] + 10.)  #Y position in global frame
    else:
        init_Y = robot_params['init_Y']
    steering_ang_lim = robot_params['steering_ang_lim']

    #initialise control input
    u_control = np.array([0.]) #steering angle of the front wheel
    u_control.shape = (1, 1)
    u_hist = [u_control[0,0]]

    #initialise current state
    c_state = np.array([init_y_dot, init_psi, init_psi_dot, init_Y])
    c_state.shape = (len(c_state), 1)
    state_hist = [c_state]

    #initialise robot 
    robot = Robot()

    #initialise MPC controller
    mpc = MPC(dt)

    #initialise trajectory generator
    ref_gen = Ref_Gen()

    x_ref, y_ref, psi_ref = ref_gen.generate_ref_signal(0, end_t, dt, y_target)

    mpc.compute_aug_matrix()
    for i in range(sim_steps + 1):
        c_aug_state = np.concatenate((copy.copy(c_state), u_control), axis = 0)
        
        #obtain reference signal windows
        aug_r = np.zeros((N_out*mpc.h_windows, 1)) 

        ref_cnt = i
        for j in range(0, aug_r.shape[0], N_out):
            if ref_cnt < len(psi_ref):
                aug_r[j, 0] = psi_ref[ref_cnt]
                aug_r[j + 1, 0] = y_ref[ref_cnt] 
                ref_cnt += 1
            else:
                aug_r[j, 0] = psi_ref[-1]
                aug_r[j + 1, 0] = y_ref[-1] 

        du = mpc.compute_control_input(c_aug_state, aug_r)

        u_control += du

        if u_control[0,0] < -steering_ang_lim:
            u_control[0,0] = -steering_ang_lim
        elif u_control[0,0] > steering_ang_lim:
            u_control[0,0] =  steering_ang_lim

        #update current states
        c_state = robot.update(c_state, u_control[0,0], dt)
        c_state.shape = (len(c_state), 1)

        #stort history
        u_hist.append(u_control[0, 0])
        state_hist.append(copy.copy(c_state))

    fig = plt.figure()
    ax  = plt.subplot(1, 1, 1)

    def init_func():
        ax.clear()
        plt.xlabel('x position (m)')
        plt.ylabel('y position (m)')

    def update_plot(i):
        #clear plot
        ax.clear()

        psi = state_hist[i][1, 0]
        y   = state_hist[i][3, 0]
        steer_ang = u_hist[i]

        #plot 
        ax.plot(x_ref, y_ref, '-r')

        #==== plot wheels ====#

        R = np.array([[np.cos(psi), -np.sin(psi)], 
                    [np.sin(psi),  np.cos(psi)]])

        #plot heading
        ax.plot([x_ref[i], 
                x_ref[i] + robot.Lf*np.cos(psi) * 2], 
                [y, 
                y + robot.Lf*np.sin(psi) * 2], 
                '-g')

        #plot body left line
        L_pt1 = np.matmul(R, np.array([[-robot.Lr], [-2.]]))
        L_pt2 = np.matmul(R, np.array([[ robot.Lf], [-2.]]))
        ax.plot([x_ref[i] + L_pt1[0,0], 
                x_ref[i] + L_pt2[0,0]], 
                [y + L_pt1[1,0], 
                y + L_pt2[1,0]], 
                '-g')

        #plot body right line
        R_pt1 = np.matmul(R, np.array([[-robot.Lr], [2.]]))
        R_pt2 = np.matmul(R, np.array([[ robot.Lf], [2.]]))
        ax.plot([x_ref[i] + R_pt1[0,0], 
                x_ref[i] + R_pt2[0,0]], 
                [y + R_pt1[1,0], 
                y + R_pt2[1,0]], 
                '-g')
        
        #plot body front line
        ax.plot([x_ref[i] + R_pt2[0,0], 
                x_ref[i] + L_pt2[0,0]], 
                [y + R_pt2[1,0], 
                y + L_pt2[1,0]], 
                '-g')

        #plot body back line
        ax.plot([x_ref[i] + R_pt1[0,0], 
                x_ref[i] + L_pt1[0,0]], 
                [y + R_pt1[1,0], 
                y + L_pt1[1,0]], 
                '-g')

        #==== plot wheels ====#
        Rw = np.array([[np.cos(psi + steer_ang), -np.sin(psi + steer_ang)], 
                    [np.sin(psi + steer_ang),  np.cos(psi + steer_ang)]])
        
        #plot front left wheel
        Lfw_pt1 = L_pt2 + np.matmul(Rw, np.array([[-robot.R_wheel], [0.]]))
        Lfw_pt2 = L_pt2 + np.matmul(Rw, np.array([[ robot.R_wheel], [0.]]))
        ax.plot([x_ref[i] + Lfw_pt1[0, 0], 
                x_ref[i] + Lfw_pt2[0, 0]], 
                [y + Lfw_pt1[1, 0], 
                y + Lfw_pt2[1, 0]], 
                '-b', linewidth = '3')
        
        #plot front right wheel
        Rfw_pt1 = R_pt2 + np.matmul(Rw, np.array([[-robot.R_wheel], [0.]]))
        Rfw_pt2 = R_pt2 + np.matmul(Rw, np.array([[ robot.R_wheel], [0.]]))
        ax.plot([x_ref[i] + Rfw_pt1[0, 0], 
                x_ref[i] + Rfw_pt2[0, 0]], 
                [y + Rfw_pt1[1, 0], 
                y + Rfw_pt2[1, 0]], 
                '-b', linewidth = '3')

        #plot back left wheel
        Lrw_pt1 = L_pt1 + np.matmul(R, np.array([[-robot.R_wheel], [0.]]))
        Lrw_pt2 = L_pt1 + np.matmul(R, np.array([[ robot.R_wheel], [0.]]))
        ax.plot([x_ref[i] + Lrw_pt1[0, 0], 
                x_ref[i] + Lrw_pt2[0, 0]], 
                [y + Lrw_pt1[1, 0], 
                y + Lrw_pt2[1, 0]], 
                '-b', linewidth = '3')
        
        #plot back right wheel
        Rrw_pt1 = R_pt1 + np.matmul(R, np.array([[-robot.R_wheel], [0.]]))
        Rrw_pt2 = R_pt1 + np.matmul(R, np.array([[ robot.R_wheel], [0.]]))
        ax.plot([x_ref[i] + Rrw_pt1[0, 0], 
                x_ref[i] + Rrw_pt2[0, 0]], 
                [y + Rrw_pt1[1, 0], 
                y + Rrw_pt2[1, 0]], 
                '-b', linewidth = '3')

        ax.set_xlim([min(x_ref), max(x_ref)])
        ax.set_ylim([min(init_Y , min(y_target) - 10),  max(init_Y, max(y_target) + 10)])
        ax.set_aspect('equal', adjustable='box')

        plt.xlabel('x position (m)')
        plt.ylabel('y position (m)')

    anim = FuncAnimation(fig, 
                update_plot,
                frames = np.arange(0, sim_steps), 
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