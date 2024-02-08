import copy as copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

from ref_gen import *
from robot import *
from MPC import *


#initialise simulation setting
end_t = 10.
dt    = 0.02
y_target = -9
sim_steps = int(end_t/dt)

is_animate  = True
is_save_gif = False

#initialise number of outputs
N_out = 2

#initialise control input
u_control = np.array([0.]) #steering angle of the front wheel
u_control.shape = (1, 1)
u_hist = [u_control[0,0]]

#initialise state
init_y_dot = 0.     #lateral velocity relative to body frame
init_psi   = 0.     #angle at the MoC relative to global x
init_psi_dot = 0.   #angular velocity at the MoC relative to global x
init_Y       = y_target + 10.  #Y position in global frame

c_state = np.array([init_y_dot, init_psi, init_psi_dot, init_Y])
c_state.shape = (len(c_state), 1)
state_hist = [c_state]

#initialise robot 
robot = Robot()

#initialise MPC controller
mpc = MPC(dt)

#initialise trajectory generator
ref_gen = Ref_Gen()
x_ref, y_ref, psi_ref = ref_gen.generate_ref_signal(end_t, dt, y_target)

mpc.compute_aug_matrix()
# mpc.mpc_simplification()
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

    if u_control[0,0] < -np.pi/6:
        u_control[0,0] = -np.pi/6
    elif u_control[0,0] > np.pi/6:
        u_control[0,0] =  np.pi/6

    #update current states
    c_state = robot.update(c_state, u_control[0,0], dt)
    c_state.shape = (len(c_state), 1)

    #stort history
    u_hist.append(u_control[0, 0])
    state_hist.append(copy.copy(c_state))


plt.plot(u_hist)
plt.show()

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

    #plot true position
    ax.plot(x_ref[i], y, 'og')

    #plot 
    ax.plot(x_ref, y_ref, '-r')

    #plot heading
    ax.plot([x_ref[i] - robot.Lr*np.cos(psi), 
             x_ref[i] + robot.Lf*np.cos(psi)], 
            [y - robot.Lr*np.sin(psi), 
             y + robot.Lf*np.sin(psi)], 
            '-g')
    
    #plot front wheel


    ax.set_xlim([min(x_ref), max(x_ref)])
    ax.set_ylim([min(init_Y , y_target - 10),  max(init_Y, y_target + 10)])
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