import numpy as np
from robot_base import *

class Ref_Gen(Robot_Base):
    def __init__(self):
        Robot_Base.__init__(self)



    def get_cubic_coeff(self, st, s_pos, s_vel, et, e_pos, e_vel):

        # a0 + a1*st + a2*st**2 + a3*st**3   = s_pos
        #  0 + a1    + 2*a2*st  + 3*a3*st**2 = s_vel
        # a0 + a1*et + a2*et**2 + a3*et**3   = e_pos
        #  0 + a1    + 2*a2*et  + 3*a3*et**2 = e_vel

        '''
            A*x = y
        '''

        A = np.array([[1, st, st**2,   st**3], 
                      [0,  1,  2*st, 3*st**2],
                      [1, et, et**2,   et**3], 
                      [0,  1,  2*et, 3*et**2]])
        
        y = np.array([[s_pos], [s_vel], [e_pos], [e_vel]])

        x = np.matmul(np.linalg.inv(A), y)

        return x

    def get_cubic_path(self, 
                       dt,
                       s_t, e_t, 
                       s_x,  s_y, 
                       e_x,  e_y,
                       s_vx, s_vy, 
                       e_vx, e_vy):

        sim_t = np.arange(s_t, e_t + dt, dt)
        cx = self.get_cubic_coeff(s_t, s_x, s_vx, e_t, e_x, e_vx)
        cy = self.get_cubic_coeff(s_t, s_y, s_vy, e_t, e_y, e_vy)

        x_ref = cx[0] + cx[1]*sim_t + cx[2]*np.power(sim_t, 2) + cx[3]*np.power(sim_t, 3)
        y_ref = cy[0] + cy[1]*sim_t + cy[2]*np.power(sim_t, 2) + cy[3]*np.power(sim_t, 3)

        #define psi reference signal
        dx = x_ref[1:len(sim_t)] - x_ref[0:len(sim_t) - 1]
        dy = y_ref[1:len(sim_t)] - y_ref[0:len(sim_t) - 1]

        x_dot = dx/dt
        y_dot = dy/dt

        x_dot = np.concatenate(([x_dot[0]], x_dot), axis = 0)
        y_dot = np.concatenate(([y_dot[0]], y_dot), axis = 0)

        psi = np.zeros(len(sim_t))
        psi_ref = np.zeros(len(sim_t))
        psi[0] = np.arctan2(dy[0], dx[0])
        psi[1:len(sim_t)] = np.arctan2(dy[0:len(sim_t)], dx[0:len(sim_t)])

        psi_ref[0] = psi[0]
        psi_diff   = psi[1:len(sim_t)] - psi[0:len(sim_t) - 1]

        for i in range(1, len(sim_t)):
            psi_d = psi_diff[i - 1]
            psi_ref[i] = psi_ref[i - 1] + psi_d
            #ensure np.deg2rad(-180.) <= psi_d <= np.deg2rad(180.) 
            if psi_d > np.math.pi: 
                psi_ref[i] -= 2 * np.math.pi
            elif psi_d < -np.math.pi:
                psi_ref[i] += 2 * np.math.pi

        x_dot_body =  np.cos(psi_ref)*x_dot + np.sin(psi_ref)*y_dot
        y_dot_body = -np.sin(psi_ref)*x_dot + np.cos(psi_ref)*y_dot

        return x_dot_body, y_dot_body, x_ref, y_ref, psi_ref

    def generate_ref_signal(self, cur_t, end_t, dt, y_target):
        sim_t = np.arange(cur_t, end_t + dt, dt)

        #define x and y reference signal
        x_ref = np.linspace(0, self.x_dot*sim_t[-1], num = len(sim_t))

        y_ref = []
        trajectory_duration = int(len(sim_t)/len(y_target))
        for i, yt in enumerate(y_target):
            if i < len(y_target) - 1:
                y_ref += [yt]*trajectory_duration
            else:
                y_ref += [yt]*(len(sim_t) - len(y_ref))
        y_ref = np.array(y_ref)

        #define psi reference signal
        dx = x_ref[1:len(sim_t)] - x_ref[0:len(sim_t) - 1]
        dy = y_ref[1:len(sim_t)] - y_ref[0:len(sim_t) - 1]

        x_dot = dx/dt
        y_dot = dy/dt

        x_dot = np.concatenate(([x_dot[0]], x_dot), axis = 0)
        y_dot = np.concatenate(([y_dot[0]], y_dot), axis = 0)

        psi = np.zeros(len(sim_t))
        psi_ref = np.zeros(len(sim_t))
        psi[0] = np.arctan2(dy[0], dx[0])
        psi[1:len(sim_t)] = np.arctan2(dy[0:len(sim_t)], dx[0:len(sim_t)])

        psi_ref[0] = psi[0]
        psi_diff   = psi[1:len(sim_t)] - psi[0:len(sim_t) - 1]

        for i in range(1, len(sim_t)):
            psi_d = psi_diff[i - 1]
            psi_ref[i] = psi_ref[i - 1] + psi_d
            #ensure np.deg2rad(-180.) <= psi_d <= np.deg2rad(180.) 
            if psi_d > np.math.pi: 
                psi_ref[i] -= 2 * np.math.pi
            elif psi_d < -np.math.pi:
                psi_ref[i] += 2 * np.math.pi

        x_dot_body =  np.cos(psi_ref)*x_dot + np.sin(psi_ref)*y_dot
        y_dot_body = -np.sin(psi_ref)*x_dot + np.cos(psi_ref)*y_dot

        return x_dot_body, y_dot_body, x_ref, y_ref, psi_ref
