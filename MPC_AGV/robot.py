import copy as copy
import numpy as np
from robot_base import *

class Robot(Robot_Base):

    def __init__(self):
        Robot_Base.__init__(self)

    def update(self, state, u1, u2, dt = 0.02):

        #get constants
        g = self.g
        m = self.m
        Iz = self.Iz
        Cf = self.Cf
        Cr = self.Cr
        Lf = self.Lf
        Lr = self.Lr
        mu = self.mu

        #get current states
        c_state = copy.copy(state)
        n_state = copy.copy(state)

        x_dot = c_state[0,0]
        y_dot = c_state[1,0]
        psi   = c_state[2,0]
        psi_dot = c_state[3,0]
        X     = c_state[4,0]
        Y     = c_state[5,0]

        sub_loop = 30
        sub_dt = dt/sub_loop 

        for i in range(0, sub_loop):

            '''
            compute lateral force (front wheel)(body frame)

            F_yf = Cf*alpha_f

            Note: 
            alpha_f = command_steering_angle - actual_velocity_angle
            tan(actual_velocity_angle) = (y_dot + psi_dot*L_f)/x_dot
            we assume that the -pi/6 <= command_steering_angle <= pi/6
            actual_velocity_angle = (y_dot + psi_dot*L_f)/x_dot

            so, F_yf = Cf*(command_steering_angle - y_dot/x_dot - psi_dot*Lf/x_dot)

            compute lateral force (rear wheel)(body frame)

            F_yr = Cr*alpha_r
            Note: 
            alpha_r = - actual_velocity_angle
            tan(actual_velocity_angle) = (y_dot - psi_dot*L_f)/x_dot
            we assume that the -pi/6 <= command_steering_angle <= pi/6
            actual_velocity_angle = (y_dot - psi_dot*L_f)/x_dot

            so, F_yr = Cf*(command_steering_angle - y_dot/x_dot + psi_dot*Lr/x_dot)
            '''

            F_yf = Cf*(u1 - y_dot/x_dot - psi_dot*Lf/x_dot)
            F_yr = Cr*(-y_dot/x_dot + psi_dot*Lr/x_dot)

            '''
            compute net force in x-/longitudinal direction (body frame)
            F_x = F_a (applied force) - friction - F_yf*sin(steer_ang)
            F_x = m*a - mu*m*g - F_yf*sin(steer_ang)

            compute net force in y-/lateral direction (body frame) 
            F_y = F_yr + F_yf*cos(steer_ang)

            compute net moment (body frame)
            Iz*psi_dot_dot = -F_yr*Lr + F_yf*cos(steer_ang)*Lf

            Note:
            F_net = m*(x_dot_dot*unit_i + x_dot*unit_i_dot + y_dot_dot*unit_j + y_dot*unit_j_dot)
                  = m*(x_dot_dot*unit_i + y_dot_dot*unit_j + cross(w_z, x_dot*unit_i)  + cross(w_z, y_dot*unit_j))
                  = m*(x_dot_dot*unit_i + y_dot_dot*unit_j + cross(w_z, [x_dot;y_dot;0]))
                  = m*(x_dot_dot*unit_i + y_dot_dot*unit_j + (-psi_dot*y_dot)*unit_i + (psi_dot*x_dot)*unit_j)
                  = m*(unit_i*(x_dot_dot - psi_dot*y_dot) + unit_j*(y_dot_dot + psi_dot*x_dot))

            unit_i_dot != 0 implies that the the direction of unit_i in global frame is changed
            unit_j_dot != 0 implies that the the direction of unit_j in global frame is changed
            unit_i_dot = cross(w_z, unit_i) = cross([0;0;ang_w_z], [1;0;0]) 
            unit_j_dot = cross(w_z, unit_j) = cross([0;0;ang_w_z], [0;1;0])

            cross(w_z, [x_dot;y_dot;0]) = unit_i*(-psi_dot*y_dot) + unit_j*(psi_dot*x_dot)

            so, 
            Fx = m*(x_dot_dot - psi_dot*y_dot)
            m*(x_dot_dot - psi_dot*y_dot) = applied_force - mu*m*g - F_yf*sin(steer_ang)
            x_dot_dot = applied_acc + (- F_yf*sin(steer_ang) - mu*m*g)/m + psi_dot*y_dot

            F_y = m*(x_dot_dot - psi_dot*y_dot) 
            m*(y_dot_dot + psi_dot*x_dot) = F_yr + F_yf*cos(steer_ang)
            y_dot_dot = (F_yf*cos(steer_ang) + F_yr)/m - psi_dot*x_dot

            M_z = Iz*psi_dot_dot = -F_yr*Lr + F_yf*cos(steer_ang)*Lf
            psi_dot_dot = (F_yf*Lf*np.cos(u1)   - F_yr*Lr)/Iz
            '''

            x_dot_dot = u2 + (- F_yf*np.sin(u1) - mu*m*g)/m + psi_dot*y_dot
            y_dot_dot = (F_yf*np.cos(u1) + F_yr)/m - psi_dot*x_dot
            psi_dot   = psi_dot
            psi_dot_dot = (F_yf*np.cos(u1)*Lf - F_yr*Lr)/Iz
            X_dot     = x_dot*np.cos(psi) - y_dot*np.sin(psi)
            Y_dot     = x_dot*np.sin(psi) + y_dot*np.cos(psi)

            # Update the state values with new state derivatives
            x_dot   += x_dot_dot*sub_dt
            y_dot   += y_dot_dot*sub_dt
            psi     += psi_dot*sub_dt
            psi_dot += psi_dot_dot*sub_dt
            X       += X_dot*sub_dt
            Y       += Y_dot*sub_dt

        # Take the last states
        n_state[0,0] = x_dot
        n_state[1,0] = y_dot
        n_state[2,0] = psi
        n_state[3,0] = psi_dot
        n_state[4,0] = X
        n_state[5,0] = Y 

        n_state.shape = (len(n_state), 1)

        return n_state, x_dot_dot, y_dot_dot, psi_dot_dot