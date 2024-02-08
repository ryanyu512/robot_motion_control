import copy as copy
import numpy as np
from robot_base import *

class Robot(Robot_Base):

    def __init__(self):
        Robot_Base.__init__(self)

    def update(self, state, u, dt = 0.02):

        c_state = copy.copy(state)
        n_state = copy.copy(state)

        y_dot = c_state[0,0]
        psi   = c_state[1,0]
        psi_dot = c_state[2,0]
        Y     = c_state[3,0]

        sub_loop = 30
        sub_dt = dt/sub_loop 

        for i in range(0, sub_loop):
            # Compute the the derivatives of the states
            y_dot_dot   = -(2*self.Cf+2*self.Cr)/(self.m*self.x_dot)*y_dot+(-self.x_dot-(2*self.Cf*self.Lf - 2*self.Cr*self.Lr)/(self.m*self.x_dot))*psi_dot + 2*self.Cf/self.m*u
            psi_dot     = psi_dot
            psi_dot_dot = -(2*self.Lf*self.Cf-2*self.Lr*self.Cr)/(self.Iz*self.x_dot)*y_dot - (2*self.Lf**2*self.Cf+2*self.Lr**2*self.Cr)/(self.Iz*self.x_dot)*psi_dot + 2*self.Lf*self.Cf/self.Iz*u
            Y_dot       = np.sin(psi)*self.x_dot + np.cos(psi)*y_dot

            # Update the state values with new state derivatives
            y_dot   += y_dot_dot*sub_dt
            psi     += psi_dot*sub_dt
            psi_dot += psi_dot_dot*sub_dt
            Y       += Y_dot*sub_dt

        # Take the last states
        n_state[0,0] = y_dot
        n_state[1,0] = psi
        n_state[2,0] = psi_dot
        n_state[3,0] = Y 

        n_state.shape = (len(n_state), 1)

        return n_state