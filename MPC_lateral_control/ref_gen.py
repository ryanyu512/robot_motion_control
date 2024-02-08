import numpy as np
from robot_base import *

class Ref_Gen(Robot_Base):
    def __init__(self):
        Robot_Base.__init__(self)

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

        return x_ref, y_ref, psi_ref
