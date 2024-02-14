import numpy as np
import matplotlib.pyplot as plt
from robot_base import *

class Ref_Gen(Robot_Base):
    def __init__(self):
        Robot_Base.__init__(self)
        self.X_waypts = None
        self.Y_waypts = None
        self.X_dot_waypts = None
        self.Y_dot_waypts = None

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

    def get_cubic_path(self, s_t, e_t, dt):


        sim_t = np.arange(s_t, e_t + dt, dt)
        section_duration = sim_t[-1]/(len(self.X_waypts) - 1)

        delay = np.zeros(len(self.X_waypts))
        for i in range(1,len(delay)):
            delay[i]= section_duration + (i - 1)*section_duration

        X=[]
        Y=[]
        for i in range(0,len(delay)-1):
            # Extract the time elements for each section separately
            if i != len(delay)-2:
                t = sim_t[int(delay[i]/dt):int(delay[i+1]/dt)]
            else:
                t = sim_t[int(delay[i]/dt):int(delay[i+1]/dt+1)]

            a_x = self.get_cubic_coeff(st = t[0], 
                                       s_pos = self.X_waypts[i], 
                                       s_vel = self.X_dot_waypts[i],
                                       et = t[-1],
                                       e_pos = self.X_waypts[i+1] - self.X_dot_waypts[i+1]*dt,
                                       e_vel = self.X_dot_waypts[i+1])
            
            a_y = self.get_cubic_coeff(st = t[0], 
                                       s_pos = self.Y_waypts[i], 
                                       s_vel = self.Y_dot_waypts[i],
                                       et = t[-1],
                                       e_pos = self.Y_waypts[i+1] - self.Y_dot_waypts[i+1]*dt,
                                       e_vel = self.Y_dot_waypts[i+1])
            
            # Compute X and Y values
            X_sub = a_x[0, 0]+a_x[1, 0]*t+a_x[2, 0]*t**2+a_x[3, 0]*t**3
            Y_sub = a_y[0, 0]+a_y[1, 0]*t+a_y[2, 0]*t**2+a_y[3, 0]*t**3

            # Concatenate X and Y values
            X=np.concatenate([X, X_sub])
            Y=np.concatenate([Y, Y_sub])

            # Round the numbers to avoid numerical errors
            X=np.round(X,8)
            Y=np.round(Y,8)

        # Vector of x and y changes per sample time
        dX=X[1:len(X)]-X[0:len(X)-1]
        dY=Y[1:len(Y)]-Y[0:len(Y)-1]

        X_dot=dX/dt
        Y_dot=dY/dt
        X_dot=np.concatenate(([X_dot[0]],X_dot),axis=0)
        Y_dot=np.concatenate(([Y_dot[0]],Y_dot),axis=0)

        # Define the reference yaw angles
        psi=np.zeros(len(X))
        psi_ref = psi
        psi[0]=np.arctan2(dY[0],dX[0])
        psi[1:len(psi)]=np.arctan2(dY[0:len(dY)],dX[0:len(dX)])

        # We want the yaw angle to keep track the amount of rotations
        dpsi=psi[1:len(psi)]-psi[0:len(psi)-1]
        psi_ref[0] = psi[0]
        for i in range(1,len(psi_ref)):
            d_psi = dpsi[i - 1]
            if d_psi < -np.pi:
                d_psi += 2*np.pi
            elif d_psi > np.pi:
                d_psi -= 2*np.pi
            
            psi_ref[i] = psi_ref[i-1] + d_psi

        x_dot_body= np.cos(psi_ref)*X_dot+np.sin(psi_ref)*Y_dot
        y_dot_body=-np.sin(psi_ref)*X_dot+np.cos(psi_ref)*Y_dot
        y_dot_body= np.round(y_dot_body)

        # plt.plot(X, Y, '-r')
        # plt.show()

        return x_dot_body, y_dot_body, psi_ref, X, Y            