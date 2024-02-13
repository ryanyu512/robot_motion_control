import numpy as np
import copy as copy
from qpsolvers import *
from robot_base import *

class MPC(Robot_Base):

    def __init__(self, N_in = 2, N_out = 4, h_windows = 10):
        Robot_Base.__init__(self)

        #define the size of inputs and outputs
        self.N_in = N_in
        self.N_out = N_out

        #define prediction windows
        self.h_windows = h_windows

        #output = [x_dot, psi, X, Y]
        self.Q = np.array([[1,    0,  0,  0], 
                           [0,  200,  0,  0],
                           [0,    0, 50,  0],
                           [0,    0,  0, 50]]) #weights for output i to i + h_windows - 1 
        self.S = np.array([[1,    0,  0,  0], 
                           [0,  200,  0,  0],
                           [0,    0, 50,  0],
                           [0,    0,  0, 50]]) #weights for output h_windows
        
        #input = [psi, acc_x]
        self.R = np.array([[100., 0], [0, 1.]]) #weight for inputs

    def get_state_space(self, c_state, u1, u2, dt):

        '''
        simplify the ground vehicle as bicycle model

        state_vector = [x_dot, y_dot, psi, psi_dot, X, Y]
        x_dot = longitudinal velocity relative to body frame
        y_dot = lateral velocity relative to body frame
        psi = heading relative to global x - axis
        psi_dot = heading velocity
        X  = X position in global frame
        Y  = Y position in global frame

        x_dot_dot = u2 + (- F_yf*np.sin(u1) - mu*m*g)/m + psi_dot*y_dot
                  = u2 -Cf*(u1 - y_dot/x_dot - psi_dot*Lf/x_dot)*np.sin(u1)/m - mu*g + psi_dot*y_dot
                  = u2 -Cf*u1/m*sin(u1) + Cf*y_dot*sin(u1)/x_dot/m + Cf*psi_dot*Lf/x_dot*sin(u1)/m - mu*g + psi_dot*y_dot
                  = (-mu*g/x_dot)*x_dot + Cf*sin(u1)/x_dot/m*y_dot + (Cf*Lf*sin(u1)/x_dot/m + y_dot)*psi_dot +
                    (-Cf*sin(u1)/m)*u1 + u2
        
        y_dot_dot = (F_yf*cos(u1) + F_yr)/m - psi_dot*x_dot        
                  = Cf*(u1 - y_dot/x_dot - psi_dot*Lf/x_dot)*cos(u1)/m + Cr*(-y_dot/x_dot + psi_dot*Lr/x_dot)/m - psi_dot*x_dot
                  = Cf*u1*cos(u1)/m - Cf*y_dot*cos(u1)/x_dot/m - Cf*psi_dot*Lf*cos(u1)/x_dot/m - Cr*y_dot/x_dot/m + Cr*psi_dot*Lr/x_dot/m - 
                    psi_dot*x_dot   
                  = (-Cf*cos(u1)/x_dot/m - Cr/x_dot/m)*y_dot + (-Cf*Lf*cos(u1)/x_dot/m + Cr*Lr/x_dot/m - x_dot)*psi_dot + 
                    (Cf*cos(u1)/m)*u1

        psi_dot   = psi_dot                    

        psi_dot_dot = (-F_yr*Lr + F_yf*np.cos(u1)*Lf)/Iz
                    = (-Cr*(-y_dot/x_dot + psi_dot*Lr/x_dot)*Lr + Cf*(u1 - y_dot/x_dot - psi_dot*Lf/x_dot)*np.cos(u1)*Lf)/Iz
                    =  Cr*y_dot/x_dot*Lr/Iz -Cr*psi_dot*Lr**2/x_dot/Iz + Cf*u1*cos(u1)*Lf/Iz - Cf*y_dot/x_dot*cos(u1)*Lf/Iz - Cf*psi_dot*Lf**2/x_dot*cos(u1)/Iz
                    = (Cr*Lr/x_dot/Iz - Cf*Lf*cos(u1)/x_dot/Iz)*y_dot + (-Cr*Lr**2/x_dot/Iz - Cf*Lf**2*cos(u1)/x_dot/Iz)*psi_dot + (Cf*Lf*cos(u1)/Iz)*u1

        X_dot     = np.cos(psi)*x_dot - np.sin(psi)*y_dot
        Y_dot     = np.sin(psi)*x_dot + np.cos(psi)*y_dot                    

        u1 = steering angle of front wheel
        u2 = applied acceleration
        '''

        g = self.g
        m = self.m
        Iz = self.Iz
        Cf = self.Cf
        Cr = self.Cr
        Lf = self.Lf
        Lr = self.Lr
        mu = self.mu

        x_dot = c_state[0]
        y_dot = c_state[1]
        psi   = c_state[2]

        a00 = -mu*g/x_dot
        a01 = Cf*np.sin(u1)/x_dot/m
        a03 = Cf*Lf*np.sin(u1)/x_dot/m + y_dot

        a11 = -Cf*np.cos(u1)/x_dot/m - Cr/x_dot/m
        a13 = -Cf*Lf*np.cos(u1)/x_dot/m + Cr*Lr/x_dot/m - x_dot

        a23 = 1.

        a31 =  Cr*Lr/x_dot/Iz - Cf*Lf*np.cos(u1)/x_dot/Iz
        a33 = -Cr*Lr**2/x_dot/Iz - Cf*Lf**2*np.cos(u1)/x_dot/Iz

        a40 =   np.cos(psi)
        a41 = - np.sin(psi)
        
        a50 = np.sin(psi)
        a51 = np.cos(psi)

        b00 = -Cf*np.sin(u1)/m
        b01 = 1.
        b10 = Cf*np.cos(u1)/m
        b30 = Cf*Lf*np.cos(u1)/Iz
        
        #define A matrix
        A = np.zeros((c_state.shape[0], c_state.shape[0]))

        A[0,0] = a00
        A[0,1] = a01
        A[0,3] = a03

        A[1,1] = a11
        A[1,3] = a13

        A[2,3] = a23

        A[3,1] = a31
        A[3,3] = a33

        A[4,0] = a40
        A[4,1] = a41

        A[5,0] = a50
        A[5,1] = a51

        #define B matrix
        B = np.zeros((c_state.shape[0], self.N_in))
        B[0,0] = b00
        B[0,1] = b01
        B[1,0] = b10
        B[3,0] = b30

        #measurement output = [x_dot, psi, X, Y]
        C = np.zeros((self.N_out, c_state.shape[0]))

        C[0,0] = 1.
        C[1,2] = 1.
        C[2,4] = 1.
        C[3,5] = 1.

        #define D matrix
        D = np.zeros((self.N_out, self.N_in))

        # Discretise the system (forward Euler)
        '''
        (x_k+1 - x_k)/dt = A*x_k + B*u_k
         x_k+1 = A*dt*x_k + B*dt*u_k + x_k
         x_k+1 = (I + dt*A)*x_k + (B*dt)*x_k
         y_k   = C*x_k + D*u_k
         y_k   = [psi_d, Y]
        '''

        self.Ad = np.identity(A.shape[1]) + dt*A
        self.Bd = B*dt
        self.Cd = C  
        self.Dd = D  
    
    def get_aug_ABCD(self):
        #=== compute augmented state matrix ===#
        Ad, Bd, Cd, Dd = self.Ad, self.Bd, self.Cd, self.Dd

        A_aug = np.concatenate((Ad, Bd), axis = 1)
        tmp   = np.concatenate((np.zeros((Bd.shape[1], Ad.shape[1])), 
                                np.identity(Bd.shape[1])), axis = 1)
        A_aug = np.concatenate((A_aug, tmp), axis = 0)

        B_aug = np.concatenate((Bd, np.identity(Bd.shape[1])), axis = 0)
        C_aug = np.concatenate((Cd, np.zeros((Cd.shape[0], Bd.shape[1]))), axis = 1)
        D_aug = Dd

        return A_aug, B_aug, C_aug, D_aug

    def compute_aug_constraints(self, c_aug_state, du, dt):

        #=== get physical constants ===#
        Cf = self.Cf
        Cr = self.Cr
        Lf = self.Lf
        mu = self.mu
        m  = self.m
        g  = self.g

        #=== get augmented A,B,C,D matrix
        A_aug, B_aug, C_aug, D_aug = self.get_aug_ABCD()

        #=== input constraints formulation ===#
        '''
            G_input*[u_k+1;u_k+2;...u_k+N] = H_input
            H_input = [in_UB; in_LB]
        '''
        u1_min = -np.pi/300.
        u1_max =  np.pi/300.
        u2_min = -0.1
        u2_max =  0.1

        #define input lower and upper bounds
        in_LB = np.zeros((self.N_in*self.h_windows, 1))
        in_UB = np.zeros((self.N_in*self.h_windows, 1))
        for i in range(self.N_in*self.h_windows):
            if i % self.N_in:
                in_LB[i] = u2_min
                in_UB[i] = u2_max
            else:
                in_LB[i] = u1_min
                in_UB[i] = u1_max

        G_input = np.concatenate(( np.eye(self.N_in*self.h_windows),
                                  -np.eye(self.N_in*self.h_windows)),
                                  axis = 0)
        h_input = np.concatenate((in_UB, in_LB), axis = 0)

        #=== state constraints formulation ===#
        '''
            G_input*[u_k+1;u_k+2;...u_k+N] = H_input
            H_input = [in_UB; in_LB]
        '''

        #need to extract states from augmented states 
        #aug_state = [x_dot, y_dot, psi, psi_dot, X, Y, delta, acc_x]
        state_ext_mat = np.zeros((self.N_out, A_aug.shape[1]))
        state_ext_mat[0,0] = 1.
        state_ext_mat[1,2] = 1.
        state_ext_mat[2,6] = 1.
        state_ext_mat[3,7] = 1.

        state_ext_mat_G = np.zeros((state_ext_mat.shape[0]*self.h_windows, 
                                    state_ext_mat.shape[1]*self.h_windows))
        
        h_states_UB = []
        h_states_LB = []

        #=== compute augmented matrix constant ===#
        #Q_aug = [[C^t*Q*C, 0, 0, 0, 0]; ... ; [0, 0, 0, C^t*Q*C, 0]; [0, 0, 0, 0, C^t*S*C]]
        #Q_aug = (CtQC.shape[0]*h_windows, CtQC.shape[1]*h_windows) 
        CtQC = np.matmul(np.matmul(np.transpose(C_aug), self.Q), C_aug)
        CtSC = np.matmul(np.matmul(np.transpose(C_aug), self.S), C_aug)
        Q_aug = np.zeros((CtQC.shape[0]*self.h_windows, CtQC.shape[1]*self.h_windows))

        #R_aug = [[R, 0, 0, 0, 0]; ... ; [0, 0, 0, 0, R]]
        R_aug = np.zeros((self.R.shape[0]*self.h_windows, self.R.shape[1]*self.h_windows))

        #C_AB = (B_aug.shape[0]*h_windows, B_aug.shape[1]*self.h_windows)
        C_AB = np.zeros((B_aug.shape[0]*self.h_windows, B_aug.shape[1]*self.h_windows))

        #A_pow = [A; A^2; A^3; A^4;...; A^t]
        A_pow = np.zeros((A_aug.shape[0]*self.h_windows, A_aug.shape[1]))
        
        #T_aug = [[Q*C, 0, 0, 0, 0]; ... ; [0, 0, 0, Q*C, 0]; [0, 0, 0, 0, S*C]]
        QC = np.matmul(self.Q, C_aug)
        SC = np.matmul(self.S, C_aug)
        T_aug = np.zeros((QC.shape[0]*self.h_windows, QC.shape[1]*self.h_windows))

         #=== compute augmented matrix  ===#
        A_prod = copy.copy(A_aug)
        aug_state_pred = copy.copy(c_aug_state)
        A_aug_preds = np.zeros((self.h_windows, A_aug.shape[0], A_aug.shape[1]))
        B_aug_preds = np.zeros((self.h_windows, B_aug.shape[0], B_aug.shape[1]))
        for i in range(self.h_windows):

            Q_aug[CtQC.shape[0]*i:CtQC.shape[0]*i+CtQC.shape[0],
                  CtQC.shape[1]*i:CtQC.shape[1]*i+CtQC.shape[1]] = CtQC if i < self.h_windows - 1 else CtSC

            T_aug[QC.shape[0]*i:QC.shape[0]*i+QC.shape[0],
                  QC.shape[1]*i:QC.shape[1]*i+QC.shape[1]] = QC if i < self.h_windows - 1 else SC

            R_aug[self.R.shape[0]*i:self.R.shape[0]*i+self.R.shape[0],
                  self.R.shape[1]*i:self.R.shape[1]*i+self.R.shape[1]] = self.R

            #A_pow = [[A0], [A1*A0], ...]
            A_pow[A_aug.shape[0]*i:A_aug.shape[0]*i+A_aug.shape[0],
                  0:A_aug.shape[1]] = A_prod
            
            A_aug_preds[i, :, :] = copy.copy(A_aug)
            B_aug_preds[i, :, :] = copy.copy(B_aug)

            #=== state constraints formulation ===#
            x_dot_min = 1.
            x_dot_max = 30.
            y_dot_min = max([-0.17*aug_state_pred[0, 0], -3.])
            y_dot_max = min([ 0.17*aug_state_pred[0, 0],  3.])
            delta_min = -np.math.pi/6.
            delta_max =  np.math.pi/6.
            x_dot_dot_min = -4.
            x_dot_dot_max =  1.
            '''
            F_yf = Cf*(u1 - y_dot/x_dot - psi_dot*Lf/x_dot)
            x_dot_dot = u2 + (- F_yf*np.sin(u1) - mu*m*g)/m + psi_dot*y_dot
            u2 = x_dot_dot - (- F_yf*np.sin(u1) - mu*m*g)/m - psi_dot*y_dot
            '''
            F_yf = Cf*(aug_state_pred[6,0] - aug_state_pred[1,0]/aug_state_pred[0,0] - aug_state_pred[3,0]*Lf/aug_state_pred[0,0])
            a_min = x_dot_dot_min - (- F_yf*np.sin(aug_state_pred[6,0]) - mu*m*g)/m - aug_state_pred[3,0]*aug_state_pred[1,0]
            a_max = x_dot_dot_max - (- F_yf*np.sin(aug_state_pred[6,0]) - mu*m*g)/m - aug_state_pred[3,0]*aug_state_pred[1,0]

            UB_states = np.array([x_dot_max, y_dot_max, delta_max, a_max])
            LB_states = np.array([x_dot_min, y_dot_min, delta_min, a_min])
            h_states_UB  = np.concatenate((h_states_UB, UB_states), axis = 0)
            h_states_LB  = np.concatenate((h_states_LB, LB_states), axis = 0)

            state_ext_mat_G[state_ext_mat.shape[0]*i:state_ext_mat.shape[0]*i + state_ext_mat.shape[0],
                            state_ext_mat.shape[1]*i:state_ext_mat.shape[1]*i + state_ext_mat.shape[1]] = state_ext_mat
        
            if i < self.h_windows - 1:
                du1 = du[self.N_in*(i + 1), 0]
                du2 = du[self.N_in*(i + 1) + self.N_in - 1, 0]

                aug_state_pred = np.matmul(A_aug, aug_state_pred) + np.matmul(B_aug, np.array([[du1], [du2]]))
                state_pred = aug_state_pred[0:6, 0]
                delta_pred = aug_state_pred[6, 0]
                acc_pred   = aug_state_pred[7, 0]

                self.get_state_space(state_pred, delta_pred, acc_pred, dt)
                A_aug, B_aug, C_aug, D_aug = self.get_aug_ABCD()
                #A_pow = [[A0], [A1*A0], ...]
                A_prod = np.matmul(A_aug, A_prod)


            for j in range(self.h_windows):
                if j <= i:
                    C_AB[B_aug.shape[0]*i:B_aug.shape[0]*i+B_aug.shape[0],
                         B_aug.shape[1]*j:B_aug.shape[1]*j+B_aug.shape[1]] = np.matmul(np.linalg.matrix_power(A_aug,((i+1)-(j+1))), B_aug)

        for i in range(self.h_windows): #for row
            for j in range(self.h_windows): #for col
                '''
                C_AB = [
                        [    B0,    0,  0],
                        [  A1B0,   B1,  0],
                        [A2A1B0, A2B1, B2],
                       ]
                '''
                if j <= i: 
                    AB_prod = np.eye(A_aug.shape[0])
                    for k in range(i, j - 1, -1):
                        '''
                        #from kth row to jth row
                        if k = 2, j = 0 => A2, A1, B0
                        if k = 2, j = 1 => A2, B1
                        if k = 0, j = 0 => B0
                        if k = 1, j = 0 => A1, B0
                        '''
                        if k > j: 
                            AB_prod = np.matmul(AB_prod, A_aug_preds[k, :, :])
                        else:
                            AB_prod = np.matmul(AB_prod, B_aug_preds[k, :, :])

                    C_AB[AB_prod.shape[0]*i:AB_prod.shape[0]*i + AB_prod.shape[0],
                         AB_prod.shape[1]*j:AB_prod.shape[1]*j + AB_prod.shape[1]] = AB_prod

        #=== continue to formulate constraints ===#
        '''
        state_ext_mat_G*C_AB + state_ext_mat_G*A_pow*x_t <= y_max
        state_ext_mat_G*C_AB <= y_max - state_ext_mat_G*A_pow*x_t

         state_ext_mat_G*C_AB + state_ext_mat_G*A_pow*x_t >= y_min
         state_ext_mat_G*C_AB >=  y_min - state_ext_mat_G*A_pow*x_t
        -state_ext_mat_G*C_AB <= -y_min + state_ext_mat_G*A_pow*x_t
        '''
        G_state = np.concatenate(( np.matmul(state_ext_mat_G, C_AB), 
                                  -np.matmul(state_ext_mat_G, C_AB)),
                                  axis = 0)
        
        h_states_UB.shape = (len(h_states_UB), 1)
        h_states_LB.shape = (len(h_states_LB), 1)
        h_state = np.concatenate((h_states_UB - np.matmul(state_ext_mat_G, np.matmul(A_pow, c_aug_state)),
                                 -h_states_LB + np.matmul(state_ext_mat_G, np.matmul(A_pow, c_aug_state))),
                                 axis = 0) 
    
        h = np.concatenate((h_input, h_state), axis = 0)
        g = np.concatenate((G_input, G_state), axis = 0)
        print(f"h: {h.shape}, g: {g.shape}")
        #H = (C_AB^t*Q_aug*C_AB + R_aug)
        H  = np.matmul(np.matmul(np.transpose(C_AB), Q_aug), C_AB) + R_aug

        #[A_pow^t*Q_aug*C_AB; -T_aug*C_AB]
        tmp1 = np.matmul(np.matmul(np.transpose(A_pow), Q_aug), C_AB)
        tmp2 = np.matmul(-T_aug, C_AB)
        Ft = np.concatenate((tmp1, tmp2), axis = 0)

        return H, Ft, C_AB, A_pow, g, h

    def compute_control_input(self, H, Ft, g, h, c_aug_state, aug_r):

        #Ug = inv(H)*(-F*[x_k; Rg])
        
        
        tmp = np.concatenate((np.transpose(c_aug_state), np.transpose(aug_r)), axis = 1)

        tmp = np.matmul(tmp, Ft)
        du = solve_qp(H, tmp, g, np.transpose(h), solver="cvxopt")
        print(du)
        #out target control input = Ug[0, 0]
        return du[0, 0]

    #==== derivation section ====#

    '''
    augment state vector
    reason: I don't want absolute steering angle (u) but change (du) in steering angle for smooth input optimization

    x_k+1 = Ad*x_k + Bd*u_k
    x_k+1 = Ad*x_k + Bd*(u_k + du_k)
    x_k+1 = Ad*x_k + Bd*u_k + Bd*du_k
    x_aug_k+1 = [[Ad, Bd], [0,0,0,0,1]]*[x_k;u_k] + Bd*du_k
    x_aug_k+1 = A_aug*x_aug_k + [Bd;1]*du_k
    last_state = u_k+1 = u_k + du_k

    y_k   = [Cd, np.zeros((2, 1))]*x_aug_k + Dd*u_k
    y_k   = [psi_d, Y]
    '''

    '''
    expanding cost function

    J = 0.5*e[-1]^t*S*e[-1] + 0.5*sum[e[i]^t*Q*e[i] + du[i]^T*R*du[i] for i in range(len(e) - 1)]
    Note: 
    e[i] = r[i] - C*x[i]
    J = 0.5*(r[-1] - C*x[-1])^t*S*(r[-1] - C*x[-1]) + 
        0.5*[(r[i] - C*x[i])^t*Q*(r[i] - C*x[i]) + du[i]^T*R*du[i] for i in range(len(e) - 1)]
    Note: 
    (r[-1] - C*x[-1])^t*S*(r[-1] - C*x[-1]
    = (r[-1]^t - x[-1]^t*C^t)*S*(r[-1] - C*x[-1])
    = r[-1]^t*S*r[-1] - r[-1]^t*S*C*x[-1] - x[-1]^t*C^t*S*r[-1] + x[-1]^t*C^t*S*C*x[-1]
    = r[-1]^t*S*r[-1] - r[-1]^t*S*C*x[-1] -(r[-1]^t*S^t*C*x[-1])^t + x[-1]^t*C^t*S*C*x[-1]
    Note: 
    r[-1]^t*S*C*x[-1] = (1, 2)*(2, 2)*(2, 5)*(2, 1) = (1, 1)
    r[-1]^t*S*C*x[-1] = r[-1]^t*S^t*C*x[-1] = scalar

    (r[-1] - C*x[-1])^t*S*(r[-1] - C*x[-1] 
    = r[-1]^t*S*r[-1] - 2*r[-1]^t*S*C*x[-1] + x[-1]^t*C^t*S*C*x[-1]

    so, 
    J = 0.5*(r[-1]^t*S*r[-1] - 2*r[-1]^t*S*C*x[-1] + x[-1]^t*C^t*S*C*x[-1]) + 
        0.5*sum[(r[i] - C*x[i])^t*Q*(r[i] - C*x[i]) + du[i]^T*R*du[i] for i in range(len(e) - 1)]

        = 0.5*r[-1]^t*S*r[-1] - r[-1]^t*S*C*x[-1] + 0.5*x[-1]^t*C^t*S*C*x[-1] + 
        0.5*sum[(r[i] - C*x[i])^t*Q*(r[i] - C*x[i]) + du[i]^T*R*du[i] for i in range(len(e) - 1)]

        = 0.5*r[-1]^t*S*r[-1] - r[-1]^t*S*C*x[-1] + 0.5*x[-1]^t*C^t*S*C*x[-1] + 
        0.5*sum[r[i]^t*Q*r[i] - 2*r[i]^t*Q*C*x[i] + x[i]^t*C^t*Q*C*x[i] + du[i]^t*R*du[i] for i in range(len(e) - 1)]
        = 0.5*r[-1]^t*S*r[-1] - r[-1]^t*S*C*x[-1] + 0.5*x[-1]^t*C^t*S*C*x[-1] + 
        sum[0.5*r[i]^t*Q*r[i] - r[i]^t*Q*C*x[i] + 0.5*x[i]^t*C^t*Q*C*x[i] + 0.5*du[i]^t*R*du[i] for i in range(len(e) - 1)]
    Note:
    set r[-1]^t*S*r[-1] = r[i]^t*Q*r[i] = 0 since they are purely offset => does not affect optimization
    when i = 0, set r[i]^t*Q*C*x[i] + 0.5*x[i]^t*C*Q*C*x[i] = 0 since we already use it => no need to optimise

    a = r[-1]^t*S*C*x[-1]
    b = 0.5*x[-1]^t*C^t*S*C*x[-1]
    c = r[i]^t*Q*C*x[i]
    d = 0.5*x[i]^t*C^t*Q*C*x[i]
    e = 0.5*du[i]^t*R*du[i]

    so, 

    J = -a + b + e[0] + (-c[1] + d[1] + e[1]) + ... + (-c[4] + d[4] + e[4])
            
    '''

    '''
    formulate polynormial form into matrix form 
    
    for example, 
    sum[x[i]^t*Q*x[i] for i in range(3)]
    = x[0]^t*Q*x[0] + ... + x[2]^t*Q*x[2]
    = [x[0]^t, x[1]^t, x[2]^t]*[[Q[0], 0, 0]; [0, Q[1], 0]; [0, 0, Q[2]]*[x[0]; x[1]; x[2]]

    Note: 
    Xg = [x[0]^t, ..., x[4]^t]^t = (2*5, 1)
    Rg = [r[0]^t, ..., r[4]^t]^t = (2*5, 1)
    Ug = [du[0]^t, ..., du[4]^t]^t = (5, 1)
    J = - r[-1]^t*S*C*x[-1] + 0.5*x[-1]^t*C^t*S*C*x[-1] + 
        sum[ - r[i]^t*Q*C*x[i] + 0.5*x[i]^t*C^t*Q*C*x[i] + 0.5*du[i]^t*R*du[i] for i in range(len(e) - 1)]
    = 0.5*Xg^t*[[C^t*Q[0]*C, 0, 0, 0, 0]; ... ; [0, 0, 0, C^t*Q[3]*C, 0]; [0, 0, 0, 0, C^t*S*C]]*Xg -
        Rg^t*[[Q[0]*C, 0, 0, 0, 0]; ... ; [0, 0, 0, Q[3]*C, 0]; [0, 0, 0, 0, S*C]]*Xg + 
        0.5*Ug^t*[[R[0], 0, 0, 0, 0]; ... ; [0, 0, 0, 0, R[4]]]*Ug
    
    so, 
    Q_aug = [[C^t*Q*C, 0, 0, 0, 0]; ... ; [0, 0, 0, C^t*Q*C, 0]; [0, 0, 0, 0, C^t*S*C]]
            = (5*5, 5*5) 
    T_aug = [[Q*C, 0, 0, 0, 0]; ... ; [0, 0, 0, Q*C, 0]; [0, 0, 0, 0, S*C]]
            = (2*5, 5*5) 
    R_aug = [[R, 0, 0, 0, 0]; ... ; [0, 0, 0, 0, R]]
    
    so, 
    J = 0.5*Xg^t*Q_aug*Xg - Rg^t*T_aug*Xg + 0.5*Ug^t*R_aug*Ug

    Note: 
    assume horizon window = 5

    x_k+1 = A*x_k + B*u_k
    x_k+2 = A*x_k+1 + B*u_k+1 = A*(A*x_k + B*u_k) + B*u_k+1 = A^2*x_k + A*B*u_k + B*u_k+1 = [A*B, B]*[u_k;u_k+1] + A^2*x_k
    x_k+3 = [A^2*B, A*B, B]*[u_k;u_k+1;u_k+1] + A^3*x_k
    x_k+4 = [A^3*B, A^2*B, A*B, B]*[u_k;u_k+1;u_k+2;u_k+3] + A^4*x_k
    x_k+5 = [A^4*B, A^3*B, A^2*B, A*B, B]*[u_k;u_k+1;u_k+2;u_k+3; u_k+4] + A^5*x_k

    stack up into one matrix
    C_AB  = [    B,     0,     0,   0, 0;
                A*B,     B,     0,   0, 0; 
                A^2*B,   A*B,     B,   0, 0; 
                A^3*B, A^2*B,   A*B,   B, 0;
                A^4*B, A^3*B, A^2*B; A*B, B] = (5*5, 5)
    A_pow = [A; A^2; A^3; A^4; A^t]

    so, 
    Xg = C_AB*Ug + A_pow*x_k

    so, 
    J = 0.5*Xg^t*Q_aug*Xg - Rg^t*T_aug + 0.5*Ug^t*R*Ug
        = 0.5*(C_AB*Ug + A_pow*x_k)^t*(C_AB*Ug + A_pow*x_k) - 
        Rg^t*T_aug*(C_AB*Ug + A_pow*x_k) + 
        0.5*Ug^t*R_aug*Ug
        = 0.5*(Ug^t*C_AB^t*Q_aug*C_AB*Ug + Ug^t*C_AB^t*Q_aug*A_pow*x_k + x_k^t*A_pow^t*Q_aug*C_AB*Ug + 
                x_k^t*A_pow^t*A_pow*x_k) - 
        Rg^t*T_aug*C_AB*Ug - Rg^t*T_aug*A_pow*x_k + 
        0.5*Ug^t*R_aug*Ug
        
    Note: 
    1) since x_k is already known => no need to optimise
    set x_k^t*A_pow^t*A_pow*x_k = Rg^t*T_aug*A_pow*x_k = 0 

    2) Ug^t*C_AB^t*Q_aug*A_pow*x_k = x_k^t*A_pow^t*Q_aug*C_AB*Ug = (1, 1) = scalar

    so,
    J  = 0.5*(Ug^t*C_AB^t*Q_aug*C_AB*Ug + x_k^t*A_pow^t*Q_aug*C_AB*Ug) - Rg^t*T_aug*C_AB*Ug + 
            0.5*Ug^t*R_aug*Ug

    J = 0.5*Ug^t*(C_AB^t*Q_aug*C_AB + R_aug)*Ug + [x_k^t, Rg^t]*[A_pow^t*Q_aug*C_AB; -T_aug*C_AB]*Ug

    Note:
    H   = (C_AB^t*Q_aug*C_AB + R_aug) = (5, 25)*(25, 25)*(25, 5) + (5, 5) = (5, 5)
    F^t = [A_pow^t*Q_aug*C_AB; -T_aug*C_AB] = (5, 25)*(25, 25)*(25, 5) + (10, 25)*(25, 5) = (5+10, 5) 

    so, 
    J  = 0.5*Ug^t*H*Ug + [x_k^t, Rg^t]*F^t*Ug

    Note: 
    [x_k^t, Rg^t]*F^t*Ug = (1, 5 + 2*5)*(5+10, 5)*(5, 1) = (1, 1)

    dJ = H*Ug + F*[x_k^t; Rg^t] = (5, 5)*(5, 1) + (5, 15)*(15, 1) = (5, 1) + (5, 1) = (5, 1)
    
    Note: 
    set dJ = 0 -> min. point of cost function (quadratic function)
    
    so,
    Ug = inv(H)*(-F*[x_k^t; Rg^t])

    then, out target control input = Ug[0]

    derive complete
    '''