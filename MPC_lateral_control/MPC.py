import numpy as np
from robot_base import *

class MPC(Robot_Base):

    def __init__(self, dt, h_windows = 20, ):
        Robot_Base.__init__(self)
        #define prediction windows
        self.h_windows = h_windows
        self.Q = np.array([[50., 0], 
                           [   0, 1]]) #weights for output i to i + h_windows - 1 
        self.S = np.array([[50., 0], 
                           [   0, 1]]) #weights for output h_windows
        self.R = np.array([1.]) #weight for inputs
        self.R.shape = (1, 1)

        self.get_state_space(dt)

    def get_state_space(self, dt):

        a = -(2*self.Cf + 2*self.Cr)/(self.m*self.x_dot)
        b = -self.x_dot - (2*self.Cf*self.Lf - 2*self.Cr*self.Lr)/(self.m*self.x_dot)
        c = -(2*self.Lf*self.Cf - 2*self.Lr*self.Cr)/(self.Iz*self.x_dot)
        d = -(2*self.Lf**2*self.Cf + 2*self.Lr**2*self.Cr)/(self.Iz*self.x_dot)

        '''
        simplify the ground vehicle as bicycle model

        state_vector = [y_dot, psi, psi_dot, Y]
        y_dot = lateral velocity relative to body frame
        psi = heading relative to global x - axis
        psi_dot = heading velocity
        Y  = Y position in global y direction

        y_dd = a*y_d + b*p_d + 2*Cf/m*u
        p_d  = p_d
        p_dd = c*y_d + d*p_d + 2*lf/Iz*u
        Y   = y_d + x_d*p_d

        u    = steering angle of front wheel
        '''
        A = np.array([[a, 0, b, 0],
                      [0, 0, 1, 0],
                      [c, 0, d, 0],
                      [1, self.x_dot, 0, 0]])

        B = np.array([
                      [2*self.Cf/self.m],
                      [0],
                      [2*self.Lf*self.Cf/self.Iz],
                      [0]
                    ])

        #measurement output = [psi_dot; Y]
        C = np.array([
                      [0, 1, 0, 0],
                      [0, 0, 0, 1]
                    ])
        
        D = 0.

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
        
    def compute_aug_matrix(self):

        #compute augmented state matrix 
        A_aug = np.concatenate((self.Ad, self.Bd), axis = 1)
        tmp   = np.concatenate((np.zeros((self.Bd.shape[1], self.Ad.shape[1])), 
                                np.identity(self.Bd.shape[1])), axis = 1)
        A_aug = np.concatenate((A_aug, tmp), axis = 0)

        B_aug = np.concatenate((self.Bd, np.identity(self.Bd.shape[1])), axis = 0)
        C_aug = np.concatenate((self.Cd, np.zeros((self.Cd.shape[0], self.Bd.shape[1]))), axis = 1)
        D_aug = self.Dd

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

        for i in range(self.h_windows):

            Q_aug[CtQC.shape[0]*i:CtQC.shape[0]*i+CtQC.shape[0],
                  CtQC.shape[1]*i:CtQC.shape[1]*i+CtQC.shape[1]] = CtQC if i < self.h_windows - 1 else CtSC

            T_aug[QC.shape[0]*i:QC.shape[0]*i+QC.shape[0],
                  QC.shape[1]*i:QC.shape[1]*i+QC.shape[1]] = QC if i < self.h_windows - 1 else SC

            R_aug[self.R.shape[0]*i:self.R.shape[0]*i+self.R.shape[0],
                  self.R.shape[1]*i:self.R.shape[1]*i+self.R.shape[1]] = self.R

            A_pow[A_aug.shape[0]*i:A_aug.shape[0]*i+A_aug.shape[0],
                  0:A_aug.shape[1]] = np.linalg.matrix_power(A_aug, i + 1)

            for j in range(self.h_windows):
                if j <= i:
                    C_AB[B_aug.shape[0]*i:B_aug.shape[0]*i+B_aug.shape[0],
                         B_aug.shape[1]*j:B_aug.shape[1]*j+B_aug.shape[1]] = np.matmul(np.linalg.matrix_power(A_aug,((i+1)-(j+1))), B_aug)

        #H = (C_AB^t*Q_aug*C_AB + R_aug)
        self.H  = np.matmul(np.matmul(np.transpose(C_AB), Q_aug), C_AB) + R_aug

        #[A_pow^t*Q_aug*C_AB; -T_aug*C_AB]
        tmp1 = np.matmul(np.matmul(np.transpose(A_pow), Q_aug), C_AB)
        tmp2 = np.matmul(-T_aug, C_AB)
        self.Ft = np.concatenate((tmp1, tmp2), axis = 0)

    def mpc_simplification(self):
        '''This function creates the compact matrices for Model Predictive Control'''
        # db - double bar
        # dbt - double bar transpose
        # dc - double circumflex
        '''
            x'_k   = [[x_k], [u_k-1]]
            x'_k+1 = [[A, B], [0, I]]*x'_k + [[B], [I]]*du
            y      = [C, 0]*x'_k 
        '''

        Ad = self.Ad
        Bd = self.Bd
        Cd = self.Cd
        Dd = self.Dd

        A_aug=np.concatenate((Ad,Bd),axis=1)
        #temp1's size = (1, 4)
        #temp2's size = (1, 1)
        #temp's size  = (1, 5)
        temp1=np.zeros((np.size(Bd,1),np.size(Ad,1)))
        temp2=np.identity(np.size(Bd,1))
        temp=np.concatenate((temp1,temp2),axis=1)

        #A_aug's size = (5, 5)
        A_aug=np.concatenate((A_aug,temp),axis=0)
        #B_aug's size = (5, 1)
        B_aug=np.concatenate((Bd,np.identity(np.size(Bd,1))),axis=0)
        #C_aug's size = (2, 5)
        C_aug=np.concatenate((Cd,np.zeros((np.size(Cd,0),np.size(Bd,1)))),axis=1)
        #D_aug's size = (1, 1)
        D_aug=Dd

        Q=self.Q
        S=self.S
        R=self.R
        hz = self.h_windows

        CQC=np.matmul(np.transpose(C_aug),Q)
        CQC=np.matmul(CQC,C_aug)

        CSC=np.matmul(np.transpose(C_aug),S)
        CSC=np.matmul(CSC,C_aug)

        QC=np.matmul(Q,C_aug)
        SC=np.matmul(S,C_aug)

        Qdb=np.zeros((np.size(CQC,0)*hz,np.size(CQC,1)*hz))
        Tdb=np.zeros((np.size(QC,0)*hz,np.size(QC,1)*hz))
        Rdb=np.zeros((np.size(R,0)*hz,np.size(R,1)*hz))
        Cdb=np.zeros((np.size(B_aug,0)*hz,np.size(B_aug,1)*hz))
        Adc=np.zeros((np.size(A_aug,0)*hz,np.size(A_aug,1)))

        for i in range(0,hz):
            '''
            for example,
            [x_k+1; x_k+2] = [[B 0]; [A*B B]]*[du_k; du_k+1] + [A; A**2]
            Adc = [A; A**2]
            Cdb = [[B 0]; [A*B B]]
            ''' 

            if i == hz-1:
                Qdb[np.size(CSC,0)*i:np.size(CSC,0)*i+CSC.shape[0],
                    np.size(CSC,1)*i:np.size(CSC,1)*i+CSC.shape[1]]=CSC
                Tdb[np.size(SC,0)*i:np.size(SC,0)*i+SC.shape[0],
                    np.size(SC,1)*i:np.size(SC,1)*i+SC.shape[1]]=SC
            else:
                Qdb[np.size(CQC,0)*i:np.size(CQC,0)*i+CQC.shape[0],
                    np.size(CQC,1)*i:np.size(CQC,1)*i+CQC.shape[1]]=CQC
                Tdb[np.size(QC,0)*i:np.size(QC,0)*i+QC.shape[0],
                    np.size(QC,1)*i:np.size(QC,1)*i+QC.shape[1]]=QC

            Rdb[np.size(R,0)*i:np.size(R,0)*i+R.shape[0],
                np.size(R,1)*i:np.size(R,1)*i+R.shape[1]]=R

            for j in range(0,hz):
                if j<=i:
                    Cdb[np.size(B_aug,0)*i:np.size(B_aug,0)*i+B_aug.shape[0],
                        np.size(B_aug,1)*j:np.size(B_aug,1)*j+B_aug.shape[1]]=np.matmul(np.linalg.matrix_power(A_aug,((i+1)-(j+1))),B_aug)


            Adc[np.size(A_aug,0)*i:np.size(A_aug,0)*i+A_aug.shape[0],
                0:0+A_aug.shape[1]]=np.linalg.matrix_power(A_aug,i+1)

        #J = 1/2*(du_k)^t*H*du_k + x'_k*F^t*du_k
        #H = C^t*Q*C + R
        #F^t = [[A^t*Q*C], [-T*C]]
        
        self.H=np.matmul(np.transpose(Cdb),Qdb)
        self.H=np.matmul(self.H,Cdb)+Rdb

        temp=np.matmul(np.transpose(Adc),Qdb)
        temp=np.matmul(temp,Cdb)

        temp2=np.matmul(-Tdb,Cdb)
        self.Ft=np.concatenate((temp,temp2),axis=0)

    def compute_control_input(self, c_aug_state, aug_r):

        #Ug = inv(H)*(-F*[x_k; Rg])
        tmp = np.concatenate((c_aug_state, aug_r), axis = 0)
        F   = np.transpose(self.Ft)
        Ug  =-np.matmul(np.linalg.inv(self.H), np.matmul(F, tmp))

        #out target control input = Ug[0, 0]
        return Ug[0, 0]

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