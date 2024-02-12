class Robot_Base():

    def __init__(self):
        #define gravity 
        self.g = 9.81
        #define mass (kg)
        self.m = 1500.
        #moment of inertia about z - axis
        self.Iz = 3000.
        #length from center of mass to front wheel
        self.Lf = 2.
        #length from center of mass to rear wheel
        self.Lr = 3.
        #stiffness of front tire
        self.Cf = 38000.
        #stiffness of rear tire
        self.Cr = 66000.
        #forward velocity (assume constant)
        #this project is for lateral control
        self.x_dot = 20.
        #define the radius of wheels
        self.R_wheel = 0.5
        #define friction coefficient
        self.mu = 0.02