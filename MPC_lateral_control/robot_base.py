class Robot_Base():

    def __init__(self):
        #define mass (kg)
        self.m = 1500.
        #moment of inertia about z - axis
        self.Iz = 3000.
        #length from center of mass to front wheel
        self.Lf = 2.
        #length from center of mass to rear wheel
        self.Lr = 3.
        #stiffness of front tire
        self.Cf = 19000.
        #stiffness of rear tire
        self.Cr = 33000.
        #forward velocity (assume constant)
        #this project is for lateral control
        self.x_dot = 20.