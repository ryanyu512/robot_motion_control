
![demo](https://github.com/ryanyu512/robot_motion_control/assets/19774686/c15878b1-62b7-4f33-b408-e82e15c219db)

This project utilizes model predictive control (MPC) to achieve full control of an autonomous vehicle. Based on sliding windows of reference trajectories, model predictive control (MPC) computes optimised control inputs to ensure the control inputs are smooth and respects motion constraints. 

1. demo.py: used for testing different parameters of MPC
2. ref_gen.py: generate references for motion control
3. MPC.py: formulate MPC for AGV
4. robot.py: define the properties of 2d vehicle
5. robot_base.py: define the physical constants of 2d vechicle
6. sim.py: custom function for running simulation
