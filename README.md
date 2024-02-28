# Model Predictive Control For 4-DOF Robotic Arm

A model predictive controller(MPC) is designed to solve the problem of trajectory tracking. Each joints of the robotic arm would generate the trajectory between its current angles to the desired angles.
The desired angles for each joints is obtained from the Inverse Kinematics solved for 4DOF via Optimization. The overall cost function for MPC problem includes the cost function
corresponding to tracking error as well as change in inputs.
