
# Thanks for the awesome content, Woolfrey. It would be of great help if you could just make me clear on the shape of the matrices, delx and qk. For my 4DOF robotic arm, are these shapes right?
# 1. delx= (3X2) where first column for (position error of x,y,x) and second column for (RPY angles of e0.)
# 2. inverse of Jacobian matrix provides shape of 4X3 . so on doing delq=inv(J)*delx, i get delq of shape 4X2.
# 3. ultimately, qk of shape 4X2, where first column for desired angles and second for desired RPY values.

# I appreciate your time.

# =================================================================================
# This one is working inverse kinematics for 4DOF robotic arm without "orientation".
# ==================================================================================


import numpy as np
from scipy.optimize import minimize
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
from scipy.integrate import odeint
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


L1 = 0.289  # in m, length of link1
L2 = 0.372  # in m, length of link2
L3 = 0.351
L4 = 0.33

# Define the objective function (error to minimize)


def objective_function(theta):

    # global errorList
    # errorList = np.array([])

    # Calculate current end effector position using forward kinematics
    current_position = forward_kinematics(theta)

    # Calculate the error (distance) between current and desired positions
    error_pos = np.linalg.norm(current_position - desired_position)

    # calculate current yaw,pitch and roll
    yaw, pitch, roll = YPR(theta)

    e_pitch = np.linalg.norm(pitch)
    e_roll = np.linalg.norm(roll)

    # errorList = np.append(errorList, error)

    return error_pos+e_pitch+e_roll


# Define forward kinematics (replace with your actual forward kinematics function)
def forward_kinematics(q):
    # Implement your forward kinematics to calculate the end effector position
    # based on the given joint angles (theta).
    q1 = (np.deg2rad(q[0])).round(decimals=4)
    q2 = (np.deg2rad(q[1])).round(decimals=4)
    q3 = (np.deg2rad(q[2])).round(decimals=4)
    q4 = (np.deg2rad(q[3])).round(decimals=4)

    Y = np.array([-np.cos(q1)*(L3*np.sin(q2+q3)+L2*np.sin(q2)-L4*np.cos(q2+q3+q4)),
                  -np.sin(q1)*(L3*np.sin(q2+q3)+L2 *
                               np.sin(q2)-L4*np.cos(q2+q3+q4)),
                  L1+L3*np.cos(q2+q3)+L2*np.cos(q2)+L4*np.sin(q2+q3+q4)])

    return Y


def YPR(q):
    q1 = (np.deg2rad(q[0])).round(decimals=4)
    q2 = (np.deg2rad(q[1])).round(decimals=4)
    q3 = (np.deg2rad(q[2])).round(decimals=4)
    q4 = (np.deg2rad(q[3])).round(decimals=4)

    Rot = np.array([[np.cos(q1)*np.cos(q2+q3+q4), np.cos(q1)*-np.sin(q2+q3+q4), np.sin(q1)],
                    [np.sin(q1)*np.cos(q2+q3+q4), np.sin(q1)*-
                     np.sin(q2+q3+q4), -np.cos(q1)],
                    [np.sin(q2+q3+q4), np.cos(q2+q3+q4), 0]])

    roll = np.arctan2(Rot[2, 1], Rot[2, 2])
    yaw = np.arctan2(Rot[1, 0], Rot[0, 0])

    if (np.cos(yaw) == 0):
        pitch = np.arctan2(-Rot[2, 0], (Rot[1, 0]/np.sin(yaw)))
    else:
        pitch = np.arctan2(-Rot[2, 0], (Rot[0, 0]/np.cos(yaw)))

    return yaw, pitch, roll


# Desired end effector position
desired_position = np.array([0.9, -0.025, 0.36])

#  Initial guess for joint angles
initial_guess = np.array([0.0, 0.0, 0.0, 0.0])

# bounds
b1 = [-180.0, 180.0]
b = [-90.0, 90.0]
bnds = (b1, b, b, b)

# Perform optimization
result = minimize(objective_function, initial_guess,
                  method='Powell', bounds=bnds, tol=1e-6)

# Extract the optimized joint angles
optimized_joint_angles = np.array(result.x)

# Display the result
print("Optimized Joint Angles:", optimized_joint_angles.round(decimals=1))
