
# =================================================================================
# This one is working inverse kinematics for 4DOF robotic arm with "orientation". Scipy optimization
# ==================================================================================


import numpy as np
from scipy.optimize import minimize
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
from scipy.integrate import odeint
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# copelia sim here-----------------------------
client = RemoteAPIClient()

# all simIK.* type of functions and constants
simIK = client.getObject('simIK')
sim = client.getObject('sim')
simBase = sim.getObject('/base_dyn')
simTip = sim.getObject('/base_dyn/tip')
# simTarget = sim.getObject('/base_dyn/target')
# sphere = sim.getObject('/base_dyn/manipSphere')
g_joint1 = sim.getObjectHandle('/base_dyn/gripper_joint1')
g_joint2 = sim.getObjectHandle('/base_dyn/gripper_joint2')
# cup_0 = sim.getObjectHandle('/Cup[0]/visible_transparent')
# cup_1 = sim.getObjectHandle('/Cup[1]/visible_transparent')
joint1 = sim.getObjectHandle('/base_dyn/joint1')
joint2 = sim.getObjectHandle('/base_dyn/joint2')
joint3 = sim.getObjectHandle('/base_dyn/joint3')
joint4 = sim.getObjectHandle('/base_dyn/joint4')


# to find the homegeneous matrix
simBase = sim.getObject('/base_dyn')
# simTip = sim.getObject('/base_dyn/tip')

sim.setJointTargetPosition(joint1,  np.deg2rad(0))
sim.setJointTargetPosition(joint2,  np.deg2rad(0))
sim.setJointTargetPosition(joint3, np.deg2rad(0))
sim.setJointTargetPosition(joint4,  np.deg2rad(0))
time.sleep(5)

L1 = 0.289  # in m, length of link1
L2 = 0.372  # in m, length of link2
L3 = 0.351
L4 = 0.40

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

    # pitch chai hunu pryo hamro case ma
    e_pitch = np.linalg.norm(pitch-np.deg2rad(5))

    # e_yaw = np.linalg.norm(yaw-np.deg2rad(0))  # jati vaye ni baal hunna

    # roll change hudeina always 90
    e_roll = np.linalg.norm(roll-np.deg2rad(90))

    error_orient = np.linalg.norm(e_pitch)
    # errorList = np.append(errorList, error)
    print(np.rad2deg(yaw), np.rad2deg(pitch), np.rad2deg(roll))
    return 2*error_pos + error_orient

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
    # the rotation matrix given by the simulation is 3-2-1(YPR) matrix. and the
    # calculation of the YPR is also same as 321

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
desired_position = np.array([0.713, 0.110, 0.1])
# cup 0.58, -0.59, 0.06
# nozzle 1, 0.05, 0.1
#  Initial guess for joint angles
initial_guess = np.array([0, 0, 0, 0])

# bounds
b1 = [-180.0, 180.0]
b2 = [-72, 72]
b3 = [-150, 150]
b4 = [-150, 150]
bnds = (b1, b2, b3, b4)

# Perform optimization
result = minimize(objective_function, initial_guess,
                  method='Powell', bounds=bnds, tol=1e-6)

# Extract the optimized joint angles
optimized_joint_angles = np.array(result.x)

# Display the result
print("Optimized Joint Angles:", optimized_joint_angles.round(decimals=1))

timeSteps = 50
# Generate time vector
timeVector = np.linspace(0, 1, timeSteps)

# Generate cubic spline trajectories for each joint
spline_joint0 = CubicSpline(
    [0, 1], [initial_guess[0], optimized_joint_angles[0]])
spline_joint1 = CubicSpline(
    [0, 1], [initial_guess[1], optimized_joint_angles[1]])
spline_joint2 = CubicSpline(
    [0, 1], [initial_guess[2], optimized_joint_angles[2]])
spline_joint3 = CubicSpline(
    [0, 1], [initial_guess[3], optimized_joint_angles[3]])

# Evaluate the splines at each time step
trajectory_joint0 = spline_joint0(timeVector)
trajectory_joint1 = spline_joint1(timeVector)
trajectory_joint2 = spline_joint2(timeVector)
trajectory_joint3 = spline_joint3(timeVector)

# Stack the individual trajectories to form the desired trajectory matrix
desiredJointAngle = np.matrix(np.column_stack(
    (trajectory_joint0, trajectory_joint1, trajectory_joint2, trajectory_joint3)))

# print(desiredJointAngle.shape)

# plt.plot(errorList)
# plt.show()

# for i in range(200):
#     print(desiredJointAngle[i][0, 1])

for i in range(50):
    sim.setJointTargetPosition(joint1, np.deg2rad(desiredJointAngle[i][0, 0]))
    sim.setJointTargetPosition(joint2, np.deg2rad(desiredJointAngle[i][0, 1]))
    sim.setJointTargetPosition(joint3, np.deg2rad(desiredJointAngle[i][0, 2]))
    sim.setJointTargetPosition(joint4, np.deg2rad(desiredJointAngle[i][0, 3]))


def get_ang():
    j1 = sim.getJointPosition(joint1)
    j2 = sim.getJointPosition(joint2)
    j3 = sim.getJointPosition(joint3)
    j4 = sim.getJointPosition(joint4)
    print(j1*(180/np.pi), j2*(180/np.pi), j3*(180/np.pi), j4*(180/np.pi))


# get_ang()
print(sim.getObjectMatrix(simTip, simBase))
print(" ")
a, b, g = sim.getObjectOrientation(simTip, simBase)
print(sim.alphaBetaGammaToYawPitchRoll(a, b, g))
