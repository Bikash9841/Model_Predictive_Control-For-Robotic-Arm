
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt
from math import *
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from functionMPC import systemSimulate
from ModelPredictiveControl_4 import ModelPredictiveControl
import time
from scipy.integrate import odeint
from InvKine_GradientDescentOpt import invOpt
from plots import visualize


###############################################################################
#  Define the MPC algorithm parameters
###############################################################################
# prediction horizon
f = 10
# control horizon
v = 10

# new constants here for 3dof arm

m1 = 0.6  # in kg, mass of link1
m2 = 0.6  # in kg, mass of link2
m3 = 0.6
m4 = 0.6
# L1 = 0.17  # in m, length of link1
# L2 = 0.20  # in m, length of link2
# L3 = 0.20
# L4 = 0.20

L1 = 0.289  # in m, length of link1
L2 = 0.372  # in m, length of link2
L3 = 0.351
L4 = 0.33

l1 = L1 / 2
l2 = L2 / 2
l3 = L3 / 2
l4 = L4/2

# moment of inertias (kg*m**2)
Iz1 = 0.00578
Iz2 = 0.00578
Iz3 = 0.00578
Iz4 = 0.00578

Ix1 = 0.00355
Ix2 = 0.00355
Ix3 = 0.00355
Ix4 = 0.00355

Iy1 = 0.00883
Iy2 = 0.00883
Iy3 = 0.00883
Iy4 = 0.00883

g = -9.8065  # m/s**2, gravity

theta = np.array([0, 0, 0, 0])
dtheta = np.array([0, 0, 0, 0])


# D matrix components


def D_mat(theta0, theta1, theta2, theta3):

    # theta starts from 0 in this matrix

    d11 = Iz1 + Iz2 + Iz3 + Iz4 + m4*pow(cos(theta0), 2)*(L3*sin(
        theta1 + theta2) + L2*sin(theta1) - L4*cos(theta1 + theta2 + theta3)) ** 2
    + L2 ** 2*m2*sin(theta1) ** 2 + m4*sin(theta0) ** 2*(L3*sin(theta1 +
                                                                theta2) + L2*sin(theta1) - L4*cos(theta1 + theta2 + theta3)) ** 2
    + m3*cos(theta0) ** 2*(L3*sin(theta1 + theta2) + L2*sin(theta1)) ** 2 + \
        m3*sin(theta0) ** 2*(L3*sin(theta1 + theta2) + L2*sin(theta1)) ** 2

    d12 = 0
    d13 = 0
    d14 = 0
    d21 = 0

    d22 = Ix3 + Ix4/2 + Iy2 + Iy4/2 + L2 ** 2*m2 + L2 ** 2*m3 + L2 ** 2 * \
        m4 + L3 ** 2*m3 + L3 ** 2*m4 + L4 ** 2*m4 - (Ix4*cos(2*theta0))/2
    + (Iy4*cos(2*theta0))/2 - Ix3*cos(theta0) ** 2 + Iy3*cos(theta0) ** 2 + Ix2 * \
        sin(theta0) ** 2 - Iy2*sin(theta0) ** 2 + \
        2*L2*L4*m4 * sin(theta2 + theta3)
    + 2*L2*L3*m3*cos(theta2) + 2*L2*L3*m4*cos(theta2) + 2*L3*L4*m4*sin(theta3)

    d23 = Ix3 + Ix4/2 + Iy4/2 + L3 ** 2*m3 + L3 ** 2*m4 + L4 ** 2 * m4 - \
        (Ix4*cos(2*theta0))/2 + (Iy4*cos(2*theta0)) / \
        2 - Ix3*cos(theta0) ** 2 + Iy3*cos(theta0) ** 2
    + L2*L4 * m4*sin(theta2 + theta3) + L2*L3*m3*cos(theta2) + \
        L2*L3*m4*cos(theta2) + 2*L3*L4*m4*sin(theta3)

    d24 = Ix4/2 + Iy4/2 + L4 ** 2*m4 - (Ix4*cos(2*theta0))/2 + (
        Iy4*cos(2*theta0))/2 + L2*L4*m4 * sin(theta2 + theta3) + L3*L4*m4*sin(theta3)

    d31 = 0
    d32 = Ix3 + Ix4/2 + Iy4/2 + L3 ** 2*m3 + L3 ** 2 * m4 + L4 ** 2*m4 - (Ix4*cos(2*theta0))/2 + (Iy4*cos(2*theta0))/2 - Ix3*cos(theta0) ** 2 + Iy3 * \
        cos(theta0) ** 2 + L2*L4*m4*sin(theta2 + theta3) + L2*L3*m3 * \
        cos(theta2) + L2*L3*m4*cos(theta2) + 2*L3*L4*m4*sin(theta3)

    d33 = Iy3 + Iy4 + L3 ** 2*m3 + L3 ** 2*m4 + L4 ** 2 * m4 + Ix3 * \
        sin(theta0) ** 2 + Ix4*sin(theta0) ** 2 - \
        Iy3*sin(theta0) ** 2 - Iy4*sin(theta0) ** 2
    + 2*L3*L4*m4*sin(theta3)

    d34 = Iy4 + L4 ** 2*m4 + Ix4 * \
        sin(theta0) ** 2 - Iy4*sin(theta0) ** 2 + L3*L4*m4*sin(theta3)

    d41 = 0
    d42 = Ix4/2 + Iy4/2 + L4 ** 2*m4 - (Ix4*cos(2*theta0))/2 + (
        Iy4*cos(2*theta0))/2 + L2*L4*m4*sin(theta2 + theta3) + L3*L4*m4*sin(theta3)
    d43 = Iy4 + L4 ** 2*m4 + Ix4 * \
        sin(theta0) ** 2 - Iy4*sin(theta0) ** 2 + L3*L4*m4*sin(theta3)
    d44 = Iy4 + L4 ** 2*m4 + Ix4*sin(theta0) ** 2 - Iy4*sin(theta0) ** 2

    # ------------------------------------------------------------------------------------

    D = np.matrix([[d11, d12, d13, d14],
                   [d21, d22, d23, d24],
                   [d31, d32, d33, d34],
                   [d41, d42, d43, d44]])
    return D


# ------------------------------------------------------------------------------------


def Cor_mat(theta1, theta2, theta3, theta4, theta1_d, theta2_d, theta3_d, theta4_d):

    # theta starts from 1 in this matrix...
    # Coriolis matrix components

    c11 = theta3_d*((L3 ** 2*m3*sin(2*theta2 + 2*theta3))/2 + (L3 ** 2*m4*sin(2*theta2 + 2*theta3))/2
                    - (L4 ** 2*m4*sin(2*theta2 + 2*theta3 + 2*theta4)) /
                    2 - (L2*L4*m4*cos(2*theta2 + theta3 + theta4))/2
                    + (L2*L4*m4*cos(theta3 + theta4))/2 - L3 *
                    L4*m4*cos(2*theta2 + 2*theta3 + theta4)
                    - (L2*L3*m3*sin(theta3))/2 - (L2*L3*m4*sin(theta3)) /
                    2 + (L2*L3*m3*sin(2*theta2 + theta3))/2
                    + (L2*L3*m4*sin(2*theta2 + theta3))/2) + theta2_d*((L3 ** 2*m3*sin(2*theta2 + 2*theta3))/2
                                                                       + (L3 ** 2*m4*sin(2*theta2 + 2*theta3))/2 + (
                                                                           L2 ** 2*m2*sin(2*theta2))/2 + (L2 ** 2*m3*sin(2*theta2))/2
                                                                       + (L2 ** 2*m4*sin(2*theta2))/2 - (L4 ** 2*m4*sin(2*theta2 + 2 *
                                                                                                                        theta3 + 2*theta4))/2 - L2*L4*m4*cos(2*theta2 + theta3 + theta4)
                                                                       - L3*L4*m4*cos(2*theta2 + 2*theta3 + theta4) + L2*L3*m3*sin(2*theta2 + theta3) + L2*L3*m4*sin(2*theta2 + theta3))
    + L4*m4*theta4_d*sin(theta2 + theta3 + theta4)*(L3*sin(theta2 +
                                                           theta3) + L2*sin(theta2) - L4*cos(theta2 + theta3 + theta4))

    c12 = theta1_d*((L3 ** 2*m3*sin(2*theta2 + 2*theta3))/2 + (L3 ** 2*m4*sin(2*theta2 + 2*theta3))/2 + (L2 ** 2*m2*sin(2*theta2))/2
                    + (L2 ** 2*m3*sin(2*theta2))/2 + (L2 ** 2*m4*sin(2*theta2)) /
                    2 - (L4 ** 2*m4*sin(2*theta2 + 2*theta3 + 2*theta4))/2
                    - L2*L4*m4*cos(2*theta2 + theta3 + theta4) -
                    L3*L4*m4*cos(2*theta2 + 2*theta3 + theta4)
                    + L2*L3*m3*sin(2*theta2 + theta3) + L2*L3*m4*sin(2*theta2 + theta3)) - (theta2_d*sin(2*theta1)*(Ix2 + Ix3 + Ix4 - Iy2 - Iy3 - Iy4))/2
    - (theta3_d*sin(2*theta1)*(Ix3 + Ix4 - Iy3 - Iy4)) / \
        2 - (theta4_d*sin(2*theta1)*(Ix4 - Iy4))/2

    c13 = theta1_d*((L3 ** 2*m3*sin(2*theta2 + 2*theta3))/2 + (L3 ** 2*m4*sin(2*theta2 + 2*theta3))/2
                    - (L4 ** 2*m4*sin(2*theta2 + 2*theta3 + 2*theta4)) /
                    2 - (L2*L4*m4*cos(2*theta2 + theta3 + theta4))/2
                    + (L2*L4*m4*cos(theta3 + theta4))/2 - L3 *
                    L4*m4*cos(2*theta2 + 2*theta3 + theta4)
                    - (L2*L3*m3*sin(theta3))/2 - (L2*L3*m4*sin(theta3)) /
                    2 + (L2*L3*m3*sin(2*theta2 + theta3))/2
                    + (L2*L3*m4*sin(2*theta2 + theta3))/2) - (theta2_d*sin(2*theta1)*(Ix3 + Ix4 - Iy3 - Iy4))/2
    - (theta3_d*sin(2*theta1)*(Ix3 + Ix4 - Iy3 - Iy4)) / \
        2 - (theta4_d*sin(2*theta1)*(Ix4 - Iy4))/2

    c14 = L4*m4*theta1_d*sin(theta2 + theta3 + theta4)*(
        L3*sin(theta2 + theta3) + L2*sin(theta2) - L4*cos(theta2 + theta3 + theta4))
    - (theta3_d*sin(2*theta1)*(Ix4 - Iy4))/2 - (theta4_d*sin(2*theta1)
                                                * (Ix4 - Iy4))/2 - (theta2_d*sin(2*theta1)*(Ix4 - Iy4))/2

    c21 = (theta2_d*sin(2*theta1)*(Ix2 + Ix3 + Ix4 - Iy2 - Iy3 - Iy4))/2 - theta1_d*((L3 ** 2*m3*sin(2*theta2 + 2*theta3))/2
                                                                                     + (L3 ** 2*m4*sin(2*theta2 + 2*theta3))/2
                                                                                     + (L2 ** 2*m2*sin(2*theta2))/2
                                                                                     + (L2 ** 2*m3*sin(2*theta2))/2
                                                                                     + (L2 ** 2*m4*sin(2*theta2))/2
                                                                                     - (L4 ** 2*m4*sin(2*theta2 + 2*theta3 + 2*theta4))/2
                                                                                     - L2*L4*m4 *
                                                                                     cos(
                                                                                         2*theta2 + theta3 + theta4)
                                                                                     - L3*L4*m4 *
                                                                                     cos(
                                                                                         2*theta2 + 2*theta3 + theta4)
                                                                                     + L2*L3*m3*sin(2*theta2 + theta3) + L2*L3*m4*sin(2*theta2 + theta3))
    + (theta3_d*sin(2*theta1)*(Ix3 + Ix4 - Iy3 - Iy4)) / \
        2 + (theta4_d*sin(2*theta1)*(Ix4 - Iy4))/2

    c22 = (theta1_d*sin(2*theta1)*(Ix2 + Ix3 + Ix4 - Iy2 - Iy3 - Iy4))/2 - L2*theta3_d*(L3*m3*sin(theta3) + L3*m4*sin(theta3)
                                                                                        - L4*m4*cos(theta3 + theta4))
    + L4*m4*theta4_d*(L2*cos(theta3 + theta4) + L3*cos(theta4))

    c23 = (theta1_d*sin(2*theta1)*(Ix3 + Ix4 - Iy3 - Iy4))/2 - L2*theta2_d*(L3*m3*sin(theta3) + L3*m4*sin(theta3)
                                                                            - L4*m4*cos(theta3 + theta4)) - L2*theta3_d*(L3*m3*sin(theta3) + L3*m4*sin(theta3)
                                                                                                                         - L4*m4*cos(theta3 + theta4))
    + L4*m4*theta4_d*(L2*cos(theta3 + theta4) + L3*cos(theta4))

    c24 = (theta1_d*sin(2*theta1)*(Ix4 - Iy4))/2 + L4*m4*theta2_d*(L2*cos(theta3 + theta4)
                                                                   + L3*cos(theta4)) + L4*m4*theta3_d*(L2*cos(theta3 + theta4)
                                                                                                       + L3*cos(theta4)) + L4*m4*theta4_d*(L2*cos(theta3 + theta4)
                                                                                                                                           + L3*cos(theta4))

    c31 = (theta2_d*sin(2*theta1)*(Ix3 + Ix4 - Iy3 - Iy4))/2 - theta1_d*((L3 ** 2*m3*sin(2*theta2 + 2*theta3))/2
                                                                         + (L3 ** 2*m4*sin(2*theta2 + 2*theta3))/2
                                                                         - (L4 ** 2*m4*sin(2*theta2 + 2*theta3 + 2*theta4))/2
                                                                         - (L2*L4*m4*cos(2*theta2 + theta3 + theta4))/2
                                                                         + (L2*L4*m4*cos(theta3 + theta4))/2
                                                                         - L3*L4*m4 *
                                                                         cos(2*theta2 + 2 *
                                                                             theta3 + theta4)
                                                                         - (L2*L3*m3*sin(theta3))/2
                                                                         - (L2*L3*m4*sin(theta3))/2 + (L2*L3*m3*sin(2*theta2 + theta3))/2
                                                                         + (L2*L3*m4*sin(2*theta2 + theta3))/2) + (theta3_d*sin(2*theta1)*(Ix3 + Ix4 - Iy3 - Iy4))/2 + (theta4_d*sin(2*theta1)*(Ix4 - Iy4))/2

    c32 = (theta1_d*sin(2*theta1)*(Ix3 + Ix4 - Iy3 - Iy4))/2 + L2*theta2_d*(L3*m3*sin(theta3) + L3*m4*sin(theta3)
                                                                            - L4*m4*cos(theta3 + theta4)) + L3*L4*m4*theta4_d*cos(theta4)

    c33 = (theta1_d*sin(2*theta1)*(Ix3 + Ix4 - Iy3 - Iy4)) / \
        2 + L3*L4*m4*theta4_d*cos(theta4)

    c34 = (theta1_d*sin(2*theta1)*(Ix4 - Iy4))/2 + L3*L4*m4*theta2_d * \
        cos(theta4) + L3*L4*m4*theta3_d * \
        cos(theta4) + L3*L4*m4*theta4_d*cos(theta4)

    c41 = (theta2_d*sin(2*theta1)*(Ix4 - Iy4))/2 + \
        (theta3_d*sin(2*theta1)*(Ix4 - Iy4))/2
    + (theta4_d*sin(2*theta1)*(Ix4 - Iy4))/2 - L4*m4*theta1_d*sin(theta2 + theta3 + theta4)*(L3*sin(theta2 + theta3)
                                                                                             + L2 *
                                                                                             sin(
                                                                                                 theta2)
                                                                                             - L4*cos(theta2 + theta3 + theta4))

    c42 = (theta1_d*sin(2*theta1)*(Ix4 - Iy4))/2 - L4*m4*theta2_d*(L2 *
                                                                   cos(theta3 + theta4) + L3*cos(theta4)) - L3*L4*m4*theta3_d*cos(theta4)

    c43 = (theta1_d*sin(2*theta1)*(Ix4 - Iy4))/2 - L3*L4*m4 * \
        theta2_d*cos(theta4) - L3*L4*m4*theta3_d*cos(theta4)

    c44 = (theta1_d*sin(2*theta1)*(Ix4 - Iy4))/2
    # ---------------------------------------------------------------------

    Cor = np.matrix([[c11, c12, c13, c14],
                     [c21, c22, c23, c24],
                     [c31, c32, c33, c34],
                     [c41, c42, c43, c44]])
    return Cor


# ----------------------------------------------------------------------------


def Ainitial_mat(theta0, theta1, theta2, theta3, dtheta0, dtheta1, dtheta2, dtheta3):
    Cor = Cor_mat(theta0, theta1, theta2, theta3,
                  dtheta0, dtheta1, dtheta2, dtheta3)
    D = D_mat(theta0, theta1, theta2, theta3)

    # ------------------------------------State Space modelling starts from here---------------------------------
    Ac = np.matrix(np.zeros((8, 8)))
    Ac[:4, 4:] = np.identity(4)

    # this one is for "-inv(M)N" from the paper to insert in lower 3X3 matrix of A
    Ac[4:, 4:] = -(np.linalg.inv(D) * Cor)
    return Ac


def ABC_final(theta0, theta1, theta2, theta3, dtheta0, dtheta1, dtheta2, dtheta3):
    # discretization constant
    sampling = 0.06

    Ac = Ainitial_mat(theta0, theta1, theta2, theta3,
                      dtheta0, dtheta1, dtheta2, dtheta3)

    Bc = np.matrix(np.zeros((8, 4)))
    Bc[4:, :] = np.identity(4)

    Cc = np.matrix(np.zeros((4, 8)))
    Cc[:4, :4] = np.identity(4)

    # model discretization
    I = np.identity(Ac.shape[0])  # this is an identity matrix
    A = np.linalg.inv(I - sampling * Ac)
    B = A * sampling * Bc
    C = Cc

    return A, B, C


A, B, C = ABC_final(
    theta[0], theta[1], theta[2], theta[3], dtheta[0], dtheta[1], dtheta[2], dtheta[3])

r = C.shape[0]  # no of outputs
m = B.shape[1]  # number of inputs
n = A.shape[0]  # state dimension

# check the eigenvalues
# eigen_A = np.linalg.eig(Ac)[0]
# eigen_Aid = np.linalg.eig(A)[0]


# W1 matrix
W1 = np.zeros(shape=(v * m, v * m))

for i in range(v):
    if (i == 0):
        W1[i * m:(i + 1) * m, i * m:(i + 1) * m] = np.eye(m, m)
    else:
        W1[i * m:(i + 1) * m, i * m:(i + 1) * m] = np.eye(m, m)
        W1[i * m:(i + 1) * m, (i - 1) * m:(i) * m] = -np.eye(m, m)

# W2 matrix
# Q0 = 0.0000000011
# Qother = 0.0001
Q0 = 0.001
Qother = 0.001

W2 = np.zeros(shape=(v * m, v * m))

for i in range(v):
    if (i == 0):
        W2[i * m:(i + 1) * m, i * m:(i + 1) * m] = Q0
    else:
        W2[i * m:(i + 1) * m, i * m:(i + 1) * m] = Qother

# W3 matrix
W3 = np.matmul(W1.T, np.matmul(W2, W1))

# W4 matrix
W4 = np.zeros(shape=(f * r, f * r))

# in the general case, this constant should be a matrix
# predWeight = 10
predWeight = 10

for i in range(f):
    W4[i * r:(i + 1) * r, i * r:(i + 1) * r] = predWeight


# desired trajectory generation
timeSteps = 250

# ------------------simple trajectory--------------------------------
optimizedJointAngles, errorList = invOpt(
    0, 0, 0, 0, 0.3169, 0.3701, 0.07)


plt.plot(errorList, linewidth=4, label='Pose Error')
plt.xlabel('iterations')
plt.ylabel('Error magnitude')
plt.legend()
plt.show()

# print(f"required joint angles: {optimizedJointAngles}")
# print(optimizedJointAngles[0, 0],
#       optimizedJointAngles[1, 0], optimizedJointAngles[2, 0], optimizedJointAngles[3, 0])


start_positions = np.matrix(
    [[0], [0], [0], [0]])  # Initial joint angles
end_positions = np.matrix([[optimizedJointAngles[0, 0]],
                           [optimizedJointAngles[1, 0]],
                           [optimizedJointAngles[2, 0]],
                           [optimizedJointAngles[3, 0]]])

# Generate time vector
timeVector = np.linspace(0, 1, timeSteps)

# Generate cubic spline trajectories for each joint
spline_joint0 = CubicSpline(
    [0, 1], [start_positions[0, 0], end_positions[0, 0]])
spline_joint1 = CubicSpline(
    [0, 1], [start_positions[1, 0], end_positions[1, 0]])
spline_joint2 = CubicSpline(
    [0, 1], [start_positions[2, 0], end_positions[2, 0]])
spline_joint3 = CubicSpline(
    [0, 1], [start_positions[3, 0], end_positions[3, 0]])

# Evaluate the splines at each time step
trajectory_joint0 = spline_joint0(timeVector)
trajectory_joint1 = spline_joint1(timeVector)
trajectory_joint2 = spline_joint2(timeVector)
trajectory_joint3 = spline_joint3(timeVector)

'''
# -------------------------------------sinusoidal trajectory generation----------------------
# Time vector
# timeVector = np.linspace(0, 2 * np.pi, timeSteps)

# # Amplitudes and frequencies for each joint
# amplitudes = [90, 20, 15]  # Adjust as needed
# frequencies = [0.1, 0.2, 0.3]  # Adjust as needed

# # Generate sinusoidal trajectories for each joint
# trajectory_joint0 = amplitudes[0] * \
#     np.cos(2*np.pi*frequencies[0] * timeVector/2*np.pi)
# trajectory_joint1 = amplitudes[1] * \
#     np.sin(2*np.pi*frequencies[1] * timeVector/2*np.pi)
# trajectory_joint2 = amplitudes[2] * \
#     np.sin(2*np.pi*frequencies[2] * timeVector/2*np.pi)

'''
# Stack the individual trajectories to form the desired trajectory matrix
desiredTrajectory = np.matrix(np.column_stack(
    (trajectory_joint0, trajectory_joint1, trajectory_joint2, trajectory_joint3)))

# print(desiredTrajectory[:, 0])

# # Plot the cubic spline trajectories
plt.plot(trajectory_joint0, linewidth=4, label='Joint 1')
plt.plot(trajectory_joint1, linewidth=4, label='Joint 2')
plt.plot(trajectory_joint2, linewidth=4, label='Joint 3')
plt.plot(trajectory_joint3, linewidth=4, label='Joint 4')
plt.xlabel('TimeSteps')
plt.ylabel('Joint Angles')
plt.legend()
plt.show()

###############################################################################
# end of definition of the reference trajectory
###############################################################################

###############################################################################
# Simulate the MPC algorithm and plot the results
###############################################################################

# set the initial state
x0test = np.matrix([[theta[0]],
                    [theta[1]],
                    [theta[2]],
                    [theta[3]],
                    [dtheta[0]],
                    [dtheta[1]],
                    [dtheta[2]],
                    [dtheta[3]]])
x0 = x0test

# print(x0)
# create the MPC object

mpc = ModelPredictiveControl(A, B, C, f, v, W3, W4, x0, desiredTrajectory)

# mpc.computeControlInputs()
# mpc.computeControlInputs()
# mpc.computeControlInputs()

# Simulate the controller
for j in range(timeSteps):
    # for j in range(220):
    if mpc.currentTimeStep:

        # Update weights based on the error (replace with your logic)
        scaling_factor = 0.2  # Adjust this scaling factor as needed
        # new_W4 = W4 + scaling_factor * error * np.eye(f * r)
        new_W4 = mpc.W4 + scaling_factor * np.eye(f * r)

        # Update weights in the MPC object
        mpc.update_weights(W3, new_W4)

    mpc.computeControlInputs()
    # if (timeSteps > 150):
    #     print(
    #         f"theta1: {mpc.states[j][0]}         veloctiy: {mpc.states[j][3]}       input1:{mpc.inputs[j][0]}")


# extract the state estimates in order to plot the results
desiredTrajectoryList1 = []
desiredTrajectoryList2 = []
desiredTrajectoryList3 = []
desiredTrajectoryList4 = []
controlledTrajectoryList1 = []
controlledTrajectoryList2 = []
controlledTrajectoryList3 = []
controlledTrajectoryList4 = []
controlInputList1 = []
controlInputList2 = []
controlInputList3 = []
controlInputList4 = []

# Append values to lists
for i in range(timeSteps):
    controlledTrajectoryList1.append(mpc.outputs[i][0, 0])
    controlledTrajectoryList2.append(mpc.outputs[i][1, 0])
    controlledTrajectoryList3.append(mpc.outputs[i][2, 0])
    controlledTrajectoryList4.append(mpc.outputs[i][3, 0])
    controlInputList1.append(mpc.inputs[i][0, 0])
    controlInputList2.append(mpc.inputs[i][1, 0])
    controlInputList3.append(mpc.inputs[i][2, 0])
    controlInputList4.append(mpc.inputs[i][3, 0])

desiredTrajectoryList1 = desiredTrajectory[:, 0]
desiredTrajectoryList2 = desiredTrajectory[:, 1]
desiredTrajectoryList3 = desiredTrajectory[:, 2]
desiredTrajectoryList4 = desiredTrajectory[:, 3]

# results visualization

visualize(desiredTrajectoryList1, controlledTrajectoryList1, desiredTrajectoryList2, controlledTrajectoryList2,
          desiredTrajectoryList3, controlledTrajectoryList3, desiredTrajectoryList4, controlledTrajectoryList4,
          controlInputList1, controlInputList2, controlInputList3, controlInputList4)

# print(len(controlledTrajectoryList1))

future1 = []
future2 = []
future3 = []
future4 = []
print(mpc.currentTimeStep)

for i in range((timeSteps*10)):
    future1.append(mpc.phout[i][0, 0])
    future2.append(mpc.phout[i][1, 0])
    future3.append(mpc.phout[i][2, 0])
    future4.append(mpc.phout[i][3, 0])


# this one to try about

# future1 = controlledTrajectoryList1.copy()
# for i in range(10):
#     print(f'previous: {future1[95+i]}')
#     future1[95+i] = (mpc.phout[950+i][0, 0])
#     print(f'predicted: {mpc.phout[950+i][0, 0]}')
#     print(f'assignged: {future1[95+i]}')
#     future2.append(mpc.phout[i][1, 0])
#     future3.append(mpc.phout[i][2, 0])
#     future4.append(mpc.phout[i][3, 0])


print(len(mpc.phout))
print("thoss")
# print(future1)

fig, axs = plt.subplots(1, 1, figsize=(8, 6))
axs.plot(controlledTrajectoryList1[0:10], linewidth=4, color='black',
         label='controlled trajectory1')
axs.plot(future1[0:10], linewidth=3, linestyle='dashed', color='black',
         label='predicted trajectory')

# axs.plot(controlledTrajectoryList2, linewidth=4, color='blue',
#          label='controlled trajectory2')
# axs.plot(future2, linewidth=3, color='blue',
#          label='future trajectory2')

# axs.plot(controlledTrajectoryList3, linewidth=4, color='yellow',
#          label='controlled trajectory3')
# axs.plot(future3, linewidth=3, color='yellow',
#          label='future trajectory3')

# axs.plot(future4[40:50], linewidth=3, color='green',
#          label='future trajectory4')
# axs.plot(controlledTrajectoryList4[40:51], linewidth=3, color='green',
#          label='controlled trajectory4')
axs.set_title('JOINT 1')
axs.set_ylabel('Joint Angles (in deg)')
axs.set_xlabel('timesteps')
axs.legend()

plt.show()


# print(mpc.phout.shape)
