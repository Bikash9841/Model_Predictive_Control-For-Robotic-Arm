
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

'''
# copelia sim here-----------------------------
client = RemoteAPIClient()

# all simIK.* type of functions and constants
simIK = client.getObject('simIK')
sim = client.getObject('sim')
simBase = sim.getObject('/base_dyn')
# simTip = sim.getObject('/base_dyn/tip')
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

sim.setJointTargetPosition(joint1, 0)
sim.setJointTargetPosition(joint2, 0)
sim.setJointTargetPosition(joint3, 0)
sim.setJointTargetPosition(joint4, 0)
'''

###############################################################################
#  Define the MPC algorithm parameters
###############################################################################
# prediction horizon
f = 10
# control horizon
v = 5

# new constants here for 3dof arm

m1 = 0.3  # in kg, mass of link1
m2 = 0.3  # in kg, mass of link2
m3 = 0.3
m4 = 0.3
L1 = 0.289  # in m, length of link1
L2 = 0.372  # in m, length of link2
L3 = 0.351
L4 = 0.33
l1 = L1 / 2
l2 = L2 / 2
l3 = L3 / 2
l4 = L4/2

# moment of inertias (kg*m^2)
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

    d11 = Iz1 + Iz2 + Iz3 + Iz4 + m4*pow(cos(theta0), 2)*(L3*sin(theta1 + theta2) + L2*sin(theta1) - L4*cos(theta1 + theta2 + theta3)) ^ 2 + L2 ^ 2*m2*sin(theta1) ^ 2 + m4*sin(theta0) ^ 2*(L3*sin(
        theta1 + theta2) + L2*sin(theta1) - L4*cos(theta1 + theta2 + theta3)) ^ 2
    + m3*cos(theta0) ^ 2*(L3*sin(theta1 + theta2) + L2*sin(theta1)) ^ 2 + \
        m3*sin(theta0) ^ 2*(L3*sin(theta1 + theta2) + L2*sin(theta1)) ^ 2

    d12 = 0
    d13 = 0
    d14 = 0
    d21 = 0
    d22 = Ix3 + Ix4/2 + Iy2 + Iy4/2 + L2 ^ 2*m2 + L2 ^ 2*m3 + L2 ^ 2 * \
        m4 + L3 ^ 2*m3 + L3 ^ 2*m4 + L4 ^ 2*m4 - (Ix4*cos(2*theta0))/2
    + (Iy4*cos(2*theta0))/2 - Ix3*cos(theta0) ^ 2 + \
        Iy3*cos(theta0) ^ 2 + Ix2*sin(theta0) ^ 2
    - Iy2*sin(theta0) ^ 2 + 2*L2*L4*m4 * \
        sin(theta2 + theta3) + 2*L2*L3*m3*cos(theta2)
    + 2*L2*L3*m4*cos(theta2) + 2*L3*L4*m4*sin(theta3)

    d23 = Ix3 + Ix4/2 + Iy4/2 + L3 ^ 2*m3 + L3 ^ 2*m4 + L4 ^ 2 * \
        m4 - (Ix4*cos(2*theta0))/2 + (Iy4*cos(2*theta0))/2
    - Ix3*cos(theta0) ^ 2 + Iy3*cos(theta0) ^ 2 + L2*L4 * \
        m4*sin(theta2 + theta3) + L2*L3*m3*cos(theta2)
    + L2*L3*m4*cos(theta2) + 2*L3*L4*m4*sin(theta3)

    d24 = Ix4/2 + Iy4/2 + L4 ^ 2*m4 - (Ix4*cos(2*theta0))/2
    + (Iy4*cos(2*theta0))/2 + L2*L4*m4 * \
        sin(theta2 + theta3) + L3*L4*m4*sin(theta3)

    d31 = 0
    d32 = Ix3 + Ix4/2 + Iy4/2 + L3 ^ 2*m3 + L3 ^ 2 * \
        m4 + L4 ^ 2*m4 - (Ix4*cos(2*theta0))/2
    + (Iy4*cos(2*theta0))/2 - Ix3*cos(theta0) ^ 2 + Iy3 * \
        cos(theta0) ^ 2 + L2*L4*m4*sin(theta2 + theta3)
    + L2*L3*m3*cos(theta2) + L2*L3*m4*cos(theta2) + 2*L3*L4*m4*sin(theta3)
    d33 = Iy3 + Iy4 + L3 ^ 2*m3 + L3 ^ 2*m4 + L4 ^ 2 * \
        m4 + Ix3*sin(theta0) ^ 2 + Ix4*sin(theta0) ^ 2
    - Iy3*sin(theta0) ^ 2 - Iy4*sin(theta0) ^ 2 + 2*L3*L4*m4*sin(theta3)

    d34 = Iy4 + L4 ^ 2*m4 + Ix4 * \
        sin(theta0) ^ 2 - Iy4*sin(theta0) ^ 2 + L3*L4*m4*sin(theta3)

    d41 = 0
    d42 = Ix4/2 + Iy4/2 + L4 ^ 2*m4 - \
        (Ix4*cos(2*theta0))/2 + (Iy4*cos(2*theta0))/2
    + L2*L4*m4*sin(theta2 + theta3) + L3*L4*m4*sin(theta3)
    d43 = Iy4 + L4 ^ 2*m4 + Ix4 * \
        sin(theta0) ^ 2 - Iy4*sin(theta0) ^ 2 + L3*L4*m4*sin(theta3)
    d44 = Iy4 + L4 ^ 2*m4 + Ix4*sin(theta0) ^ 2 - Iy4*sin(theta0) ^ 2

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

    c11 = theta3_d*((L3 ^ 2*m3*sin(2*theta2 + 2*theta3))/2 + (L3 ^ 2*m4*sin(2*theta2 + 2*theta3))/2
                    - (L4 ^ 2*m4*sin(2*theta2 + 2*theta3 + 2*theta4)) /
                    2 - (L2*L4*m4*cos(2*theta2 + theta3 + theta4))/2
                    + (L2*L4*m4*cos(theta3 + theta4))/2 - L3 *
                    L4*m4*cos(2*theta2 + 2*theta3 + theta4)
                    - (L2*L3*m3*sin(theta3))/2 - (L2*L3*m4*sin(theta3)) /
                    2 + (L2*L3*m3*sin(2*theta2 + theta3))/2
                    + (L2*L3*m4*sin(2*theta2 + theta3))/2) + theta2_d*((L3 ^ 2*m3*sin(2*theta2 + 2*theta3))/2
                                                                       + (L3 ^ 2*m4*sin(2*theta2 + 2*theta3))/2 + (
                                                                           L2 ^ 2*m2*sin(2*theta2))/2 + (L2 ^ 2*m3*sin(2*theta2))/2
                                                                       + (L2 ^ 2*m4*sin(2*theta2))/2 - (L4 ^ 2*m4*sin(2*theta2 + 2 *
                                                                                                                      theta3 + 2*theta4))/2 - L2*L4*m4*cos(2*theta2 + theta3 + theta4)
                                                                       - L3*L4*m4*cos(2*theta2 + 2*theta3 + theta4) + L2*L3*m3*sin(2*theta2 + theta3) + L2*L3*m4*sin(2*theta2 + theta3))
    + L4*m4*theta4_d*sin(theta2 + theta3 + theta4)*(L3*sin(theta2 +
                                                           theta3) + L2*sin(theta2) - L4*cos(theta2 + theta3 + theta4))

    c12 = theta1_d*((L3 ^ 2*m3*sin(2*theta2 + 2*theta3))/2 + (L3 ^ 2*m4*sin(2*theta2 + 2*theta3))/2 + (L2 ^ 2*m2*sin(2*theta2))/2
                    + (L2 ^ 2*m3*sin(2*theta2))/2 + (L2 ^ 2*m4*sin(2*theta2)) /
                    2 - (L4 ^ 2*m4*sin(2*theta2 + 2*theta3 + 2*theta4))/2
                    - L2*L4*m4*cos(2*theta2 + theta3 + theta4) -
                    L3*L4*m4*cos(2*theta2 + 2*theta3 + theta4)
                    + L2*L3*m3*sin(2*theta2 + theta3) + L2*L3*m4*sin(2*theta2 + theta3)) - (theta2_d*sin(2*theta1)*(Ix2 + Ix3 + Ix4 - Iy2 - Iy3 - Iy4))/2
    - (theta3_d*sin(2*theta1)*(Ix3 + Ix4 - Iy3 - Iy4)) / \
        2 - (theta4_d*sin(2*theta1)*(Ix4 - Iy4))/2

    c13 = theta1_d*((L3 ^ 2*m3*sin(2*theta2 + 2*theta3))/2 + (L3 ^ 2*m4*sin(2*theta2 + 2*theta3))/2
                    - (L4 ^ 2*m4*sin(2*theta2 + 2*theta3 + 2*theta4)) /
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

    c21 = (theta2_d*sin(2*theta1)*(Ix2 + Ix3 + Ix4 - Iy2 - Iy3 - Iy4))/2 - theta1_d*((L3 ^ 2*m3*sin(2*theta2 + 2*theta3))/2
                                                                                     + (L3 ^ 2*m4*sin(2*theta2 + 2*theta3))/2
                                                                                     + (L2 ^ 2*m2*sin(2*theta2))/2
                                                                                     + (L2 ^ 2*m3*sin(2*theta2))/2
                                                                                     + (L2 ^ 2*m4*sin(2*theta2))/2
                                                                                     - (L4 ^ 2*m4*sin(2*theta2 + 2*theta3 + 2*theta4))/2
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

    c31 = (theta2_d*sin(2*theta1)*(Ix3 + Ix4 - Iy3 - Iy4))/2 - theta1_d*((L3 ^ 2*m3*sin(2*theta2 + 2*theta3))/2
                                                                         + (L3 ^ 2*m4*sin(2*theta2 + 2*theta3))/2
                                                                         - (L4 ^ 2*m4*sin(2*theta2 + 2*theta3 + 2*theta4))/2
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


# -----------------------------------------------------------------------


# def Gra_mat(theta1, theta2):
#     # gravity vectors for 3DOF robotic arm
#     g1 = 0
#     g2 = m2 * g * l2 * cos(theta1) + m3 * g * L2 * cos(theta1) + \
#         m3 * g * l3 * cos(theta1 + theta2)
#     g3 = m3 * g * l3 * cos(theta1 + theta2)

#     # ----------------------------------------------------------------------------
#     Gra = np.matrix([[g1],
#                      [g2],
#                      [g3]])
#     return Gra


# ----------------------------------------------------------------------------


def Ainitial_mat(theta0, theta1, theta2, theta3, dtheta0, dtheta1, dtheta2, dtheta3):
    Cor = Cor_mat(theta0, theta1, theta2, theta3,
                  dtheta0, dtheta1, dtheta2, dtheta3)
    D = D_mat(theta0, theta1, theta2, theta3)

    # ------------------------------------State Space modelling starts from here---------------------------------
    Ac = np.matrix(np.zeros((6, 6)))
    Ac[:3, 3:] = np.identity(3)

    # this one is for "-inv(M)N" from the paper to insert in lower 3X3 matrix of A
    Ac[3:, 3:] = -(np.linalg.inv(D) * Cor)
    return Ac


def ABC_final(theta0, theta1, theta2, theta3, dtheta0, dtheta1, dtheta2, dtheta3):
    # discretization constant
    sampling = 0.06

    Ac = Ainitial_mat(theta0, theta1, theta2, theta3,
                      dtheta0, dtheta1, dtheta2, dtheta3)

    Bc = np.matrix(np.zeros((6, 3)))
    Bc[3:, :] = np.identity(3)

    Cc = np.matrix(np.zeros((3, 6)))
    Cc[:3, :3] = np.identity(3)

    # model discretization
    I = np.identity(Ac.shape[0])  # this is an identity matrix
    A = np.linalg.inv(I - sampling * Ac)
    B = A * sampling * Bc
    C = Cc

    return A, B, C


def invK(x, y, z, q1, q2, q3):

    # converting origin angles from degress to radian
    q2 = np.deg2rad(q2)
    q3 = np.deg2rad(q3)

    q1 = np.arctan2(y, x)

    ex = L2*np.cos(q2)+L3*np.cos(q2+q3)
    ez = L2*np.sin(q2)+L3*np.sin(q2+q3)

    q3 = np.arccos((pow(ex, 2)+pow(ez, 2)-(pow(L2, 2)+pow(L3, 2)))/2*L2*L3)
    q2 = np.arctan2(ez, ex)-np.arctan2((L3*np.sin(q3)), (L2+L3*np.cos(q3)))
    return np.rad2deg(q1), np.rad2deg(q2), np.rad2deg(q3)


A, B, C = ABC_final(
    theta[0], theta[1], theta[2], dtheta[0], dtheta[1], dtheta[2])

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
Q0 = 0.00000000011
Qother = 0.0001

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
timeSteps = 200

# ------------------simple trajectory--------------------------------
xd, yd, zd = invK(0.4, 0.2, 0, 0, 90, 0)
print(xd, yd, zd)
start_positions = np.matrix([[0], [0], [0]])  # Initial joint angles
# end_positions = np.matrix([[45], [45], [-90]])  # Final joint angles
end_positions = np.matrix([[xd], [yd], [zd]])

# Generate time vector
timeVector = np.linspace(0, 1, timeSteps)

# Generate cubic spline trajectories for each joint
spline_joint0 = CubicSpline(
    [0, 1], [start_positions[0, 0], end_positions[0, 0]])
spline_joint1 = CubicSpline(
    [0, 1], [start_positions[1, 0], end_positions[1, 0]])
spline_joint2 = CubicSpline(
    [0, 1], [start_positions[2, 0], end_positions[2, 0]])

# Evaluate the splines at each time step
trajectory_joint0 = spline_joint0(timeVector)
trajectory_joint1 = spline_joint1(timeVector)
trajectory_joint2 = spline_joint2(timeVector)

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
    (trajectory_joint0, trajectory_joint1, trajectory_joint2)))

# print(desiredTrajectory[:, 0])

# # Plot the cubic spline trajectories
plt.plot(trajectory_joint0, label='Joint 0')
plt.plot(trajectory_joint1, label='Joint 1')
plt.plot(trajectory_joint2, label='Joint 2')
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
                    [dtheta[0]],
                    [dtheta[1]],
                    [dtheta[2]]])
x0 = x0test

# print(x0)
# create the MPC object

mpc = ModelPredictiveControl(A, B, C, f, v, W3, W4, x0, desiredTrajectory)

# mpc.computeControlInputs()


# Simulate the controller
for j in range(timeSteps):
    # for j in range(220):
    if mpc.currentTimeStep:

        # Update weights based on the error (replace with your logic)
        scaling_factor = 0.001  # Adjust this scaling factor as needed
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
controlledTrajectoryList1 = []
controlledTrajectoryList2 = []
controlledTrajectoryList3 = []
controlInputList1 = []
controlInputList2 = []
controlInputList3 = []

# Append values to lists
for i in range(timeSteps):
    controlledTrajectoryList1.append(mpc.outputs[i][0, 0])
    controlledTrajectoryList2.append(mpc.outputs[i][1, 0])
    controlledTrajectoryList3.append(mpc.outputs[i][2, 0])
    controlInputList1.append(mpc.inputs[i][0, 0])
    controlInputList2.append(mpc.inputs[i][1, 0])
    controlInputList3.append(mpc.inputs[i][2, 0])

desiredTrajectoryList1 = desiredTrajectory[:, 0]
desiredTrajectoryList2 = desiredTrajectory[:, 1]
desiredTrajectoryList3 = desiredTrajectory[:, 2]

# -----------------------------starts the arm animation----------------------------------------

total_time = 2  # seconds
time_vector = np.linspace(0, total_time, num=timeSteps)

# Iterate over time steps

theta0_sim = np.zeros_like(time_vector)
theta1_sim = np.zeros_like(time_vector)
theta2_sim = np.zeros_like(time_vector)

# Create a figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)

# Initialize line objects for the links and end effector
link0, = ax.plot([], [], 'o-', lw=2, label='Link 0')
link1, = ax.plot([], [], 'o-', lw=2, label='Link 1')
link2, = ax.plot([], [], 'o-', lw=2, label='Link 2')
end_effector, = ax.plot([], [], 'o', label='End Effector')
trajectory_line, = ax.plot([], [], '-', lw=1, color='gray', label='Trajectory')


# Function to update the animation
def update(frame):

    # Set the desired trajectory for the current time step

    # Extract joint angles

    # Access the elements from mpc.states[frame]
    theta0_sim[frame] = (mpc.states[frame][0, 0])*np.pi/180
    theta1_sim[frame] = ((mpc.states[frame][1, 0])*np.pi/180)  # -(np.pi/2)
    theta2_sim[frame] = (mpc.states[frame][2, 0])*np.pi/180

    # print(frame,theta0_sim[frame],theta1_sim[frame],theta2_sim[frame])

    # Update the positions of the links and end effector
    x0, y0 = 0, 0
    x1 = L1 * np.cos(theta0_sim[frame])
    y1 = L1 * np.sin(theta0_sim[frame])
    x2 = x1+L2 * np.cos(theta0_sim[frame] + theta1_sim[frame])
    y2 = y1+L2 * np.sin(theta0_sim[frame] + theta1_sim[frame])
    x3 = x2+L3 * np.cos(theta0_sim[frame] +
                        theta1_sim[frame] + theta2_sim[frame])
    y3 = y2+L3 * np.sin(theta0_sim[frame] +
                        theta1_sim[frame] + theta2_sim[frame])

    # ----------derived forward kinematics--------------------------
    # x1 = L2 * np.cos(theta1_sim[frame])
    # y1 = L2 * np.sin(theta1_sim[frame])
    # x2 = L3 * np.cos(theta2_sim[frame])
    # y2 = L3 * np.sin(theta2_sim[frame])
    # x3 = L2*np.cos(theta0_sim[frame])*np.cos(theta1_sim[frame]) + L3*np.cos(
    #     theta0_sim[frame])*np.cos(theta1_sim[frame]+theta2_sim[frame])
    # y3 = L2*np.cos(theta1_sim[frame])*np.sin(theta0_sim[frame]) + L3*np.sin(
    #     theta1_sim[frame])*np.cos(theta1_sim[frame]+theta2_sim[frame])

    link0.set_data([x0, x1], [y0, y1])
    link1.set_data([x1, x2], [y1, y2])
    link2.set_data([x2, x3], [y2, y3])
    end_effector.set_data(x3, y3)

    # Initialize end_effector_path as a global variable
    global end_effector_path
    # Update the trajectory line
    end_effector_path.append((x3, y3))
    trajectory_line.set_data(*zip(*end_effector_path))


# Initialize end_effector_path
end_effector_path = []

# Create an animation
# animation = FuncAnimation(fig, update, frames=range(
#     timeSteps), interval=100, blit=False, repeat=False)


# animation.save('robotic_arm_animation.gif', writer='imagemagick')

# Show the plot
# plt.legend()
# plt.show()

fig, axs = plt.subplots(2, 3)
axs[0, 0].plot(desiredTrajectoryList1, linewidth=4,
               label='Desired trajectory1')
axs[0, 0].plot(controlledTrajectoryList1, linewidth=3,
               label='Controlled trajectory1')
axs[0, 0].set_title('JOINT 1')
axs[0, 0].legend()
# -------------------------------------------------------------

axs[0, 1].plot(desiredTrajectoryList2, linewidth=4,
               label='Desired trajectory2')
axs[0, 1].plot(controlledTrajectoryList2, linewidth=3,
               label='Controlled trajectory2')
axs[0, 1].set_title('JOINT 2')
axs[0, 1].legend()
# ---------------------------------------------------------------

axs[0, 2].plot(desiredTrajectoryList3, linewidth=4,
               label='Desired trajectory3')
axs[0, 2].plot(controlledTrajectoryList3, linewidth=3,
               label='Controlled trajectory3')
axs[0, 2].set_title('JOINT 3')
axs[0, 2].legend()
# ---------------------------------------------------------------

axs[1, 0].plot(controlInputList1, linewidth=4,
               label='Control Torque1')
axs[1, 0].set_title('Torque of Joint1')
axs[1, 0].legend()
axs[1, 1].plot(controlInputList2, linewidth=4,
               label='Control Torque2')
axs[1, 1].set_title('Torque of Joint2')
axs[1, 1].legend()
axs[1, 2].plot(controlInputList3, linewidth=4,
               label='Control Torque3')
axs[1, 2].set_title('Torque of Joint3')
axs[1, 2].legend()

# plt.show()


for i in range(timeSteps):

    sim.setJointTargetPosition(joint1, mpc.outputs[i][0, 0]*(np.pi/180))
    sim.setJointTargetPosition(joint2, mpc.outputs[i][1, 0]*(np.pi/180))
    sim.setJointTargetPosition(joint3, mpc.outputs[i][2, 0]*(np.pi/180))
    time.sleep(0.01)


def get_ang():
    j1 = sim.getJointPosition(joint1)
    j2 = sim.getJointPosition(joint2)
    j3 = sim.getJointPosition(joint3)
    # j4 = sim.getJointPosition(joint4)
    print(j1*(180/np.pi), j2*(180/np.pi), j3*(180/np.pi))


get_ang()
