import numpy as np
from math import *


# new constants here for 3dof arm

m1 = 0.3        # in kg, mass of link1
m2 = 0.3        # in kg, mass of link2
m3 = 0.3
L1 = 0.4         # in m, length of link1
L2 = 0.4         # in m, length of link2
L3 = 0.4
l1 = L1/2
l2 = L2/2
l3 = L3/2

# moment of inertias (kg*m^2)
Iz1 = 0.00578
Iz2 = 0.00578
Iz3 = 0.00578
Ix1 = 0.00355
Ix2 = 0.00355
Ix3 = 0.00355
Iy1 = 0.00883
Iy2 = 0.00883
Iy3 = 0.00883

g = 9.8065     # m/s**2, gravity


# define the continuous-time system matrices
# Ac = np.matrix([[0, 1, 0, 0],
#                 [-(k1+k2)/m1,  -(d1+d2)/m1, k2/m1, d2/m1],
#                 [0, 0,  0, 1],
#                 [k2/m2,  d2/m2, -k2/m2, -d2/m2]])
# Bc = np.matrix([[0], [0], [0], [1/m2]])
# Cc = np.matrix([[1, 0, 0, 0]])

# -------------------------------------------------------------------------------------------------------------
# new matrix here
theta = np.array([0, 0, 0])
dtheta = np.array([0, 0, 0])

# D matrix components

d11 = Iz1 + Iz2 + Iz3 + L2**2*m2*cos(theta[1]) ** 2 + m3*cos(theta[0])**2*(L3*cos(theta[1] + theta[2]) + L2*cos(
    theta[1])) ** 2 + m3*sin(theta[0])**2*(L3*cos(theta[1] + theta[2]) + L2*cos(theta[1])) ** 2
d12 = 0
d13 = 0
d21 = 0
d22 = Ix2 + Ix3 + L2 ** 2*m2 + L2 ** 2*m3 + L3 ** 2*m3 - Ix2*cos(theta[0]) ** 2 - Ix3*cos(
    theta[0]) ** 2 + Iy2*cos(theta[0]) ** 2 + Iy3*cos(theta[0]) ** 2 + 2*L2*L3*m3*cos(theta[2])

d23 = Ix3 + L3 ** 2*m3 - Ix3 * \
    cos(theta[0]) ** 2 + Iy3*cos(theta[0]) ** 2 + L2*L3*m3*cos(theta[2])

d31 = 0
d32 = Ix3 + L3 ** 2*m3 - Ix3 * \
    cos(theta[0]) ** 2 + Iy3*cos(theta[0]) ** 2 + L2*L3*m3*cos(theta[2])
d33 = Iy3 + L3 ** 2*m3 + Ix3*sin(theta[0]) ** 2 - Iy3*sin(theta[0]) ** 2

# ------------------------------------------------------------------------------------

D = np.matrix([[d11, d12, d13],
               [d21, d22, d23],
               [d31, d32, d33]])

# ------------------------------------------------------------------------------------


# Coriolis matrix components

c11 = - dtheta[1]*((L3 ** 2*m3*sin(2*theta[1] + 2*theta[2]))/2 + (L2 ** 2*m2*sin(2*theta[1]))/2 + (L2 ** 2*m3*sin(2*theta[1]))/2 + L2 *
                   L3*m3*sin(2*theta[1] + theta[2])) - (L3*m3*dtheta[2]*(L3*sin(2*theta[1] + 2*theta[2]) + L2*sin(theta[2]) + L2*sin(2*theta[1] + theta[2])))/2

c12 = - dtheta[0]*((L3 ** 2*m3*sin(2*theta[1] + 2*theta[2]))/2 + (L2 ** 2*m2*sin(2*theta[1]))/2 + (L2 ** 2*m3*sin(2*theta[1]))/2 + L2 *
                   L3*m3*sin(2*theta[1] + theta[2])) - (dtheta[1]*sin(2*theta[0])*(Ix2 + Ix3 - Iy2 - Iy3))/2 - (dtheta[2]*sin(2*theta[0])*(Ix3 - Iy3))/2

c13 = - (dtheta[1]*sin(2*theta[0])*(Ix3 - Iy3))/2 - (dtheta[2]*sin(2*theta[0])*(Ix3 - Iy3))/2 - (L3*m3*dtheta[0]*(L3*sin(2*theta[1] + 2*theta[2]) +
                                                                                                                  L2*sin(theta[2]) + L2*sin(2*theta[1] + theta[2])))/2

c21 = dtheta[0]*((L3 ** 2*m3*sin(2*theta[1] + 2*theta[2]))/2 + (L2 ** 2*m2*sin(2*theta[1]))/2 + (L2 ** 2*m3*sin(2*theta[1]))/2 + L2 *
                 L3*m3*sin(2*theta[1] + theta[2])) + (dtheta[1]*sin(2*theta[0])*(Ix2 + Ix3 - Iy2 - Iy3))/2 + (dtheta[2]*sin(2*theta[0])*(Ix3 - Iy3))/2

c22 = (dtheta[0]*sin(2*theta[0])*(Ix2 + Ix3 - Iy2 - Iy3)) / \
    2 - L2*L3*m3*dtheta[2]*sin(theta[2])
c23 = (dtheta[0]*sin(2*theta[0])*(Ix3 - Iy3))/2 - L2*L3*m3 * \
    dtheta[1]*sin(theta[2]) - L2*L3*m3*dtheta[2]*sin(theta[2])

c31 = (dtheta[1]*sin(2*theta[0])*(Ix3 - Iy3))/2 + (dtheta[2]*sin(2*theta[0])*(Ix3 - Iy3))/2 + (L3*m3*dtheta[0]*(L3*sin(2*theta[1] + 2*theta[2]) +
                                                                                                                L2*sin(theta[2]) + L2*sin(2*theta[1] + theta[2])))/2

c32 = (dtheta[0]*sin(2*theta[0])*(Ix3 - Iy3)) / \
    2 + L2*L3*m3*dtheta[1]*sin(theta[2])
c33 = (dtheta[0]*sin(2*theta[0])*(Ix3 - Iy3))/2


# ---------------------------------------------------------------------

Cor = np.matrix([[c11, c12, c13],
                 [c21, c22, c23],
                 [c31, c32, c33]])

# -----------------------------------------------------------------------

# gravity vectors for 3DOF robotic arm
g1 = 0
g2 = m2*g*l2*cos(theta[1])+m3*g*L2*cos(theta[1]) + \
    m3*g*l3*cos(theta[1]+theta[2])
g3 = m3*g*l3*cos(theta[1]+theta[2])

# ----------------------------------------------------------------------------
Gra = np.matrix([[g1],
                 [g2],
                 [g3]])
# ----------------------------------------------------------------------------

Ac = np.matrix(np.zeros((6, 6)))


Ac[:3, 3:] = np.identity(3)

# this one is for "-inv(M)N" from the paper to insert in lower 3X3 matrix of A
Ac[3:, 3:] = -(np.linalg.inv(D)*Cor)


Bc = np.matrix(np.zeros((6, 3)))
Bc[3:, :] = np.identity(3)

Cc = np.matrix(np.zeros((3, 6)))
Cc[:3, :3] = np.identity(3)

# discretization constant
sampling = 0.05

# model discretization
I = np.identity(Ac.shape[0])  # this is an identity matrix
A = np.linalg.inv(I-sampling*Ac)

X = np.matrix([[1],
               [theta[1]],
               [theta[2]],
               [dtheta[0]],
               [8],
               [dtheta[2]]])

# print((Cc*X))

X1 = np.matrix([[theta[0]],
               [theta[1]],
               [theta[2]],
               [dtheta[0]],
               [dtheta[1]],
               [dtheta[2]]])


states = []

A = np.array(np.ones((20, 3)))*np.pi/180
# print(A.reshape(60, 1))
print(A)

# r = Cc.shape[0]  # no of outputs
# m = Bc.shape[1]  # number of inputs
# n = Ac.shape[0]  # state dimension
