# -*- coding: utf-8 -*-
"""
Unconstrained Model Predictive Control Implementation in Python 

Tutorial page that explains how to derive the algorithm is given here:
https://aleksandarhaber.com/model-predictive-control-mpc-tutorial-1-unconstrained-formulation-derivation-and-implementation-in-python-from-scratch/
    
"""

import numpy as np
from math import *


class ModelPredictiveControl(object):
    # A,B,C - system matrices
    # f -  prediction horizon
    # v  - control horizon
    # W3 - input weight matrix
    # W4 - prediction weight matrix
    # x0 - initial state of the system
    # desiredControlTrajectoryTotal - total desired control trajectory
    #                               later on, we will take segments of this
    #                               desired state trajectory

    # discretization constant
    # sampling = 0.05
    sampling = 0.06

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

    g = 9.8065  # m/s**2, gravity

    # -------------------states computed after propagating dynamics
    state_kp1 = np.zeros((8, 1))

    def __init__(self, A, B, C, f, v, W3, W4, x0, desiredControlTrajectoryTotal):

        # initialize variables
        self.A = A
        self.B = B
        self.C = C
        self.f = f
        self.v = v
        self.W3 = W3
        self.W4 = W4
        self.desiredControlTrajectoryTotal = desiredControlTrajectoryTotal

        # dimensions of the matrices

        self.n = A.shape[0]
        self.r = C.shape[0]
        self.m = B.shape[1]

        # this variable is used to track the current time step k of the controller
        # after every calculation of the control inpu, this variables is incremented for +1
        self.currentTimeStep = 0

        # we store the state vectors of the controlled state trajectory
        self.states = []
        self.states.append(x0)

        # we store the computed inputs
        self.inputs = []

        # we store the output vectors of the controlled state trajectory
        self.outputs = []

        # form the lifted system matrices and vectors
        # the gain matrix is used to compute the solution
        # here we pre-compute it to save computational time
        self.O, self.M, self.gainMatrix = self.formLiftedMatrices(self.states[0][0, 0], self.states[0][1, 0],
                                                                  self.states[0][2, 0],
                                                                  self.states[0][3,
                                                                                 0], self.states[0][4, 0],
                                                                  self.states[0][5, 0], self.states[0][6, 0], self.states[0][7, 0])

        # ----------------------------------------------------------new segment added from here----------------------------

    # ---------------------------------------------------Matrices--------------------------------------------------

    def D_mat(self, theta0, theta1, theta2, theta3):

        # theta starts from 0 in this matrix

        d11 = self.Iz1 + self.Iz2 + self.Iz3 + self.Iz4 + self.m4*pow(cos(theta0), 2)*(self.L3*sin(theta1 + theta2) + self.L2*sin(theta1)
                                                                                       - self.L4*cos(theta1 + theta2 + theta3)) ** 2 + self.L2 ** 2*self.m2*sin(theta1) ** 2 + self.m4*sin(theta0) ** 2*(self.L3*sin(
                                                                                           theta1 + theta2) + self.L2*sin(theta1) - self.L4*cos(theta1 + theta2 + theta3)) ** 2
        + self.m3*cos(theta0) ** 2*(self.L3*sin(theta1 + theta2) + self.L2*sin(theta1)) ** 2 + \
            self.m3*sin(theta0) ** 2*(self.L3*sin(theta1 +
                                                  theta2) + self.L2*sin(theta1)) ** 2

        d12 = 0
        d13 = 0
        d14 = 0
        d21 = 0
        d22 = self.Ix3 + self.Ix4/2 + self.Iy2 + self.Iy4/2 + self.L2 ** 2*self.m2 + self.L2 ** 2*self.m3 + self.L2 ** 2 * \
            self.m4 + self.L3 ** 2*self.m3 + self.L3 ** 2*self.m4 + \
            self.L4 ** 2*self.m4 - (self.Ix4*cos(2*theta0))/2
        + (self.Iy4*cos(2*theta0))/2 - self.Ix3*cos(theta0) ** 2 + \
            self.Iy3*cos(theta0) ** 2 + self.Ix2*sin(theta0) ** 2
        - self.Iy2*sin(theta0) ** 2 + 2*self.L2*self.L4*self.m4 * \
            sin(theta2 + theta3) + 2*self.L2*self.L3*self.m3*cos(theta2)
        + 2*self.L2*self.L3*self.m4 * \
            cos(theta2) + 2*self.L3*self.L4*self.m4*sin(theta3)

        d23 = self.Ix3 + self.Ix4/2 + self.Iy4/2 + self.L3 ** 2*self.m3 + self.L3 ** 2*self.m4 + self.L4 ** 2 * \
            self.m4 - (self.Ix4*cos(2*theta0))/2 + (self.Iy4*cos(2*theta0))/2
        - self.Ix3*cos(theta0) ** 2 + self.Iy3*cos(theta0) ** 2 + self.L2*self.L4 * \
            self.m4*sin(theta2 + theta3) + self.L2*self.L3*self.m3*cos(theta2)
        + self.L2*self.L3*self.m4*cos(theta2) + \
            2*self.L3*self.L4*self.m4*sin(theta3)

        d24 = self.Ix4/2 + self.Iy4/2 + self.L4 ** 2 * \
            self.m4 - (self.Ix4*cos(2*theta0))/2
        + (self.Iy4*cos(2*theta0))/2 + self.L2*self.L4*self.m4 * \
            sin(theta2 + theta3) + self.L3*self.L4*self.m4*sin(theta3)

        d31 = 0
        d32 = self.Ix3 + self.Ix4/2 + self.Iy4/2 + self.L3 ** 2*self.m3 + self.L3 ** 2 * \
            self.m4 + self.L4 ** 2*self.m4 - (self.Ix4*cos(2*theta0))/2
        + (self.Iy4*cos(2*theta0))/2 - self.Ix3*cos(theta0) ** 2 + self.Iy3 * \
            cos(theta0) ** 2 + self.L2*self.L4*self.m4*sin(theta2 + theta3)
        + self.L2*self.L3*self.m3*cos(theta2) + self.L2*self.L3 * \
            self.m4*cos(theta2) + 2*self.L3*self.L4*self.m4*sin(theta3)
        d33 = self.Iy3 + self.Iy4 + self.L3 ** 2*self.m3 + self.L3 ** 2*self.m4 + self.L4 ** 2 * \
            self.m4 + self.Ix3*sin(theta0) ** 2 + self.Ix4*sin(theta0) ** 2
        - self.Iy3*sin(theta0) ** 2 - self.Iy4*sin(theta0) ** 2 + \
            2*self.L3*self.L4*self.m4*sin(theta3)

        d34 = self.Iy4 + self.L4 ** 2*self.m4 + self.Ix4 * \
            sin(theta0) ** 2 - self.Iy4*sin(theta0) ** 2 + \
            self.L3*self.L4*self.m4*sin(theta3)

        d41 = 0
        d42 = self.Ix4/2 + self.Iy4/2 + self.L4 ** 2*self.m4 - \
            (self.Ix4*cos(2*theta0))/2 + (self.Iy4*cos(2*theta0))/2
        + self.L2*self.L4*self.m4*sin(theta2 + theta3) + \
            self.L3*self.L4*self.m4*sin(theta3)
        d43 = self.Iy4 + self.L4 ** 2*self.m4 + self.Ix4 * \
            sin(theta0) ** 2 - self.Iy4*sin(theta0) ** 2 + \
            self.L3*self.L4*self.m4*sin(theta3)
        d44 = self.Iy4 + self.L4 ** 2*self.m4 + self.Ix4 * \
            sin(theta0) ** 2 - self.Iy4*sin(theta0) ** 2

        # ------------------------------------------------------------------------------------

        D = np.matrix([[d11, d12, d13, d14],
                       [d21, d22, d23, d24],
                       [d31, d32, d33, d34],
                       [d41, d42, d43, d44]])
        return D.round(decimals=5)

    def Cor_mat(self, theta1, theta2, theta3, theta4, theta1_d, theta2_d, theta3_d, theta4_d):

        # theta starts from 1 in this matrix...
        # Coriolis matrix components

        c11 = theta3_d*((self.L3 ** 2*self.m3*sin(2*theta2 + 2*theta3))/2 + (self.L3 ** 2*self.m4*sin(2*theta2 + 2*theta3))/2
                        - (self.L4 ** 2*self.m4*sin(2*theta2 + 2*theta3 + 2*theta4)) /
                        2 - (self.L2*self.L4*self.m4 *
                             cos(2*theta2 + theta3 + theta4))/2
                        + (self.L2*self.L4*self.m4*cos(theta3 + theta4))/2 - self.L3 *
                        self.L4*self.m4*cos(2*theta2 + 2*theta3 + theta4)
                        - (self.L2*self.L3*self.m3*sin(theta3))/2 - (self.L2*self.L3*self.m4*sin(theta3)) /
                        2 + (self.L2*self.L3*self.m3*sin(2*theta2 + theta3))/2
                        + (self.L2*self.L3*self.m4*sin(2*theta2 + theta3))/2) + theta2_d*((self.L3 ** 2*self.m3*sin(2*theta2 + 2*theta3))/2
                                                                                          + (self.L3 ** 2*self.m4*sin(2*theta2 + 2*theta3))/2 + (
                            self.L2 ** 2*self.m2*sin(2*theta2))/2 + (self.L2 ** 2*self.m3*sin(2*theta2))/2
            + (self.L2 ** 2*self.m4*sin(2*theta2))/2 - (self.L4 ** 2*self.m4*sin(2*theta2 + 2 *
                                                                                 theta3 + 2*theta4))/2 - self.L2*self.L4*self.m4*cos(2*theta2 + theta3 + theta4)
            - self.L3*self.L4*self.m4*cos(2*theta2 + 2*theta3 + theta4) + self.L2*self.L3*self.m3*sin(2*theta2 + theta3) + self.L2*self.L3*self.m4*sin(2*theta2 + theta3))
        + self.L4*self.m4*theta4_d*sin(theta2 + theta3 + theta4)*(self.L3*sin(theta2 +
                                                                              theta3) + self.L2*sin(theta2) - self.L4*cos(theta2 + theta3 + theta4))

        c12 = theta1_d*((self.L3 ** 2*self.m3*sin(2*theta2 + 2*theta3))/2 + (self.L3 ** 2*self.m4*sin(2*theta2 + 2*theta3))/2 + (self.L2 ** 2*self.m2*sin(2*theta2))/2
                        + (self.L2 ** 2*self.m3*sin(2*theta2))/2 + (self.L2 ** 2*self.m4*sin(2*theta2)) /
                        2 - (self.L4 ** 2*self.m4 *
                             sin(2*theta2 + 2*theta3 + 2*theta4))/2
                        - self.L2*self.L4*self.m4*cos(2*theta2 + theta3 + theta4) -
                        self.L3*self.L4*self.m4 *
                        cos(2*theta2 + 2*theta3 + theta4)
                        + self.L2*self.L3*self.m3*sin(2*theta2 + theta3) + self.L2*self.L3*self.m4*sin(2*theta2 + theta3)) - (theta2_d*sin(2*theta1)*(self.Ix2 + self.Ix3 + self.Ix4 - self.Iy2 - self.Iy3 - self.Iy4))/2
        - (theta3_d*sin(2*theta1)*(self.Ix3 + self.Ix4 - self.Iy3 - self.Iy4)) / \
            2 - (theta4_d*sin(2*theta1)*(self.Ix4 - self.Iy4))/2

        c13 = theta1_d*((self.L3 ** 2*self.m3*sin(2*theta2 + 2*theta3))/2 + (self.L3 ** 2*self.m4*sin(2*theta2 + 2*theta3))/2
                        - (self.L4 ** 2*self.m4*sin(2*theta2 + 2*theta3 + 2*theta4)) /
                        2 - (self.L2*self.L4*self.m4 *
                             cos(2*theta2 + theta3 + theta4))/2
                        + (self.L2*self.L4*self.m4*cos(theta3 + theta4))/2 - self.L3 *
                        self.L4*self.m4*cos(2*theta2 + 2*theta3 + theta4)
                        - (self.L2*self.L3*self.m3*sin(theta3))/2 - (self.L2*self.L3*self.m4*sin(theta3)) /
                        2 + (self.L2*self.L3*self.m3*sin(2*theta2 + theta3))/2
                        + (self.L2*self.L3*self.m4*sin(2*theta2 + theta3))/2) - (theta2_d*sin(2*theta1)*(self.Ix3 + self.Ix4 - self.Iy3 - self.Iy4))/2
        - (theta3_d*sin(2*theta1)*(self.Ix3 + self.Ix4 - self.Iy3 - self.Iy4)) / \
            2 - (theta4_d*sin(2*theta1)*(self.Ix4 - self.Iy4))/2

        c14 = self.L4*self.m4*theta1_d*sin(theta2 + theta3 + theta4)*(
            self.L3*sin(theta2 + theta3) + self.L2*sin(theta2) - self.L4*cos(theta2 + theta3 + theta4))
        - (theta3_d*sin(2*theta1)*(self.Ix4 - self.Iy4))/2 - (theta4_d*sin(2*theta1)
                                                              * (self.Ix4 - self.Iy4))/2 - (theta2_d*sin(2*theta1)*(self.Ix4 - self.Iy4))/2

        c21 = (theta2_d*sin(2*theta1)*(self.Ix2 + self.Ix3 + self.Ix4 - self.Iy2 - self.Iy3 - self.Iy4))/2 - theta1_d*((self.L3 ** 2*self.m3*sin(2*theta2 + 2*theta3))/2
                                                                                                                       + (self.L3 ** 2*self.m4*sin(2*theta2 + 2*theta3))/2
                                                                                                                       + (self.L2 ** 2*self.m2*sin(2*theta2))/2
                                                                                                                       + (self.L2 ** 2*self.m3*sin(2*theta2))/2
                                                                                                                       + (self.L2 ** 2*self.m4*sin(2*theta2))/2
                                                                                                                       - (self.L4 ** 2*self.m4*sin(2*theta2 + 2*theta3 + 2*theta4))/2
                                                                                                                       - self.L2*self.L4*self.m4 *
                                                                                                                       cos(
                                                                                                                           2*theta2 + theta3 + theta4)
                                                                                                                       - self.L3*self.L4*self.m4 *
                                                                                                                       cos(
            2*theta2 + 2*theta3 + theta4)
            + self.L2*self.L3*self.m3*sin(2*theta2 + theta3) + self.L2*self.L3*self.m4*sin(2*theta2 + theta3))
        + (theta3_d*sin(2*theta1)*(self.Ix3 + self.Ix4 - self.Iy3 - self.Iy4)) / \
            2 + (theta4_d*sin(2*theta1)*(self.Ix4 - self.Iy4))/2

        c22 = (theta1_d*sin(2*theta1)*(self.Ix2 + self.Ix3 + self.Ix4 - self.Iy2 - self.Iy3 - self.Iy4))/2 - self.L2*theta3_d*(self.L3*self.m3*sin(theta3) + self.L3*self.m4*sin(theta3)
                                                                                                                               - self.L4*self.m4*cos(theta3 + theta4))
        + self.L4*self.m4*theta4_d * \
            (self.L2*cos(theta3 + theta4) + self.L3*cos(theta4))

        c23 = (theta1_d*sin(2*theta1)*(self.Ix3 + self.Ix4 - self.Iy3 - self.Iy4))/2 - self.L2*theta2_d*(self.L3*self.m3*sin(theta3) + self.L3*self.m4*sin(theta3)
                                                                                                         - self.L4*self.m4*cos(theta3 + theta4)) - self.L2*theta3_d*(self.L3*self.m3*sin(theta3) + self.L3*self.m4*sin(theta3)
                                                                                                                                                                     - self.L4*self.m4*cos(theta3 + theta4))
        + self.L4*self.m4*theta4_d * \
            (self.L2*cos(theta3 + theta4) + self.L3*cos(theta4))

        c24 = (theta1_d*sin(2*theta1)*(self.Ix4 - self.Iy4))/2 + self.L4*self.m4*theta2_d*(self.L2*cos(theta3 + theta4)
                                                                                           + self.L3*cos(theta4)) + self.L4*self.m4*theta3_d*(self.L2*cos(theta3 + theta4)
                                                                                                                                              + self.L3*cos(theta4)) + self.L4*self.m4*theta4_d*(self.L2*cos(theta3 + theta4)
                                                                                                                                                                                                 + self.L3*cos(theta4))

        c31 = (theta2_d*sin(2*theta1)*(self.Ix3 + self.Ix4 - self.Iy3 - self.Iy4))/2 - theta1_d*((self.L3 ** 2*self.m3*sin(2*theta2 + 2*theta3))/2
                                                                                                 + (self.L3 ** 2*self.m4*sin(2*theta2 + 2*theta3))/2
                                                                                                 - (self.L4 ** 2*self.m4*sin(2*theta2 + 2*theta3 + 2*theta4))/2
                                                                                                 - (self.L2*self.L4*self.m4*cos(2*theta2 + theta3 + theta4))/2
                                                                                                 + (self.L2*self.L4*self.m4*cos(theta3 + theta4))/2
                                                                                                 - self.L3*self.L4*self.m4 *
                                                                                                 cos(2*theta2 + 2 *
                                                                                                     theta3 + theta4)
                                                                                                 - (self.L2*self.L3*self.m3*sin(theta3))/2
                                                                                                 - (self.L2*self.L3*self.m4*sin(theta3))/2 + (
            self.L2*self.L3*self.m3*sin(2*theta2 + theta3))/2
            + (self.L2*self.L3*self.m4*sin(2*theta2 + theta3))/2) + (theta3_d*sin(2*theta1)*(self.Ix3 + self.Ix4 - self.Iy3 - self.Iy4))/2 + (theta4_d*sin(2*theta1)*(self.Ix4 - self.Iy4))/2

        c32 = (theta1_d*sin(2*theta1)*(self.Ix3 + self.Ix4 - self.Iy3 - self.Iy4))/2 + self.L2*theta2_d*(self.L3*self.m3*sin(theta3) + self.L3*self.m4*sin(theta3)
                                                                                                         - self.L4*self.m4*cos(theta3 + theta4)) + self.L3*self.L4*self.m4*theta4_d*cos(theta4)

        c33 = (theta1_d*sin(2*theta1)*(self.Ix3 + self.Ix4 - self.Iy3 - self.Iy4)) / \
            2 + self.L3*self.L4*self.m4*theta4_d*cos(theta4)

        c34 = (theta1_d*sin(2*theta1)*(self.Ix4 - self.Iy4))/2 + self.L3*self.L4*self.m4*theta2_d * \
            cos(theta4) + self.L3*self.L4*self.m4*theta3_d * \
            cos(theta4) + self.L3*self.L4*self.m4*theta4_d*cos(theta4)

        c41 = (theta2_d*sin(2*theta1)*(self.Ix4 - self.Iy4))/2 + \
            (theta3_d*sin(2*theta1)*(self.Ix4 - self.Iy4))/2
        + (theta4_d*sin(2*theta1)*(self.Ix4 - self.Iy4))/2 - self.L4*self.m4*theta1_d*sin(theta2 + theta3 + theta4)*(self.L3*sin(theta2 + theta3)
                                                                                                                     + self.L2 *
                                                                                                                     sin(
            theta2)
            - self.L4*cos(theta2 + theta3 + theta4))

        c42 = (theta1_d*sin(2*theta1)*(self.Ix4 - self.Iy4))/2 - self.L4*self.m4*theta2_d*(self.L2 *
                                                                                           cos(theta3 + theta4) + self.L3*cos(theta4)) - self.L3*self.L4*self.m4*theta3_d*cos(theta4)

        c43 = (theta1_d*sin(2*theta1)*(self.Ix4 - self.Iy4))/2 - self.L3*self.L4*self.m4 * \
            theta2_d*cos(theta4) - self.L3*self.L4*self.m4*theta3_d*cos(theta4)

        c44 = (theta1_d*sin(2*theta1)*(self.Ix4 - self.Iy4))/2
        # ---------------------------------------------------------------------

        Cor = np.matrix([[c11, c12, c13, c14],
                        [c21, c22, c23, c24],
                        [c31, c32, c33, c34],
                        [c41, c42, c43, c44]])
        return Cor.round(decimals=5)

    def Gra_mat(self, theta2, theta3, theta4):
        # gravity vectors for 3DOF robotic arm
        # theta starts from 1
        g1 = 0

        g2 = self.m2*self.g*self.l2*cos(theta2)+self.m3*self.g*self.L2*cos(theta2) + self.m4*self.g*self.l3*cos(
            theta2+theta3)+self.m4*self.g*self.L2*cos(theta2)+self.m4*self.g*self.L3*cos(theta2+theta3)
        +self.m4*self.g*self.l4*cos(theta2+theta3+theta4)

        g3 = self.m3*self.g*self.l3*cos(theta2+theta3)+self.m4*self.g*self.L3*cos(
            theta2+theta3)+self.m4*self.g*self.l4*cos(theta2+theta3+theta4)

        g4 = self.m4*self.g*self.l4*cos(theta2+theta3+theta4)
        # ----------------------------------------------------------------------------
        Gra = np.matrix([[g1],
                         [g2],
                         [g3],
                         [g4]])
        return Gra.round(decimals=5)

    def Ainitial_mat(self, theta0, theta1, theta2, theta3, dtheta0, dtheta1, dtheta2, dtheta3):

        Cor = self.Cor_mat(theta0, theta1, theta2, theta3,
                           dtheta0, dtheta1, dtheta2, dtheta3)
        D = self.D_mat(theta0, theta1, theta2, theta3)

        # ------------------------------------State Space modelling starts from here---------------------------------
        Ac = np.matrix(np.zeros((8, 8)))
        Ac[:4, 4:] = np.identity(4)

        # this one is for "-inv(M)N" from the paper to insert in lower 3X3 matrix of A
        Ac[4:, 4:] = -(np.linalg.inv(D) * Cor)
        return Ac

    def ABC_final(self, theta0, theta1, theta2, theta3, dtheta0, dtheta1, dtheta2, dtheta3):

        Ac = self.Ainitial_mat(theta0, theta1, theta2, theta3,
                               dtheta0, dtheta1, dtheta2, dtheta3)

        Bc = np.matrix(np.zeros((8, 4)))
        Bc[4:, :] = np.identity(4)

        Cc = np.matrix(np.zeros((4, 8)))
        Cc[:4, :4] = np.identity(4)

        # model discretization
        I = np.identity(Ac.shape[0])  # this is an identity matrix
        self.A = (np.linalg.inv(I - self.sampling * Ac))
        self.B = self.A * self.sampling * Bc
        self.C = Cc

        return self.A.round(decimals=4), self.B.round(decimals=4), self.C.round(decimals=4)

    # ---------------------------------------------------------------new section added ended here------------------------

    # this function forms the lifted matrices O and M, as well as the
    # the gain matrix of the control algorithm
    # and returns them

    def formLiftedMatrices(self, theta0, theta1, theta2, theta3, dtheta0, dtheta1, dtheta2, dtheta3):
        f = self.f
        v = self.v
        r = self.r
        n = self.n
        m = self.m
        '''
        A = self.A
        B = self.B
        C = self.C
        '''

        self.A, self.B, self.C = self.ABC_final(
            theta0, theta1, theta2, theta3, dtheta0, dtheta1, dtheta2, dtheta3)

        # lifted matrix O
        O = np.zeros(shape=(f * r, n))

        for i in range(f):
            if (i == 0):
                powA = self.A
            else:
                powA = np.matmul(powA, self.A)
            O[i * r:(i + 1) * r, :] = (np.matmul(self.C, powA))

        # lifted matrix M
        M = np.zeros(shape=(f * r, v * m))

        for i in range(f):
            # until the control horizon
            if (i < v):
                for j in range(i + 1):
                    if (j == 0):
                        powA = np.eye(n, n)
                    else:
                        powA = np.matmul(powA, self.A)
                    M[i * r:(i + 1) * r, (i - j) * m:(i - j + 1) *
                      m] = np.matmul(self.C, np.matmul(powA, self.B))

            # from control horizon until the prediction horizon
            else:
                for j in range(v):
                    # here we form the last entry
                    if j == 0:
                        sumLast = np.zeros(shape=(n, n))
                        for s in range(i - v + 2):
                            if (s == 0):
                                powA = np.eye(n, n)
                            else:
                                powA = np.matmul(powA, self.A)
                            sumLast = sumLast + powA
                        M[i * r:(i + 1) * r, (v - 1) * m:(v) * m] = np.matmul(self.C,
                                                                              np.matmul(sumLast, self.B))
                    else:
                        powA = np.matmul(powA, self.A)
                        M[i * r:(i + 1) * r, (v - 1 - j) * m:(v - j) *
                          m] = np.matmul(self.C, np.matmul(powA, self.B))

        tmp1 = np.matmul(M.T, np.matmul(self.W4, M))
        # tmp2 = np.linalg.inv(tmp1+self.W3)
        tmp2 = np.linalg.pinv((tmp1 + self.W3) + 1e-03 *
                              np.eye((tmp1 + self.W3).shape[1]))
        gainMatrix = np.matmul(tmp2, np.matmul(M.T, self.W4))

        return O.round(decimals=4), M.round(decimals=4), gainMatrix.round(decimals=4)

    # this function propagates the dynamics
    # x_{k+1}=Ax_{k}+Bu_{k}

    def propagateDynamics(self, controlInput, state):

        xkp1 = np.zeros(shape=(self.n, 1))
        yk = np.zeros(shape=(self.r, 1))

        xkp1 = np.matmul(self.A, state) + np.matmul(self.B, controlInput)

        yk = np.matmul(self.C, state)

        return xkp1, yk

    # this function computes the control inputs, applies them to the system
    # by calling the propagateDynamics() function and appends the lists
    # that store the inputs, outputs, states
    def computeControlInputs(self):

        remaining_timesteps = len(
            self.desiredControlTrajectoryTotal) - self.currentTimeStep

        if remaining_timesteps >= self.f:  # *3:
            desiredControlTrajectory = self.desiredControlTrajectoryTotal[
                self.currentTimeStep:self.currentTimeStep + self.f]  # * 3]
            # print(f"desiredControlnotpadding: {desiredControlTrajectory}")
        else:

            # Repeat the last available value to fill the remaining timesteps
            last_value = self.desiredControlTrajectoryTotal[-1]
            padding = self.f - remaining_timesteps  # * 3 - remaining_timesteps
            desiredControlTrajectory = np.concatenate([
                self.desiredControlTrajectoryTotal[self.currentTimeStep:],
                np.tile(last_value, (padding, 1))
            ])

        # desiredControlTrajectory = desiredControlTrajectory.reshape(60, 1)
        desiredControlTrajectory = desiredControlTrajectory.reshape(40, 1)
        # print(f"desiredControlafterReshaping: {desiredControlTrajectory}")

        # compute the vector s
        vectorS = desiredControlTrajectory - \
            np.matmul(self.O, self.states[self.currentTimeStep])

        # compute the control sequence
        inputSequenceComputed = np.matmul(self.gainMatrix, vectorS)

        D = self.D_mat(self.state_kp1[0, 0],
                       self.state_kp1[1, 0], self.state_kp1[2, 0], self.state_kp1[3, 0])
        G = self.Gra_mat(
            self.state_kp1[1, 0], self.state_kp1[2, 0], self.state_kp1[3, 0])

        inputApplied = np.zeros(shape=(4, 1))

        inputApplied[0:4, :] = inputSequenceComputed[0:4, :]

        # --------------------------------------this one is torque that is computed----------------------------

        # print(inputApplied, self.states[self.currentTimeStep])
        # compute the next state and output
        self.state_kp1, output_k = self.propagateDynamics(
            inputApplied, self.states[self.currentTimeStep])

        # append the lists
        inputApplied = np.matmul(D, inputApplied) + G

        # self.states.append(self.state_kp1[:])
        self.states.append(self.state_kp1)
        self.outputs.append(output_k)
        self.inputs.append(inputApplied)

        self.state_kp1 = self.state_kp1*(np.pi/180)
        self.O, self.M, self.gainMatrix = self.formLiftedMatrices(self.state_kp1[0, 0], self.state_kp1[1, 0],
                                                                  self.state_kp1[2,
                                                                                 0], self.state_kp1[3, 0],
                                                                  self.state_kp1[4, 0], self.state_kp1[5, 0], self.state_kp1[6, 0], self.state_kp1[7, 0])
        # increment the time step
        self.currentTimeStep = self.currentTimeStep + 1

    def update_weights(self, new_W3, new_W4):
        # Update internal variables for weights
        self.W3 = new_W3
        self.W4 = new_W4
