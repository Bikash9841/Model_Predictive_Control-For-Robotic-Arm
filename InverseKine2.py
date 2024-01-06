import math
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import scipy
'''
def pseudoJac(t1, t2, t3, t4):

    # link length
    a1 = 0.214
    a2 = 0.37
    a3 = 0.354
    a4 = 0.28

    # changing degrees to radian

    t1 = t1*math.pi/180
    t2 = t2*math.pi/180
    t3 = t3*math.pi/180
    t4 = t4*math.pi/180

    J = [[-math.sin(t1)*(a3*math.cos(t2+t3)+a2*math.cos(t2)+a4*math.cos(t2+t3+t4)),
          -math.cos(t1)*(a3*math.sin(t2+t3)+a2 *
                         math.sin(t2)+a4*math.sin(t2+t3+t4)),
          -math.cos(t1)*(a3*math.sin(t2+t3)+a4*math.sin(t2+t3+t4)),
          -a4*math.sin(t2+t3+t4)*math.cos(t1)],
         [math.cos(t1)*(a3*math.cos(t2+t3)+a2*math.cos(t2)+a4*math.cos(t2+t3+t4)),
         -math.sin(t1)*(a3*math.sin(t2+t3)+a2 *
                        math.sin(t2)+a4*math.sin(t2+t3+t4)),
         -math.sin(t1)*(a3*math.sin(t2+t3)+a4*math.sin(t2+t3+t4)),
         -a4*math.sin(t2+t3+t4)*math.sin(t1)],
         [0,
         a3*math.cos(t2+t3)+a2*math.cos(t2)+a4*math.cos(t2+t3+t4),
         (a3*math.cos(t2+t3)+a4*math.cos(t2+t3+t4)),
         a4*math.cos(t2+t3+t4)]]

    Jt = np.transpose(J)

    Jdot = np.dot(J, Jt)

    Jseudo = np.dot(Jt, np.linalg.inv(Jdot))
    print(Jt)
    print(Jseudo)
    return Jseudo

    # print(Jseudo)

# theta=np.array([[0],[0],[0],[-90]])
# s=1
# Xt=np.array([[0.22],[-0.80],[0.76]])
# Xi=np.array([[0.78],[-0.56],[1.811]])

# Jplus=pseudoJac(0,0,0,-90)

# theta=theta+s*np.dot(Jplus,(Xt-Xi))
# print(theta)
'''

# =====================================================================
# =====================================================================
# This one is also working inverse kinematics with orientation too, Gradient Descent optimization.
# =====================================================================
# -----------lets start the inverse kinematics via optimization


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
sim.setJointTargetPosition(joint4, np.deg2rad(90))

time.sleep(1)


# link length
# a1 = 0.214
# a2 = 0.37
# a3 = 0.354
# a4 = 0.28
L1 = 0.289  # in m, length of link1
L2 = 0.372  # in m, length of link2
L3 = 0.351
L4 = 0.40

# Example: Initial guess for joint angles
initial_guess = np.array([0.0, 0.0, 0.0, 0.0])


def computeTransformationMatrix(q1, q2, q3, q4):
    '''
    takes in the current joint angles(rad) and return
    rotation matrix and displacement vector for 4DOF arm
    '''

    # # changing from degrees to radian

    # t1 = t1*math.pi/180
    # t2 = t2*math.pi/180
    # t3 = t3*math.pi/180
    # t4 = t4*math.pi/180

    # forward kinematics equations

    x = -np.cos(q1)*(L3*np.sin(q2+q3)+L2*np.sin(q2)-L4*np.cos(q2+q3+q4))
    y = -np.sin(q1)*(L3*np.sin(q2+q3)+L2 * np.sin(q2)-L4*np.cos(q2+q3+q4))
    z = L1+L3*np.cos(q2+q3)+L2*np.cos(q2)+L4*np.sin(q2+q3+q4)

    H = np.array([[np.cos(q1)*np.cos(q2+q3+q4), np.cos(q1)*-np.sin(q2+q3+q4), np.sin(q1), x],
                  [np.sin(q1)*np.cos(q2+q3+q4), np.sin(q1)*-
                   np.sin(q2+q3+q4), -np.cos(q1), y],
                  [np.sin(q2+q3+q4), np.cos(q2+q3+q4), 0, z],
                  [0, 0, 0, 1]])

    # H = np.array([[np.cos(q1)*np.cos(q2+q3+q4), 0, 0, x],
    #               [0, 0, -np.cos(q1), y],
    #               [0, np.cos(q2+q3+q4), 0, z],
    #               [0, 0, 0, 1]])

    RotMat = np.matrix(np.ones((3, 3)))
    DisVector = np.matrix(np.ones((3, 1)))
    DisVector[:3, :] = H[:3, 3:]
    RotMat[:3, :3] = H[:3, :3]

    return RotMat.round(decimals=3), DisVector.round(decimals=3)


def desiredTransformationMatrix(x_desired, y_desired, z_desired):
    '''
    takes in desired position of end effector.
    desired rotation matrix defined already inside the function.
    '''
    # RotMat = np.matrix(np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))
    RotMat = np.matrix(
        np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))
    DisVector = np.matrix(np.array([[x_desired], [y_desired], [z_desired]]))

    return RotMat, DisVector

# simply to calculate rotation matrix by hit and trial


def rot(q1, q2, q3, q4):

    q1 = np.deg2rad(q1)
    q2 = np.deg2rad(q2)
    q3 = np.deg2rad(q3)
    q4 = np.deg2rad(q4)
    H = np.array([[np.cos(q1)*np.cos(q2+q3+q4), np.cos(q1)*-np.sin(q2+q3+q4), np.sin(q1), 1],
                  [np.sin(q1)*np.cos(q2+q3+q4), np.sin(q1)*-
                 np.sin(q2+q3+q4), -np.cos(q1), 1],
                  [np.sin(q2+q3+q4), np.cos(q2+q3+q4), 0, 1],
                  [0, 0, 0, 1]])
    print(H)


# rot(-1.6, -66.3,   1.0,   22.2)

# ALERT________________YOU NEED TO FIND JACOBIAN MATRIX FROM MATLAB
# desiredTransformationMatrix(0.4, 0.4, 0.4)


def pseudoJac(t1, t2, t3, t4):
    '''
    takes in the joint angles in radian and return
    pseudo inverse of jacobian matrix.
    '''
    J = np.matrix([[np.sin(t1)*(L3*np.sin(t2+t3)+L2*np.sin(t2)-L4*np.cos(t2+t3+t4)),
                    -np.cos(t1)*(L3*np.cos(t2+t3)+L2 *
                                 np.cos(t2)+L4*np.sin(t2+t3+t4)),
                    -np.cos(t1)*(L3*np.cos(t2+t3)+L4*np.sin(t2+t3+t4)),
                    -L4*np.sin(t2+t3+t4)*np.cos(t1)],
                   [-np.cos(t1)*(L3*np.sin(t2+t3)+L2*np.sin(t2)-L4*np.cos(t2+t3+t4)),
                    -np.sin(t1)*(L3*np.cos(t2+t3)+L2 *
                                 np.cos(t2)+L4*np.sin(t2+t3+t4)),
                    -np.sin(t1)*(L3*np.cos(t2+t3)+L4*np.sin(t2+t3+t4)),
                    -L4*np.sin(t2+t3+t4)*np.sin(t1)],
                   [0,
                    -L3*np.sin(t2+t3)-L2*np.sin(t2)+L4*np.cos(t2+t3+t4),
                    (-L3*np.sin(t2+t3)+L4*np.cos(t2+t3+t4)),
                    L4*np.cos(t2+t3+t4)],
                   [0, np.sin(t1), np.sin(t1), np.sin(t1)],
                   [0, -np.cos(t1), -np.cos(t1), -np.cos(t1)],
                   [1, 0, 0, 0]])
    return J.T
    # return np.matrix(np.linalg.pinv(J).round(decimals=4))


def invOpt(q1_initial, q2_initial, q3_initial, q4_initial, x, y, z):
    errorList = np.array([1])
    err = 1
    q = np.matrix([[np.deg2rad(q1_initial)], [np.deg2rad(q2_initial)], [
                  np.deg2rad(q3_initial)], [np.deg2rad(q4_initial)]])

    delx = np.matrix(np.zeros((6, 1)))
    iter = 0
    Rd, Dd = desiredTransformationMatrix(x, y, z)
    # Example: Joint angle bounds (replace -np.pi and np.pi with your desired bounds)
    # jb = [(-np.pi/2, np.pi/2)for _ in range(len(q))]

    jb = [
        (np.deg2rad(-180), np.deg2rad(180)),  # Bounds for q1
        (np.deg2rad(-72), np.deg2rad(72)),      # Bounds for q2
        (np.deg2rad(-150), np.deg2rad(150)),  # Bounds for q3
        (np.deg2rad(-150), np.deg2rad(150))   # Bounds for q4
    ]
    # print(jb)

    while err >= 1e-4 and iter < 10000:

        Rk, Dk = computeTransformationMatrix(
            q[0, 0], q[1, 0], q[2, 0], q[3, 0])

        # difference in actual and desired pose( both position and orientation error)
        ep = Dd-Dk
        # eo = Rd*Rk.T

        # # extracting roll,pitch and yaw from the rotation matrix
        roll = np.arctan2(Rk[2, 1], Rk[2, 2])
        yaw = np.arctan2(Rk[1, 0], Rk[0, 0])

        if (np.cos(yaw) == 0):
            pitch = np.arctan2(-Rk[2, 0], (Rk[1, 0]/np.sin(yaw)))
        else:
            pitch = np.arctan2(-Rk[2, 0], (Rk[0, 0]/np.cos(yaw)))

        # deriving the pose error vector
        delx[0:3, :] = ep
        delx[3:6, :] = [[0], [np.deg2rad(5)-pitch], [0]]

        Jinv = pseudoJac(q[0, 0], q[1, 0], q[2, 0], q[3, 0])
        delq = Jinv * delx

        q = q+0.05*delq

        # Apply joint angle bounds
        # q = np.clip(q, *zip(*jb))

        # Enforce joint angle bounds
        for i in range(len(jb)):
            q[i, 0] = np.clip(q[i, 0], *jb[i])

        err = np.linalg.norm(delq)

        errorList = np.append(errorList, err)

        # if iter < 20:
        # print(delq)

        iter = iter+1

    return np.rad2deg(q).round(decimals=1), errorList


optimizedJointAngles, errorList = invOpt(
    initial_guess[0], initial_guess[1], initial_guess[2], initial_guess[3],
    0.731, 0.003, 0.5)

print(optimizedJointAngles)
print(errorList[-1])
plt.plot(errorList, linewidth=4, label='Position Error')
plt.xlabel('timesteps')
plt.ylabel('Error magnitude')
plt.legend()
plt.show()

timeSteps = 50
# Generate time vector
timeVector = np.linspace(0, 1, timeSteps)

# Generate cubic spline trajectories for each joint
spline_joint0 = CubicSpline(
    [0, 1], [initial_guess[0], optimizedJointAngles[0][0]])
spline_joint1 = CubicSpline(
    [0, 1], [initial_guess[1], optimizedJointAngles[1][0]])
spline_joint2 = CubicSpline(
    [0, 1], [initial_guess[2], optimizedJointAngles[2][0]])
spline_joint3 = CubicSpline(
    [0, 1], [initial_guess[3], optimizedJointAngles[3][0]])

# Evaluate the splines at each time step
trajectory_joint0 = spline_joint0(timeVector)
trajectory_joint1 = spline_joint1(timeVector)
trajectory_joint2 = spline_joint2(timeVector)
trajectory_joint3 = spline_joint3(timeVector)

# Stack the individual trajectories to form the desired trajectory matrix
desiredJointAngle = np.matrix(np.column_stack(
    (trajectory_joint0, trajectory_joint1, trajectory_joint2, trajectory_joint3)))

# print(desiredJointAngle.shape)

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


get_ang()
