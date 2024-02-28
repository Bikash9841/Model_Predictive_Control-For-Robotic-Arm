

# This one is also working inverse kinematics with orientation too without simulation.
# just to analayze the error graphs for multiple situations.

import numpy as np
import matplotlib.pyplot as plt


# link length
# a1 = 0.214
# a2 = 0.37
# a3 = 0.354
# a4 = 0.28

# L1 = 0.289  # in m, length of link1
# L2 = 0.372  # in m, length of link2
# L3 = 0.351
# L4 = 0.33

L1 = 0.17  # in m, length of link1
L2 = 0.20  # in m, length of link2
L3 = 0.20
L4 = 0.20


# Example: Initial guess for joint angles
# initial_guess = np.array([0, 0, 0, 0])


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


def pseudoJac(t1, t2, t3, t4):
    '''
    takes in the joint angles in radian and return
    pseudo inverse of jacobian matrix.
    '''

    # the first 3 rows across the columns of this jacobian matrix J are correct.
    # confused about the last 3 rows.
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
    '''
    takes joint angles in degress and target position in metres
    '''

    errorList = np.array([1])
    err = 1
    q = np.matrix([[np.deg2rad(q1_initial)], [np.deg2rad(q2_initial)], [
                  np.deg2rad(q3_initial)], [np.deg2rad(q4_initial)]])

    delx = np.matrix(np.zeros((6, 1)))
    iter = 0
    Rd, Dd = desiredTransformationMatrix(x, y, z)

    # Joint angle bounds (replace -np.pi and np.pi with your desired bounds)
    # jb = [(-np.pi/2, np.pi/2)for _ in range(len(q))]
    jb = [
        (np.deg2rad(-180), np.deg2rad(180)),  # Bounds for q1
        (np.deg2rad(-72), np.deg2rad(72)),      # Bounds for q2
        (np.deg2rad(-150), np.deg2rad(150)),  # Bounds for q3
        (np.deg2rad(-150), np.deg2rad(150))   # Bounds for q4
    ]
    # print(jb)

    while err >= 1e-2 and iter < 250:

        Rk, Dk = computeTransformationMatrix(
            q[0, 0], q[1, 0], q[2, 0], q[3, 0])

        # difference in actual and desired pose( both position and orientation error)
        ep = Dd-Dk

        # eo = Rd*Rk.T

        # # extracting roll,pitch and yaw from the rotation matrix(old way)
        # roll = np.arctan2(eo[2, 1], eo[2, 2])
        # yaw = np.arctan2(eo[1, 0], eo[0, 0])

        # if (np.cos(yaw) == 0):
        #     pitch = np.arctan2(-eo[2, 0], (eo[1, 0]/np.sin(yaw)))
        # else:
        #     pitch = np.arctan2(-eo[2, 0], (eo[0, 0]/np.cos(yaw)))

        # new way
        roll = np.arctan2(Rk[2, 1], Rk[2, 2])
        yaw = np.arctan2(Rk[1, 0], Rk[0, 0])

        if (np.cos(yaw) == 0):
            pitch = np.arctan2(-Rk[2, 0], (Rk[1, 0]/np.sin(yaw)))
        else:
            pitch = np.arctan2(-Rk[2, 0], (Rk[0, 0]/np.cos(yaw)))

        # deriving the pose error vector
        delx[0:3, :] = ep

        # pitch error should be zero. but offset of 5deg is taken as it provide good results
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


# optimizedJointAngles, _ = invOpt(
#     0, 0, 0, 0, 0.587, -0.764, 0.05)
# print(optimizedJointAngles)

'''
optimizedJointAngles, errorList = invOpt(
    initial_guess[0], initial_guess[1], initial_guess[2], initial_guess[3],
    0.58, -0.58, 0.1)


print(optimizedJointAngles)
print(errorList[-1])

plt.plot(errorList, linewidth=4, label='Position Error')
plt.xlabel('timesteps')
plt.ylabel('Error magnitude')
plt.legend()
plt.show()
'''
