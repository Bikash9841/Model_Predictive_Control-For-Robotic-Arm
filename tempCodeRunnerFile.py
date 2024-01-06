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
timeSteps = 50

# ------------------simple trajectory--------------------------------
xd, yd, zd = invK(0.4, 0.4, 0.4, 0, 0, 0)
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