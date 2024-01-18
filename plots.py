import matplotlib.pyplot as plt


def visualize(desiredTrajectoryList1, controlledTrajectoryList1, desiredTrajectoryList2, controlledTrajectoryList2,
              desiredTrajectoryList3, controlledTrajectoryList3, desiredTrajectoryList4, controlledTrajectoryList4,
              controlInputList1, controlInputList2, controlInputList3, controlInputList4):

    fig, axs = plt.subplots(2, 4)
    axs[0, 0].plot(desiredTrajectoryList1, linewidth=4, color='black',
                   label='Desired trajectory1')
    axs[0, 0].plot(controlledTrajectoryList1, linewidth=3, linestyle='dashed', color='black',
                   label='Controlled trajectory1')
    axs[0, 0].set_title('JOINT 1')
    axs[0, 0].set_ylabel('Joint Angles (in deg)')
    axs[0, 0].set_xlabel('timesteps')
    axs[0, 0].legend()
    # -------------------------------------------------------------

    axs[0, 1].plot(desiredTrajectoryList2, linewidth=4, color='black',
                   label='Desired trajectory2')
    axs[0, 1].plot(controlledTrajectoryList2, linewidth=3, linestyle='dashed', color='black',
                   label='Controlled trajectory2')
    axs[0, 1].set_title('JOINT 2')
    axs[0, 1].set_xlabel('timesteps')
    axs[0, 1].legend()
    # ---------------------------------------------------------------

    axs[0, 2].plot(desiredTrajectoryList3, linewidth=4, color='black',
                   label='Desired trajectory3')
    axs[0, 2].plot(controlledTrajectoryList3, linewidth=3, linestyle='dashed', color='black',
                   label='Controlled trajectory3')
    axs[0, 2].set_title('JOINT 3')
    axs[0, 2].set_ylabel('Joint Angles (in deg)')
    axs[0, 2].set_xlabel('timesteps')
    axs[0, 2].legend()
    # ---------------------------------------------------------------

    axs[0, 3].plot(desiredTrajectoryList4, linewidth=4, color='black',
                   label='Desired trajectory4')
    axs[0, 3].plot(controlledTrajectoryList4, linewidth=3, linestyle='dashed', color='black',
                   label='Controlled trajectory4')
    axs[0, 3].set_title('JOINT 4')
    axs[0, 3].set_xlabel('timesteps')
    axs[0, 3].legend()

    axs[1, 0].plot(controlInputList1, linewidth=4, color='black',
                   label='Control Torque1')
    axs[1, 0].set_title('Torque of Joint1')
    axs[1, 0].set_ylabel('Torque (Nm)')
    axs[1, 0].set_xlabel('timesteps')
    axs[1, 0].legend()

    axs[1, 1].plot(controlInputList2, linewidth=4, color='black',
                   label='Control Torque2')
    axs[1, 1].set_title('Torque of Joint2')
    axs[1, 1].set_xlabel('timesteps')
    axs[1, 1].legend()

    axs[1, 2].plot(controlInputList3, linewidth=4, color='black',
                   label='Control Torque3')
    axs[1, 2].set_title('Torque of Joint3')
    axs[1, 2].set_ylabel('Torque (Nm)')
    axs[1, 2].set_xlabel('timesteps')
    axs[1, 2].legend()

    axs[1, 3].plot(controlInputList4, linewidth=4, color='black',
                   label='Control Torque4')
    axs[1, 3].set_title('Torque of Joint4')
    axs[1, 3].set_xlabel('timesteps')
    axs[1, 3].legend()

    plt.show()
