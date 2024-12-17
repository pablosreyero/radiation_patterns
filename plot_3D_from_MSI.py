import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf

def msi_file_reader(msi_path):
    """
    This function reads the most relevant information out of an .msi file
    to further reconstruct the 3D radiation pattern of a given antenna

    Input:
    - msi_path (str) --> A sting containing the path to the .msi file

    Otuputs:
    - horizontal_gains (tf.constant)
    - vertical_gains (tf.constant)
    - antenna_gain (float)
    """

    # Initialize lists to hold the angles and gains for horizontal and vertical patterns
    horizontal_gains = []
    vertical_gains = []

    # Parse the .msi file
    with open(msi_path, 'r') as file:
        for line in file: # let us firstly search for the antenna gain
            # Use regex to match the 'GAIN' keyword and extract the value
            match = re.search(r"GAIN\s+([\d.]+)\s*dBd", line, re.IGNORECASE)
            if match:
                antenna_gain = float(match.group(1))
                antenna_gain = antenna_gain + 2.15 # turn dBds into dBis
                break
            else:
                print(f'No antenna GAIN detected !!')

        # let us extract both horizontal and vertical gains
        lines = file.readlines()
        horizontal_section = False
        vertical_section = False

        for line in lines:
            # Identify sections
            if "HORIZONTAL" in line:
                horizontal_section = True
                vertical_section = False
                continue
            elif "VERTICAL" in line:
                vertical_section = True
                horizontal_section = False
                continue

            # Parse angles and gains
            if horizontal_section:
                _, gain = map(float, line.strip().split())
                horizontal_gains.append(gain)
            elif vertical_section:
                _, gain = map(float, line.strip().split())
                vertical_gains.append(gain)

        # let us convert both gain lists into tensorflow
        horizontal_gains = tf.constant(horizontal_gains)
        vertical_gains = tf.constant(vertical_gains)

    return antenna_gain, horizontal_gains, vertical_gains


def cross_weighted_algorithm(antenna_gain, horizontal_gains, vertical_gains):
    '''
    This function takes as input both horizontal and vertical converted
    angles as well as their associated gains. The aim is to reconstruct in
    3D the radiation pattern, given its .msi file.
    The function outputs the antenna gain on both angles.

    Inputs:
    - antenna_gain (float) -> The gain of the antenna, read from the .msi file
    - horizontal_gains (tf.constant) -> A file containing the gains from the
        horizontal plane of the antenna.
    - vertical_gains (tf.constant) -> A file containing the gains from the
        vertical plane of the antenna.

    Outputs:
    - 
    '''

    # Convert gains from dB to linear scale
    hor_gains_linear = 10 ** (horizontal_gains / 10)
    ver_gains_linear = 10 ** (vertical_gains / 10)

    # Initialize a matrix for storing gain values
    G_w_theta_phi = np.zeros((len(horizontal_angles), len(vertical_angles)))
    print(f"Horizontal gains: {horizontal_gains}")
    print(f"Vertical gains: {vertical_gains}")

    # constant to check the first values of both vertical and horizontal planes
    a = 0

    # Loop through horizontal and vertical angles
    for i, horizontal_gain in enumerate(horizontal_angles):
        for j, vertical_gain in enumerate(vertical_angles):

            if a <= 3:
                print(f"These are the gain values for the HORIZONTAL plane: {horizontal_gain}")
                print(f"These are the gain values for the VERTICAL plane: {vertical_gain}")
                a += 1

            # Normalize linear gains
            g_h_phi_i = (hor_gains_linear[i] - np.min(hor_gains_linear)) / (np.max(hor_gains_linear) - np.min(hor_gains_linear))
            g_v_theta_i = (ver_gains_linear[j] - np.min(ver_gains_linear)) / (np.max(ver_gains_linear) - np.min(ver_gains_linear))

            # check that both values are between 0 and 1
            if (g_h_phi_i>1 or g_h_phi_i<0) and (g_v_theta_i>1 or g_v_theta_i<0):
                print('One of the normalized values is not inside (0, ..., 1)')
                exit()

            # Compute weight functions
            w1_theta_phi_i = g_v_theta_i * (1 - g_h_phi_i)
            w2_theta_phi_i = g_h_phi_i * (1 - g_v_theta_i)

            # Convert normalized linear gains back to dB
            G_h_phi_dB_i = 10 * np.log10(g_h_phi_i)
            G_v_theta_dB_i = 10 * np.log10(g_v_theta_i)

            # Compute weighted antenna gain
            k = 2  # Normalization parameter
            G_w_theta_phi[i, j] = (G_h_phi_dB_i) * w1_theta_phi_i/((w1_theta_phi_i**k + w2_theta_phi_i**k)**(1/k)) + (G_v_theta_dB_i) * w2_theta_phi_i / (
                (w1_theta_phi_i**k + w2_theta_phi_i**k)**(1/k)
            )
    print(f"These are the values for i and j: {i}, {j}")
    print(f"This is the shape of G_w_theta_phi: {np.shape(G_w_theta_phi)}")
    return G_w_theta_phi

if __name__ == '__main__':
    # Load the data from the file
    path = '80010465_0791_x_co.msi'

    # let us read the given .msi file
    antenna_gain, horizontal_gains, vertical_gains = msi_file_reader(path)
    print(f'This is the antenna GAIN: {antenna_gain}')
    print(f'These are the HORIZONTAL GAINS of the antenna: {horizontal_gains}')
    print(f'These are the VERTICAL GAINS of the antenna: {vertical_gains}')

    # only select gains associated to angles liying on the positive side of the x plane, i.e., 1st and 4th quadrant
    


    '''# Call the cross-weighted algorithm
    weighted_antenna_gains = cross_weighted_algorithm(
        horizontal_angles_converted,
        vertical_angles_converted,
        horizontal_gains,
        vertical_gains
    )

    # Create a mesh grid for azimuth and elevation angles
    azimuth, elevation = np.meshgrid(horizontal_angles_converted, vertical_angles_converted, indexing='ij')

    # Convert spherical to Cartesian coordinates for plotting
    x = weighted_antenna_gains * np.sin(elevation) * np.cos(azimuth)
    y = weighted_antenna_gains * np.sin(elevation) * np.sin(azimuth)
    z = weighted_antenna_gains * np.cos(elevation)

    # Plotting the 3D radiation pattern
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(x, y, z, cmap='jet', edgecolor='k', alpha=0.6)

    # Add color bar with dB label and title formatting
    colorbar = fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10)
    colorbar.set_label("Gain (dB)")

    # Set title and labels
    ax.set_title(r"3D Radiation Pattern $G(\theta, \varphi)$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")

    plt.show()
'''