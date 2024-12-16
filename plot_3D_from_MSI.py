import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def msi_file_reader(msi_path):
    """
    This function reads the most relevant information out of an .msi file
    to further reconstruct the 3D radiation pattern of a given antenna

    Input:
    - msi_path (str) --> A sting containing the path to the .msi file

    Otuputs:
    - horizontal_angles ()
    - horizontal_gains ()
    - vertical_angles ()
    - vertical_gains ()
    """

    # Initialize lists to hold the angles and gains for horizontal and vertical patterns
    # horizontal_angles = []
    horizontal_gains = []
    # vertical_angles = []
    vertical_gains = []

    # Parse the .msi file
    with open(msi_path, 'r') as file:
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
                # horizontal_angles.append(np.radians(angle))
                horizontal_gains.append(gain)
            elif vertical_section:
                _, gain = map(float, line.strip().split())
                # if angle < 180:  # Only process angles less than or equal to 180 degrees
                    # vertical_angles.append(np.radians(angle))
                vertical_gains.append(gain)

        # test if vertical gains and vertical angles have the same shape or not 
        '''if len(vertical_angles) == len(vertical_gains):
            print(f"Both lists (ANGLES & GAINS) on the VERTICAL plane have the same length")'''

        # repeat for the horizontal plane
        '''if len(horizontal_angles) == len(horizontal_gains):
            print(f"Both lists (ANGLES & GAINS) on the HORIZONTAL plane have the same length")'''

        # Convert horizontal angles to a NumPy array and adjust range to [-pi, pi]
        # horizontal_angles = np.array(horizontal_angles)
        # horizontal_angles_converted = np.where(horizontal_angles > np.pi, horizontal_angles - 2 * np.pi, horizontal_angles)
        # horizontal_gains_2 = [3.10 + x for x in horizontal_gains]
        
        horizontal_gains_2 = [(3.10 + x) if (3.10 + x) <= 3.10 else (3.10 - x) for x in horizontal_gains]
        horizontal_gains = np.array(horizontal_gains_2)

        # Convert vertical angles to a NumPy array and adjust range to [0, pi]
        vertical_angles = np.array(vertical_angles)
        vertical_angles_converted = np.where(vertical_angles > np.pi, 2 * np.pi - vertical_angles, vertical_angles)
        # vertical_gains_2 = [3.10 + x for x in vertical_gains]
        vertical_gains_2 = [(3.10 + x) if (3.10 + x) <= 3.10 else (3.10 - x) for x in vertical_gains]
        vertical_gains = np.array(vertical_gains_2)


# Definition of the interpolation method
def cross_weighted_algorithm(horizontal_angles, vertical_angles, horizontal_gains, vertical_gains):
    '''
    This function takes as input both horizontal and vertical converted
    angles as well as their associated gains. The aim is to reconstruct in
    3D the radiation pattern, given its .msi file.
    The function outputs the antenna gain on both angles.
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

# Load the data from the file
path = '80010465_0791_x_co.msi'

# Call the cross-weighted algorithm
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
