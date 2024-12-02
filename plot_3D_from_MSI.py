import numpy as np
import matplotlib.pyplot as plt
from matplotlib import _cm
from mpl_toolkits.mplot3d import Axes3D


# definition of the interpolation method
def cross_weighted_algorithm(horizontal_angles, vertical_angles,
                             horizontal_gains, vertical_gains):

    '''
    This function takes as input, both horizontal and vertical converted
    angles as well as their associated gains. The aim is to reconstruct in
    3D the radiation pattern, given its .msi file.
    The function outputs the antenna gain on both angles
    '''

    # let us firstly transform our gains from dBs -> linear units
    hor_gains_linear = 10**(horizontal_gains/10)
    ver_gains_linear = 10**(vertical_gains/10)

    # let us normalize our linear gains
    g_h_phi = (hor_gains_linear - np.min(hor_gains_linear))/(np.max(hor_gains_linear) - np.min(hor_gains_linear))
    g_v_theta = (ver_gains_linear - np.min(ver_gains_linear))/(np.max(ver_gains_linear) - np.min(ver_gains_linear))

    # Check if all values are within the range [0, 1]
    if np.all((g_h_phi >= 0) & (g_h_phi <= 1)) and np.all((g_v_theta >= 0) & (g_v_theta <= 1)):
        print("The vector is populated with values ranging from 0 to 1.")
    else:
        print("The vector contains values outside the range [0, 1].")

    # let us now compute our weight functions
    w1_theta_phi = g_v_theta*(1-g_h_phi)
    w2_theta_phi = g_h_phi*(1-g_v_theta)

    # let us now convert our linear normalized gains into dBs
    G_h_phi_dB = 10*np.log10(g_h_phi)
    G_v_theta_dB = 10*np.log10(g_v_theta)

    # we now estimate the antenna gain by weighting the gains we computed
    k = 2
    G_w_theta_phi = (G_h_phi_dB*w1_theta_phi + G_v_theta_dB*w2_theta_phi)/((w1_theta_phi**(k)+w2_theta_phi**(k))**(1/k))
    return G_w_theta_phi


# Load the data from the file
path = 'Ejemplos/80010465/pattern/80010465_0791_x_co.msi'

# Initialize lists to hold the angles and gains for horizontal and vertical patterns
horizontal_angles = []
horizontal_gains = []
vertical_angles = []
vertical_gains = []

# Parse the .msi file
with open(path, 'r') as file:
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
            angle, gain = map(float, line.strip().split())
            horizontal_angles.append(np.radians(angle))
            horizontal_gains.append(gain)
        elif vertical_section:
            angle, gain = map(float, line.strip().split())
            vertical_angles.append(np.radians(angle))
            vertical_gains.append(gain)
            '''if angle <= 180:  # This was <= instead of < (Only process angles less than or equal to 180 degrees)
                vertical_angles.append(np.radians(angle))
                vertical_gains.append(gain)'''

print(f"Number of HORIZONTAL angles: {len(horizontal_angles)}")
print(f"Number of VERTICAL angles: {len(vertical_angles)} MAX_value: {np.max(vertical_angles)} MIN_val: {np.min(vertical_angles)}")

# turn HORIZONTAL (phi) angle values into np.array
horizontal_angles = np.array(horizontal_angles)
print(f"We are now checking the horizontal angles: {horizontal_angles}")
# convert from [0, 2*pi] to [-pi, pi]
horizontal_angles_converted = np.where(horizontal_angles > np.pi, horizontal_angles - 2*np.pi, horizontal_angles)
# compute the horizontal gains
horizontal_gains = np.array(horizontal_gains)
print(f"We are now checking the CONVERTED horizontal angles: {horizontal_angles_converted}")

# turn VERTICAL (theta) angle values into np.array
vertical_angles = np.array(vertical_angles)
print(f"We are now checking the vertical angles: {vertical_angles}")
# convert from [0, 2*pi] to [0, pi]
vertical_angles_filtered = np.delete(vertical_angles, np.where(vertical_angles > 180))
vertical_angles_converted = np.where(vertical_angles > np.pi, 2*np.pi - vertical_angles, vertical_angles)
vertical_gains = np.array(vertical_gains)
print(f"We are now checking the CONVERTED vertical angles: {vertical_angles_converted}")

# let us now estimate the antenna gain using the cross_weighted algorithm depicted in some papers
weighted_antenna_gains = cross_weighted_algorithm(horizontal_angles_converted,
                                                  vertical_angles_converted,
                                                  horizontal_gains,
                                                  vertical_gains)

# let us check the outputs we got from the applied algorithm
print(f"This is the type of WEIGHTED_ANTENNA_GAINS: {type(weighted_antenna_gains)}")
print(f"This is the NUMBER of elements inside of WEIGHTED_ANTENNA_GAINS: {len(weighted_antenna_gains)}")
print(f"These are the cross-weighted antenna values: {weighted_antenna_gains}")


# Create a grid for azimuth and elevation angles
azimuth, elevation = np.meshgrid(horizontal_angles_converted, vertical_angles_converted, indexing='ij') # default indexing is 'xy'
# radii = np.abs(vertical_gains)**2 + np.abs(horizontal_gains)**2
# radii = 10 * np.log10(radii)

# Convert spherical to Cartesian coordinates for plotting
x = weighted_antenna_gains * np.sin(elevation) * np.cos(azimuth)
y = weighted_antenna_gains * np.sin(elevation) * np.sin(azimuth)
z = weighted_antenna_gains * np.cos(elevation)

# Plotting the 3D radiation pattern
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(x, y, z, cmap='jet', edgecolor='k', alpha=0.6)

# Color bar with dB label and title formatting
colorbar = fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10)
colorbar.set_label("dB")

# Set title and labels with LaTeX formatting
ax.set_title(r"3D visualization of the radiation pattern $G(\theta, \varphi)$")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_zlabel(r"$z$")

plt.show()
