import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io
from matplotlib import cm

def summing_algorithm(horizontal_angles, vertical_angles, horizontal_gains, vertical_gains):
    '''
    This function takes as input both horizontal and vertical converted
    angles as well as their associated gains. The aim is to reconstruct in
    3D the radiation pattern, given its .msi file.
    The function outputs the antenna gain on both angles.
    '''
    # normalize 
    # horizontal_gains = horizontal_gains - np.max(horizontal_gains)
    # vertical_gains = vertical_gains - np.max(vertical_gains)

    # Initialize a matrix for storing gain values
    G_w_theta_phi = np.zeros((len(horizontal_angles), len(vertical_angles)))

    for i, horizontal_gain in enumerate(horizontal_angles):
        for j, vertical_gain in enumerate(vertical_angles):
            G_w_theta_phi[i,j] = horizontal_gains[i] + vertical_gains[j]

    return G_w_theta_phi

def plot3D (weighted_antenna_gains):
    # Create a mesh grid for azimuth and elevation angles
    theta = np.linspace(0.0, np.pi, 181)
    theta = np.flip(theta)
    theta = np.roll(theta,90)

    phi = np.linspace(0, 2*np.pi, 360)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
    theta_grid = np.transpose(theta_grid)
    phi_grid = np.transpose(phi_grid)

    # g = np.transpose(10**(MATLAB_3D_pattern/10))
    g = weighted_antenna_gains
    
    print("This is the shape of the summing resulting gain: ", {np.shape(g)})

    x = g * np.sin(theta_grid) * np.cos(phi_grid)
    y = g * np.sin(theta_grid) * np.sin(phi_grid)
    z = g * np.cos(theta_grid)
     
    # g = np.maximum(g, 1e-6)
    g_db = 10*np.log10(g)
     
    def norm(x, x_max, x_min):
        """Maps input to [0,1] range"""
        x = 10**(x/10)
        x_max = 10**(x_max/10)
        x_min = 10**(x_min/10)
        if x_min==x_max:
            x = np.ones_like(x)
        else:
            x -= x_min
            x /= np.abs(x_max-x_min)
        return x

    g_db_min = np.min(g_db)
    g_db_max = np.max(g_db)
     
    fig_3d = plt.figure()
    ax = fig_3d.add_subplot(1,1,1, projection='3d')
    ax.plot_surface(x, y, z, rstride=1, cstride=1, linewidth=0,
                    antialiased=False, alpha=0.7,
                    facecolors=cm.turbo(norm(g_db, g_db_max, g_db_min)))
    # plt.xlim([-1, 1]) 
     
    sm = cm.ScalarMappable(cmap=plt.cm.turbo)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", location="right",
                        shrink=0.7, pad=0.15)
    xticks = cbar.ax.get_yticks()
    xticklabels = cbar.ax.get_yticklabels()
    xticklabels = g_db_min + xticks*(g_db_max-g_db_min)
    xticklabels = [f"{z:.2f} dB" for z in xticklabels]
    cbar.ax.set_yticks(xticks)
    cbar.ax.set_yticklabels(xticklabels)
     
    ax.view_init(elev=30., azim=-45)
    plt.xlabel("x")
    plt.ylabel("y")
    ax.set_zlabel("z")
    plt.suptitle(r"3D visualization of the radiation pattern $G(\theta,\varphi)$")

# Load the data from the file
path = '80010465_0791_x_co.msi'

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
            # if angle < 180:  # Only process angles less than or equal to 180 degrees
            vertical_angles.append(np.radians(angle))
            vertical_gains.append(gain)

# test if vertical gains and vertical angles have the same shape or not 
if len(vertical_angles) == len(vertical_gains):
    print(f"Both lists (ANGLES & GAINS) on the VERTICAL plane have the same length")

# repeat for the horizontal plane
if len(horizontal_angles) == len(horizontal_gains):
    print(f"Both lists (ANGLES & GAINS) on the HORIZONTAL plane have the same length")

# Convert horizontal angles to a NumPy array and adjust range to [-pi, pi]
horizontal_angles = np.array(horizontal_angles) 
horizontal_gains = np.array(horizontal_gains) 


# Convert vertical angles to a NumPy array and adjust range to [0, pi]
vertical_angles = np.array(vertical_angles) 
vertical_angles = np.append(vertical_angles[0:91],vertical_angles[270:360])

vertical_gains = np.array(vertical_gains)
vertical_gains = np.append(vertical_gains[0:91],vertical_gains[270:360])



# Call the summing algorithm
weighted_antenna_gains = summing_algorithm(
    horizontal_angles,
    vertical_angles,
    horizontal_gains,
    vertical_gains
)

weighted_antenna_gains = 3.1 - weighted_antenna_gains 
weighted_antenna_gains = 10**(weighted_antenna_gains/10)

plot3D(weighted_antenna_gains)