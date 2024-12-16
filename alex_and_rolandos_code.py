import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
import sys

import tensorflow as tf
from scipy.interpolate import interp1d
    
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera

from sionna.rt.antenna import Antenna, visualize
from sionna.constants import PI
from sionna.rt.antenna import polarization_model_2, polarization_model_1


# let us define our functions
def split_values(data):
    column2 = []
    for ii in data:
        temp1, temp2 = ii.strip().split()
        column2.append(float(temp2))

    np_column2 = np.array(column2)
    return np_column2
    
def importar_msi(archivo):
    with open(archivo, 'r') as f:
        lineas = f.readlines() # Reading all lines
        horizontal_data = split_values(lineas[6:366]) # processing h_data
        vertical_data = split_values(lineas[367:727]) # processing v_data

        temp, freq = lineas[1].strip().split()        # taking the frequency in MHz
        freq = float(freq)

        temp, gain, unit_dBd  = lineas[2].strip().split() # taking the gain in dBd (unit_dBd contains the unit)
        gain = float(gain) + 2.15 # converting to dBi
        
    # Regular expression
    return horizontal_data, vertical_data, freq, gain

#READING MSI FILE
archivo_msi = '80010465_0791_x_co.msi'
horizontal, vertical, freq, gain = importar_msi(archivo_msi)

# Taking the positive x-side of the vertical plane, 1st and 4th quadrant: 
temp1 = vertical[0:91]
temp1 = temp1[::-1]
temp2 = vertical[270:360]
temp2 = temp2[::-1]
vertical_x_side = np.concatenate((temp1, temp2))
vertical = vertical_x_side

#Data in msi file is attenuation in dB, so the real gain is: G(angles) = Gain - MSI_DATA(angles)
g_hor_dB = gain - horizontal
g_ver_dB = gain - vertical #Remember, we selected 181 angles corresponding to first and fourth quadrant

# The normalized values in dB will be -MSI_DATA(angles)
g_hor_dB_n = - horizontal
g_ver_dB_n = - vertical 

#normalized gain from dB to lineal. Normalized values have been utilized, the gain factor will be added at the end.
gain_v = 10**(g_ver_dB_n/10)
gain_h_temp = 10**(g_hor_dB_n/10) #A temporal variable has been used because we are going to shift the angles to [-pi,pi)

theta = np.linspace(0, 2*np.pi, 360)
# Shift the theta values to range from -pi to pi
theta_shifted = np.where(theta > np.pi, theta - 2*np.pi, theta)
sorted_indices = np.argsort(theta_shifted)
gain_h = gain_h_temp[sorted_indices]

# Once angles and gain in the MSI file has been parse to be aligned with the spherical coordinate 
# we can work with the variables (theta,phi).
theta_rad = np.linspace(0, np.pi, 181)  # Zenith angles
phi_rad = np.linspace(-np.pi, np.pi, 360)  # Azimuth angles

# Interpolation of each plane: It is required because we only have data with a resolution of one degree from the MSI file
interp_v = interp1d(theta_rad, gain_v, kind='linear', fill_value="extrapolate")
interp_h = interp1d(phi_rad, gain_h, kind='linear', fill_value="extrapolate")

# Custom antenna pattern function to be used in Sionna
def custom_pattern(theta, phi, slant_angle=0.0, polarization_model=2, dtype=tf.complex64):

    rdtype = dtype.real_dtype
    k_dB = gain  # IMPORTANT: the gain should be in dBi not in dBd (previous conversion has been done)
    #k_dB = 0    # Uncomment if you want to normalize the radiattion pattern
    k_dB = tf.cast(k_dB,rdtype)    
    k = 10**(k_dB/10)    
    theta = tf.cast(theta, rdtype)
    phi = tf.cast(phi, rdtype)
    slant_angle = tf.cast(slant_angle, rdtype) # This angle is associate with the polarization

    # Common for SIONNA: (IMPORTANTE) same shape does not mean same intervals: (0,pi) and (-pi,pi)
    if not theta.shape== phi.shape: 
        raise ValueError("theta and phi must have the same shape.")
    if polarization_model not in [1,2]:
        raise ValueError("polarization_model must be 1 or 2")

    
    #Use the interpolator to create a function for each plane
    gain_theta = tf.py_function(func=interp_v, inp=[theta], Tout=tf.float32)
    gain_phi = tf.py_function(func=interp_h, inp=[phi], Tout=tf.float32)

    # Estimation of the 3D pattern using summing algorithms: G_dBn(theta,phi) = G_dBn(theta) + G_dBn(phi)
    gain_theta_dB_n = 10.0 * tf.math.log(gain_theta) / tf.math.log(10.0)
    gain_phi_dB_n = 10.0 * tf.math.log(gain_phi) / tf.math.log(10.0)
    gain_theta_phi_dB_n = gain_theta_dB_n + gain_phi_dB_n

    
    #Converting from dBn to linear
    g_theta_phi = tf.pow(10.0, gain_theta_phi_dB_n / 10.0) #normalized gain for each per (theta, phi) 
    
    # The values from the MSI files are gain/attenuation values (power)
    # We need to convert from power to field components F = sqrt(P)    
    temp = tf.sqrt(k)*tf.sqrt(g_theta_phi)             
    c = tf.complex(temp, tf.zeros_like(temp))

    # Common step in sionna, this functions obtain the Field components based on the slant angle
    if polarization_model==1:
        return polarization_model_1(c, theta, phi, slant_angle)
    else:
        return polarization_model_2(c, slant_angle)
    

antenna = Antenna(pattern=custom_pattern, polarization="H", dtype=tf.complex64)

fig_v, fig_h, fig_3d = visualize(custom_pattern)


