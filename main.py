# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 19:19:21 2024

@author: alexe
"""
#   TODO LIST
# move the parameter definitions with explanations to another file
# use package like sdeint
# learn more and implement the right array types
# implement 3D equations
# check that the noise floor is right and make other checks
# implement gaussian tweezer
# implement tightly focused tweezer

#%% perform calculations
import numpy as np
import matplotlib.pyplot as plt
from define_constants import  c, hbar, kB
from scipy.signal import welch

from define_parameters import   Tgas, Mgas,Gamma_tot, T_cm, A, omega_x, omega_y, omega_z, M, omegas, k, wx, wy, zR
from define_functions import plottracefun, psdplot, lorentz, plot_velocity_histogram_and_fit, fit_offsets, FgradX, FgradY, FgradZ, E0sq, FscattX, FscattY, FscattZ
import time

#%% #%% plot the grad force and potential
xvec = np.linspace(-5*wx,5*wx,int(1e3))
yvec = np.zeros(int(1e3))
zvec = np.linspace(-5*zR,5*zR,int(1e3))


fvecx = FgradX(xvec,yvec,yvec)
fvecy = FgradY(yvec,xvec,yvec)
fvecz = FgradZ(yvec,yvec,zvec)
fscx = FscattX(xvec,yvec,yvec)
fscz = FscattZ(yvec,yvec,xvec)


plt.figure()
plt.plot(xvec,fvecx,'b')
plt.plot(xvec,-M*omega_x**2*xvec,'b')
plt.plot(xvec,fscx,'k')
plt.ylim(np.min(fvecx),np.max(fvecx))
plt.xlabel('x [m]')
plt.ylabel(r'$F_{grad}(x)$ [N]')
plt.legend(['gradient force of a gaussian beam','linear approximation'])

plt.figure()
plt.plot(xvec,fvecy,'b')
plt.plot(xvec,-M*omega_y**2*xvec,'b')
plt.ylim(np.min(fvecy),np.max(fvecy))
plt.xlabel('y [m]')
plt.ylabel(r'$F_{grad}(y)$ [N]')
plt.legend(['gradient force of a gaussian beam','linear approximation'])


plt.figure()
plt.plot(zvec,fvecz,'b')
plt.plot(zvec,-M*omega_z**2*zvec,'b')
plt.ylim(np.min(fvecz),np.max(fvecz))
plt.xlabel('z [m]')
plt.ylabel(r'$F_{grad}(z)$ [N]')
plt.legend(['gradient force of a gaussian beam','linear approximation'])
plt.plot(zvec,fscz,'k')





#%% perform simulation
######################### Simulation parameters################################


import sdeint

def a(Xt, t):
    X, Y, Z, Vx, Vy, Vz = Xt
    dXdt = Vx
    dYdt = Vy
    dZdt = Vz
    dVxdt = FgradX(X,Y,Z)/M - Gamma_tot * Vx
    dVydt = FgradY(X,Y,Z)/M - Gamma_tot * Vy
    dVzdt = FgradZ(X,Y,Z)/M + 0.*FscattZ(X,Y,Z)/M - Gamma_tot * Vz
    return np.array([dXdt, dYdt, dZdt, dVxdt, dVydt, dVzdt])

def b(Xt, t):
    # Assuming A is defined and represents the intensity of the noise affecting each velocity component
    # Return a matrix with noise affecting Vx, Vy, Vz
    return np.diag([0, 0, 0, A, A, A])

# Simulation parameters
N = int(2**15)#65536
Fs=1e6
dt = 1/Fs
T = N * dt
t = np.linspace(0, T, N+1)
NumSim = 1  # Adjust for multiple simulations

# Initial conditions
mu_x = mu_y = mu_z =  0
mu_vx = mu_vy = mu_vz = 0
# Preallocate arrays for efficiency
Xem = np.zeros((NumSim, len(t), 3))  # Now storing X, Y, Z
Vem = np.zeros((NumSim, len(t), 3))  # Storing Vx, Vy, Vz

for i in range(NumSim):
    print(i)
    X0 = np.array([mu_x, mu_y, mu_z, mu_vx, mu_vy, mu_vz])  # Initial conditions for each simulation
    np.random.seed(int(time.time()))
    result = sdeint.itoint(a, b, X0, t)
    Xem[i, :, :] = result[:, :3]  # Extract positions
    Vem[i, :, :] = result[:, 3:]  # Extract velocities
    


#%%

################## PLOT RESULTS ##############################################
# Placeholder for plottracefun and psdplot functions
# You should replace these with actual function calls or implementations
# subtract offset
offsets = fit_offsets(Xem)

dim = 2
poptx,f,Sxx = plottracefun(t, Xem, Vem, 0, 0, T, dt, 1/T, 1/dt, False,omegas)
popty,f,Syy = plottracefun(t, Xem, Vem, 0, 1, T, dt, 1/T, 1/dt, False,omegas)
poptz,f,Szz = plottracefun(t, Xem, Vem, 0, 2, T, dt, 1/T, 1/dt, False,omegas)
popt=[poptx,popty,poptz]
# f, psdx, popt2 = psdplot(Xem, 1/dt, "position",dim, "m", "-b", 1e3, 4e5, False)
# plot_velocity_histogram_and_fit(Vem, NumSim-1, dim)

# Sxxth = lorentz(2*np.pi*f,Tgas,Gamma_tot,0,omega_x)
# plt.figure(5)
# plt.loglog(f,Sxxth)
# plt.loglog(f,Sxx,'-r')



# check equipartition theorem
# Parameters

# Integrate the PSD over all frequencies
# remember that the twosided fft = 1/2 onesided fft in terms of total power
total_power = 0.5*M*omega_z**2*np.trapz(Szz, f)
total_power_th = M*omega_z**2*np.trapz(lorentz(2*np.pi*f, T_cm, Gamma_tot, omega_z), 2*np.pi*f)

# Expected total power (for one degree of freedom: 1/2 k_B T)
expected_power = 0.5* kB * T_cm

print(f"Total integrated power: {total_power}")
print(f"Expected power: {expected_power}")
print(f"Total integrated power in theory: {total_power_th}")



#%% Plot for the review article
from define_functions import plot_with_inset_histogram
from matplotlib.gridspec import GridSpec
from define_functions import plot_psd

# Assuming Xem, Vem, and t are defined somewhere in your script
# For demonstration, let's create some dummy data
Np = N
Npi = N
dim=2
# Create figure and gridspec
fig = plt.figure(figsize=(14, 12))
gs = GridSpec(2, 2, height_ratios=[3, 3], width_ratios=[5, 1], hspace=0.3, wspace=0.05)

# Plot for Xem with scaling to micrometers
plot_with_inset_histogram(fig, gs[0, 0], gs[0, 1], Xem, t, Np, Npi, 'Position x of a nanosphere trapped in an optical tweezer', 'Position (μm)', dim, "X", scale_y=1e6, yunit='μm', fontsize=16)

# Plot for Vem without scaling for y (velocity remains in m/s)
plot_with_inset_histogram(fig, gs[1, 0], gs[1, 1], Vem, t, Np, Npi, r'Velocity $v_{x}$ of a nanosphere trapped in an optical tweezer', 'Velocity (m/s)', dim, "V", scale_y=1, yunit='m/s', fontsize=16)

popt=np.array([0, 0, 0])
plt.show()
plot_psd(Xem, dt,popt)
#plt.xlim(0,4e5)



#%% save data
import pickle

# Dictionary containing only Xem and Vem
variables_to_save = {'Xem': Xem, 'Vem': Vem}

# Save the dictionary to a file
with open('variables.pkl', 'wb') as f:
    pickle.dump(variables_to_save, f)
#%% load variables again

import pickle

# Load the dictionary from the file
with open('variables.pkl', 'rb') as f:
    loaded_variables = pickle.load(f)

# Assign the loaded variables back to their names
Xem = loaded_variables['Xem']
Vem = loaded_variables['Vem']
