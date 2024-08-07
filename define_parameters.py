# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 19:24:31 2024

@author: alexe
"""
# lambda_laser, k, omega, wx, wy, zR, Gamma_tot, T_cm, A, omega_x, omega_y, omega_z
import numpy as np
from define_constants import c,eps0, kB, hbar

# Define laser parameters
lambda_laser = 1560e-9  # Laser wavelength in meters
k = (2 * np.pi) / lambda_laser  # Corresponding k-number
omega = 2 * np.pi * c / lambda_laser  # Laser angular frequency
wx = 1.8e-6#1.36537e-6  # Laser beam waist in meters
wy = 1.8e-6#1.46372e-6 # laser beam waist in meters
zR = np.pi*wx*wy/(lambda_laser)#3.67946e-6 #np.pi*wx*wy/(lambda_laser)
P0 = 0.4  # Input power in watts
I0 = 2*P0 / (np.pi * wx*wy)  # peak intensity at waist but average over one cycle! 1/2 factor
E0 = np.sqrt(2*I0/(c*eps0)) # peak electric field at waist
# Define trapped particle parameters
n = 1.4439 # Refractive index
R = 170e-9 / 2  # Radius of the sphere in meters
V = (4/3) * np.pi * R**3  # Volume of the sphere
rho = 2200  # Density of glass in kg/m^3
M = rho * V  # Mass of the particle
# Polarizability alpha for a homogeneous sphere, using the Clausius-Mossotti formula
alpha = 3 * V * eps0 * ((n**2 - 1) / (n**2 + 2))
alpha_comp = alpha/(1-1j*alpha*k**3/(6*np.pi*eps0))
alpha_p = np.real(alpha_comp)
alpha_pp = np.imag(alpha_comp)
sigma = (k**4 * alpha_p**2) / (6 * np.pi * eps0**2)  # Scattering cross section
Pdip = sigma * I0  # Scattered power

# Define gas parameters
Tgas = 300  # Background gas temperature in Kelvin
Pgas = 100  # Background gas pressure in Pascals
Mgas = 28 * 1.66054e-27  # Mass of gas molecules (assuming nitrogen here) in kg
vrms = np.sqrt(3 * kB * Tgas / Mgas)  # RMS speed of gas molecules
d = 372e-12  # Kinetic diameter of nitrogen molecules in meters
l = kB * Tgas / (np.sqrt(2) * np.pi * d**2 * Pgas)  # Mean free path of gas molecules
eta = Pgas * l * np.sqrt(2 * Mgas / (kB * Tgas * np.pi))  # Gas viscosity

Kn = l / R  # Knudsen number
cK = 0.31 * Kn / (0.785 + 1.152 * Kn + Kn**2)  # Correction factor for non-continuum effects
######################### Damping constants ######################################
Gamma_gas = (6*np.pi*eta*R/M)*0.619/(0.619+Kn)*(1+0.31*Kn/(0.785+1.152*Kn+Kn**2))
Gamma_rad = 2/5*Pdip/(2*np.pi*M*c**2)
Gamma_tot = Gamma_rad + Gamma_gas

# Force spectral densities
S_gas = M*kB*Tgas*Gamma_gas/np.pi
S_rad = 2/5*hbar*np.pi*Pdip/(2*np.pi*c)
S_tot = S_gas + S_rad

# Center of motion temperature and total thermal acceleration
T_cm = np.pi*S_tot/(kB*M*Gamma_tot)
A = np.sqrt(2*kB*Tgas*Gamma_tot/M)
# Natural oscillation frequency
omega_x = np.sqrt(3*(n**2-1)/(n**2+2)*4*P0/(np.pi*rho*c*wx**3*wy))
omega_y = np.sqrt(3*(n**2-1)/(n**2+2)*4*P0/(np.pi*rho*c*wy**3*wx))
omega_z = np.sqrt(3/2*(n**2-1)/(n**2+2)*4*P0/(np.pi*rho*c*wy*wx*zR**2))
omegas = np.array([omega_x, omega_y, omega_z])
##################################################################################