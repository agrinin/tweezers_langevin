# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 19:41:30 2024

@author: alexe
"""
import matplotlib.pyplot as plt
from scipy.signal import welch
import numpy as np
from scipy.optimize import curve_fit
from define_constants import kB
from scipy.stats import norm
from define_parameters import M, omegas, Pgas, T_cm, wx, wy, zR, alpha, E0, alpha_p,alpha_pp, k
from scipy.integrate import quad
from scipy.special import jv
from functools import partial



def plottracefun(t, Xem, Vem, numtr, dim, Tmax,dt, Fmin, Fmax, plotswitch,omegas):

    if dim == 0:
        str_label = "x"
    elif dim == 1:
        str_label = "y"
    elif dim == 2:
        str_label = "z"

    # Plotting position and velocity
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(t, Xem[numtr, :,dim], 'r-')  # Adjusted indexing for Python (0-based)
    plt.grid(True)
    plt.xlabel('time [s]', fontsize=16)
    plt.ylabel(f'position {str_label}[m]', fontsize=16)
    plt.xlim([0, Tmax])

    plt.subplot(2, 1, 2)
    plt.plot(t, Vem[numtr, :,dim], 'b-')  # Adjusted indexing for Python (0-based)
    plt.xlim([0, Tmax])
    plt.grid(True)
    plt.xlabel('time [$\mu$s]', fontsize=12)
    plt.ylabel(f'{str_label}-velocity [m/s]', fontsize=16)
    plt.show()
    T_guess = plot_velocity_histogram_and_fit(Vem, numtr, dim)
    # PSD calculation
    plt.figure(figsize=(12, 8))
    Fs = 1 / dt  # Sampling frequency

    # Define segment length and number of FFT points (zero padding)
    nperseg = 2**14  # Segment length
    nfft = 2**14  # Number of FFT points, with zero padding

    # Calculate PSD with zero padding
    f, Sxx = welch(Xem[numtr, :, dim], Fs, nperseg=nperseg, nfft=nfft)
    f, Svv = welch(Vem[numtr, :, dim], Fs, nperseg=nperseg, nfft=nfft)


    # Position PSD
    plt.subplot(2, 1, 1)
    plt.loglog(f, Sxx, 'b-', label='Original PSD')
    # cakculate normalized variables and parameters
    omega =2*np.pi*f
    
    # Initial guesses based on the PSD characteristics
    omega_guess = omega[np.argmax(Sxx)]  # Frequency at max value
    
    # Estimating FWHM (simplified approach)
    half_max = np.max(Sxx) / 2
    indices_above_half = np.where(Sxx > half_max)[0]
    if len(indices_above_half) > 0:
        fwhm_guess = omega[indices_above_half[-1]] - omega[indices_above_half[0]]
    else:
        fwhm_guess = omega[1] - omega[0]  # Minimal non-zero frequency interval if no peak is found
    Gamma_guess = 5*fwhm_guess
    #change fn and Sxx to a small region around the max
    # Determine the start and end indices, ensuring they are within the array bounds
    max_index = np.argmax(Sxx)
    start_index = max(0, max_index - 100)
    end_index = min(len(Sxx), max_index + 100 + 1)  # +1 to include the element at position max_index + 100
    omegan=omega[start_index:end_index]
    Sxxn=Sxx[start_index:end_index]
    initial_guesses = [T_guess, Gamma_guess, omega_guess]

    # Define bounds for the parameters: (T, Gamma0, S_imp, omega_c)
    # Assuming T, Gamma0, and omega_c can vary freely within reasonable physical limits
    # S_imp is restricted to be >= 0
    
    max_function_evaluations = 10000  # For example, limit to 1000 evaluations
    # Perform the curve fitting with a limit on the number of function evaluations
    popt, pcov = curve_fit(lorentz, omegan, Sxxn, p0=initial_guesses, maxfev=max_function_evaluations)
    # T_int=0.5*M*popt[2]**2*np.trapz(Sxx, f)/kB
    # partial_lorentz = partial(lorentz, T=T_int)
    # popt2, pcov2 = curve_fit(partial_lorentz, omegan, Sxxn, p0=initial_guesses[1:2], maxfev=max_function_evaluations)

    Sxx_fit = lorentz(omega, *popt)

    plt.loglog(f, Sxx_fit, 'r-', label='Lorentzian Fit')
    
    plt.grid(True)
    plt.title(f'{str_label}-position PSD and Lorentzian Fit')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD')
    plt.legend()

    if Fmax < Fs / 2:
        plt.xlim([Fmin, Fmax])
    else:
        plt.xlim([Fmin, Fs / 2])

    # plot now Svv
    # plt.subplot(2, 1, 2)
    # plt.loglog(f, Svv, 'b-', label='Original PSD')


    plt.show()    
    return popt,f,Sxx
def psdplot(x, fs, name, dim, unit_x, disp, Fmin, Fmax, plotswitch):
    """
    Plots the Power Spectral Density (PSD) of a signal x(t).
    
    Parameters:
    - x: Input signal.
    - fs: Sampling frequency in Hz.
    - name: Name of the variable (for labeling purposes).
    - unit_x: Prefactor of the unit with respect to SI units (e.g., '1e-6' for microseconds).
    - disp: Display style for the plot (e.g., 'r-' for a red line).
    - plotswitch: Boolean indicating whether to plot the PSD.
    
    Returns:
    - freq: Frequencies at which the PSD is calculated.
    - psdx: PSD values.
    """
    from define_parameters import M
    from define_constants import kB
    if dim == 0:
        str_label = "x"
    elif dim == 1:
        str_label = "y"
    elif dim == 2:
        str_label = "z"
    N = len(x[0,:,dim])  # Number of points
    freq = np.linspace(0, fs/2, N//2 + 1)
    xdft = np.fft.fft(x[0,:,dim].flatten())
    xdft = xdft[:N//2 + 1]
    psdx = (1/(fs*N)) * np.abs(xdft)**2
    psdx[1:-1] = 2*psdx[1:-1]  # Compensate for the symmetry
    
    if plotswitch:
        plt.figure()
        plt.loglog(freq, psdx, disp)
        plt.xlim([freq[1], max(freq)])
        plt.title(f"Single-Sided Power Spectral Density of the {name} [{unit_x}]")
        plt.xlabel("f (Hz)")
        plt.ylabel(f"{name} PSD [{unit_x}^2/Hz]")
        plt.grid(True)
        plt.show()
        
    # Initial guesses based on the PSD characteristics
    omega_guess = 2*np.pi*freq[np.argmax(psdx)]  # Frequency at max value
    
    # Estimating FWHM (simplified approach)
    half_max = np.max(psdx) / 2
    indices_above_half = np.where(psdx > half_max)[0]
    if len(indices_above_half) > 0:
        fwhm_guess = freq[indices_above_half[-1]] - freq[indices_above_half[0]]
    else:
        fwhm_guess = freq[1] - freq[0]  # Minimal non-zero frequency interval if no peak is found
    Gamma_guess = fwhm_guess/(2*np.pi)
    T_guess = 1000#np.max(psdx)*M*np.pi*omega_guess**2*Gamma_guess/kB
    #amplitude_guess = T_guess*kB/(np.pi*M*fwhm_guess*center_freq_guess**2)
    initial_guesses = [T_guess, Gamma_guess, omega_guess]
    # Assuming lorentz is defined correctly to accept these parameters
    popt, pcov = curve_fit(lorentz, 2*np.pi*freq, psdx, p0=initial_guesses)
    Sxx_fit = lorentz(2*np.pi*freq, *popt)
    plt.loglog(freq, Sxx_fit, 'r-', label='Lorentzian Fit')
    
    plt.grid(True)
    plt.title(f'{str_label}-position PSD and Lorentzian Fit')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD')
    plt.legend()

    if Fmax < fs / 2:
        plt.xlim([Fmin, Fmax])
    else:
        plt.xlim([Fmin, fs / 2])


    return freq, psdx, popt
def lorentz_normalized(omega, T, Gamma0, S_imp, omega_c):

    """
    Normalized Lorentzian function.

    Parameters:
    - omega: Frequency (independent variable).
    - T: Temperature (or a scaling factor related to amplitude).
    - Gamma0: Damping coefficient.
    - S_imp: Impedance or offset in the PSD.
    - omega_c: Center frequency of the Lorentzian peak.

    Returns:
    - The value of the Lorentzian function at each frequency `omega`.
    """
    numerator =kB* T * Gamma0
    denominator = np.pi*M*((omega**2 - omega_c**2)**2 + (Gamma0 * omega)**2)
    return numerator / denominator + S_imp


def lorentz_normalized_wrapper(omega,T_guess,Gamma0, S_imp, omega_c):
    return lorentz(omega, T_guess, Gamma0, S_imp, omega_c)

def lorentz(omega, T, Gamma0, omega_c):
    from define_parameters import M
    kB = 1.380649e-23  # Boltzmann constant in J/K
    numerator = kB * T * Gamma0
    denominator = np.pi*M*((omega**2-omega_c**2)**2 + (Gamma0 * omega)**2)
    return numerator / denominator

# Wrapper function that includes T_guess as a fixed parameter
def lorentz_wrapper(omega, Gamma0, S_imp, omega_c, T_guess):
    return lorentz(omega, T_guess, Gamma0, S_imp, omega_c)

def plot_velocity_histogram_and_fit(velocities, numtr, dim_index):
    """
    Plots a histogram of velocities for a given dimension and fits a Gaussian distribution.
    
    Parameters:
    - velocities: A numpy array of shape (num_traces, num_points, 3) representing velocity vectors.
    - dim_index: The index for the dimension (0 for x, 1 for y, 2 for z).
    """
    # include packages
    from define_parameters import M
    from scipy.stats import norm
    from define_constants import kB
    # Flatten the velocity data for the specified dimension across all traces
    v_flat = velocities[numtr, :, dim_index].flatten()
    
    # Plot the histogram of velocities
    plt.figure(figsize=(10, 6))
    # Assuming v_flat is your flattened velocity data
    mean = np.mean(v_flat)
    std = np.std(v_flat)

    # Define the range from -5 to +5 standard deviations from the mean
    min_bin = mean - 5 * std
    max_bin = mean + 5 * std
    
    # Generate 301 bin edges for 300 bins within the specified range
    bins = np.linspace(min_bin, max_bin, 301)
    
    # Plot the histogram with the specified bins
    n, bins, patches = plt.hist(v_flat, bins=bins, density=True, alpha=0.6, color='g', label='Velocity Data')

    # Fit a Gaussian distribution to the data
    mu, std = norm.fit(v_flat)
    
    # Plot the PDF of the fitted Gaussian
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2, label='Fit: $\mu$ = %.2f, $\sigma$ = %.2f' % (mu, std))
    
    plt.xlabel('Velocity')
    plt.ylabel('Density')
    plt.title('Velocity Distribution and Gaussian Fit')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Print fitted parameters
    print(f"Fitted mean (mu): {mu}")
    print(f"Fitted standard deviation (sigma): {std}")
    print(f"Mass (M) from 'define_parameters.py': {M}")
    
    # Calculate the theoretical temperature
    T_theoretical = M * std**2 / kB
    print(f"Theoretical temperature: {T_theoretical} K")
    return T_theoretical
# Example usage:
# fs = 1000  # Sampling frequency in Hz
# t = np.arange(0, 1, 1/fs)  # Time vector
# x = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)  # Example signal
# psdplot(x, fs, 'Example Signal', '1', 'r-', True)

def fit_offsets(data):
    """
    Fits a straight line to each dimension of a (1, N, 3) shaped vector and returns the offsets.
    
    Parameters:
    - data: A numpy array of shape (1, N, 3).
    
    Returns:
    - A tuple of size 3 containing the offset (average value) for each of the three dimensions.
    """
    # Ensure data is correctly shaped
    if data.shape[2] != 3:
        raise ValueError("Data must be of shape (1, N, 3)")
    
    # Initialize an array to hold the offsets
    offsets = np.zeros(3)
    
    # Extract the N dimension length
    N = data.shape[1]
    
    # Generate an array of indices to use as the x values in the fitting
    x = np.arange(N)
    
    # Loop through each of the three dimensions
    for dim in range(3):
        # Extract the data for the current dimension
        y = data[0, :, dim]
        
        # Fit a straight line (polynomial of degree 1) to the data
        # polyfit returns the coefficients [slope, intercept]
        p = np.polyfit(x, y, 1)
        
        # The intercept (offset) is the second element
        offset = p[1]
        
        # Store the offset
        offsets[dim] = offset
    
    # Return the offsets as a tuple
    return tuple(offsets)




def plot_with_inset_histogram(fig, gs_main, gs_hist, data, t, Np, Npi, title, ylabel, dim, varXV, scale_y=1, yunit='', fontsize=16):

    # Extract the specified dimension from the data
    data_dim = data[:, :Np, dim] * scale_y

    # Main plot
    ax_main = fig.add_subplot(gs_main)
    ax_main.plot(t[:Np]*1e6, np.mean(data_dim, axis=0), label='Main plot')
    ax_main.set_xlabel('Time (μs)', color='black', fontsize=fontsize)
    ax_main.set_ylabel(ylabel, color='black', fontsize=fontsize)
    ax_main.tick_params(axis='both', labelsize=fontsize)
    
    # Histogram + Gaussian fit for the specified dimension
    ax_hist = fig.add_subplot(gs_hist)
    data_hist = data[:, :Npi, dim] * scale_y
    data_flat = data_hist.flatten()
    mean, std = np.mean(data_flat), np.std(data_flat)
    bins = np.linspace(mean - 5*std, mean + 5*std, 100)
    n, bins, patches = ax_hist.hist(data_flat, bins=bins, orientation='horizontal', color='gray', alpha=0.6, density=True)
    
    # Set the histogram's y-axis limits to match the main plot's y-axis limits
    ax_hist.set_ylim(ax_main.get_ylim())
    
    # Fit and plot Gaussian
    mu, std = norm.fit(data_flat)
    y = np.linspace(*ax_hist.get_ylim(), 100)
    p = norm.pdf(y, mu, std)
    ax_hist.plot(p, y, 'k', linewidth=2)
        
    if varXV == "X":
        T_theoretical = M * omegas[dim]**2 * (std*1e-6)**2 / kB
    elif varXV == "V":
        T_theoretical = M * std**2 / kB
        
    # Round the theoretical temperature to the nearest integer and format the text
    temp_text = f"T={int(round(T_theoretical))} K"
    
    # Annotate with a double arrow at one standard deviation and add text
    arrow_height = np.exp(-0.5) * max(p)  # Height at one std dev, using the peak of the Gaussian
    ax_hist.annotate('', xy=(arrow_height, mu + std), xytext=(arrow_height, mu - std),
                     arrowprops=dict(arrowstyle="<->", color='black', lw=2))
    # Adjust text orientation and position
    ax_hist.text(arrow_height * 1.3, mu, temp_text, fontsize=fontsize, rotation=-90, color='black', verticalalignment='center', horizontalalignment='right')
    
    ax_hist.axis('off')  # Hide axis for clarity
    ax_main.set_title(title, fontsize=fontsize)

    # Add horizontal two-sided arrow on the main plot
    ax_main.annotate('', xy=(1300, -0.8), xytext=(2600, -0.8),
                     arrowprops=dict(arrowstyle="<->", color='black', lw=2))
    ax_main.text(1900, -0.75, r'$\approx\frac{2\pi}{\Gamma}$', fontsize=22, color='black', ha='center')
    

def plot_psd(Xem, dt, popt):
    """
    Plots the Power Spectral Density (PSD) for each of the three components (x, y, z) of the input vector Xem.

    Parameters:
    - Xem: A NumPy array of shape (N, 3), where N is the number of samples, and the columns represent the x, y, and z components.
    - dt: Sampling interval in seconds.
    Returns:
    energy integrals
    E = [Ex, Ey, Ez]
    """

    plt.figure(figsize=(12, 8))
    Fs = 1 / dt  # Sampling frequency
    
    # Define segment length and number of FFT points (zero padding)
    nperseg = 2**14  # Segment length
    nfft = 2**14  # Number of FFT points, with zero padding
    E = np.zeros(3)
    colors = ['b', 'g', 'r']  # Colors for x, y, z components
    labels = ['x-component', 'y-component', 'z-component']  # Labels for x, y, z components
    for i in range(3):  # Loop over the three components
        # Calculate PSD with zero padding for each component
        f, Pxx = welch(Xem[0,:, i], Fs, nperseg=nperseg, nfft=nfft)
        # Pxx = 0.5*omegas[i]**2*M*Pxx
        # Plot PSD
        plt.loglog(f, Pxx, color=colors[i], label=labels[i])
        E[i] = 0.5*M*omegas[i]**2*np.trapz(Pxx, f)
        #plt.semilogy(f,lorentz(2*np.pi*f, *popt[i]),color=colors[i],linewidth=2)
        print(f"Total integrated energy: {E[i]} in {i} direction")
    
    fontsize = 18
    #plt.ylim((1e-24,1e-15))
    # plt.xlim((1e2,5e4))
    plt.xlabel('Frequency [Hz]',fontsize=fontsize)
    plt.ylabel(r'$S_{xx}$ [$m^2$/Hz]',fontsize=fontsize)
    plt.legend([r'$S_{xx}$','fit $S_{xx}$', r'$S_{yy}$','fit $S_{yy}$', r'$S_{zz}$','fit $S_{zz}$'], fontsize=fontsize)
    plt.title(r'Power Spectral Density of x, y, z Components at $P_{gas}$='+f'{Pgas}'+r'Pa, $T_{Gas}$ = '+f'{np.round(T_cm)}K',fontsize=fontsize)
    plt.show()
    

def wX(Z):
    w = wx*np.sqrt(1+(Z/zR)**2)
    return w

def wY(Z):
    w = wy*np.sqrt(1+(Z/zR)**2)
    return w

def E0sq(X,Y,Z):
    Ir = E0**2/(1+(Z/zR)**2)*np.exp(-2*X**2/wX(Z)**2-2*Y**2/wY(Z)**2)
    return Ir 

def FgradX(X,Y,Z):
    Fg = -alpha_p*E0sq(X,Y,Z)/(1+(Z/zR)**2)*X/wX(Z)**2
    return Fg
    
def FgradY(X,Y,Z):
    Fg = -alpha_p*E0sq(X,Y,Z)/(1+(Z/zR)**2)*Y/wY(Z)**2
    return Fg
    
def FgradZ(X,Y,Z):
    Fg = -alpha_p*E0sq(X,Y,Z)/(1+(Z/zR)**2)*(Z/(2*zR**2))*(1-2*X**2/wx**2//(1+(Z/zR)**2)-2*Y**2/wy**2//(1+(Z/zR)**2))
    return Fg

def FscattX(X,Y,Z):
    Fsc = alpha_pp*E0**2*k/2*X*Z/zR**2
    return Fsc
    
def FscattY(X,Y,Z):
    Fsc = alpha_pp*E0**2*k/2*Y*Z/zR**2
    return Fsc

def FscattZ(X,Y,Z):
    xi0 = 1-1/(k*zR)
    xix = 2/wx**2*(wx**2/(4*zR**2)-xi0)
    xiy = 2/wy**2*(wx**2/(4*zR**2)-xi0)
    xiz = -xi0/zR**2
    Fsc = alpha_pp*E0**2*k/2*(xi0+xix*X**2+xiy*Y**2+xiz*Z**2)
    return Fsc



# def I_00_integrand(theta, k, f, NA):
#     # This is a placeholder for the actual integrand of I_00.
#     # You'll need to replace it with the correct expression.
#     return np.sin(theta) * jv(0, k * f * NA * np.sin(theta))

# def I_01_integrand(theta, k, f, NA):
#     # Placeholder for I_01 integrand
#     return np.sin(theta) ** 2 * jv(1, k * f * NA * np.sin(theta))

# def I_02_integrand(theta, k, f, NA):
#     # Placeholder for I_02 integrand
#     return np.sin(theta) ** 2 * jv(2, k * f * NA * np.sin(theta))

# def calculate_integral(integrand, k, f, NA):
#     result, _ = quad(integrand, 0, np.pi/2, args=(k, f, NA))
#     return result

# def calculate_E(rho, phi, z, wavelength, f, n1, n2, E0, w0, NA):
#     theta_max = np.arcsin(NA)
    
#     #calculate first the I_nm integrals
#     f0 = w0/(f*np.sin(theta_max)) #filling factor
#     fw = np.exp(-1/f0**2*(np.sin(theta))**2/(f*np.sin(theta_max))**2)
#     k = 2 * np.pi / wavelength
#     prefactor = -1j * k * f / 2 * sqrt(n1/n2) * E0 * exp(-1j * k * f)
    
#     # Calculate integrals
#     I_00 = calculate_integral(I_00_integrand, k, f, NA)
#     I_01 = calculate_integral(I_01_integrand, k, f, NA)
#     I_02 = calculate_integral(I_02_integrand, k, f, NA)
    
#     # Calculate electric field components
#     E_rho = prefactor * (I_00 + I_02 * cos(2 * phi))
#     E_phi = prefactor * (I_02 * sin(2 * phi))
#     E_z = prefactor * (-2j * I_01 * cos(phi))
    
#     return E_rho, E_phi, E_z

# # Example parameters
# rho = 1e-6  # meters
# phi = np.pi / 4  # radians
# z = 1e-6  # meters
# wavelength = 500e-9  # 500 nm
# f = 1e-3  # focal length in meters
# n1 = 1.0  # refractive index of medium 1
# n2 = 1.5  # refractive index of medium 2
# E0 = 1.0  # electric field amplitude
# NA = 1.4  # Numerical aperture

# E_rho, E_phi, E_z = calculate_E(rho, phi, z, wavelength, f, n1, n2, E0, NA)
# print(f"E_rho: {E_rho}, E_phi: {E_phi}, E_z: {E_z}")
