# -*- coding: utf-8 -*-
"""
Created on Wed Mar 3  2025

@author: dongyu, songz( ͡⚆ ͜ʖ ͡⚆)╭∩╮, AI
PVSK project, Yu Lab, UC Davis

MAPbI3 Pump-probe photocurrent simulation

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm  # For progress bar

# Experimental conditions grouped for easy modification
class ExperimentalConditions:
    channel_length = 10e-4  # 10 um in cm
    r_beam = 1e-4  # beam radius in cm (stdev of gaussian)
    a_beam = np.pi * r_beam * r_beam  # beam spot size in cm^2
    skindepth = 20e-7  # skin depth in cm
    wavelength = 730  # in nm
    Ppr_avg = 1e-6  # probe power in W
    Ppu_avg = 10e-6  # pump power in W
    pulse_width = 100e-15  # pulse width in s (100 fs)
    Vbias = np.array([-6, -3, -1.5, -0.7, -0.2 ])  # applied bias in V
    E_field = Vbias / channel_length  # electric field in V/cm
    temperature = 30  # in Kelvin
    m_e = 0.12 * 9.11e-31  # electron effective mass in kg
    m_h = 0.15 * 9.11e-31  # hole effective mass in kg
    mu_e0 = 50  # free electron mobility at 300K in cm^2/(V·s)
    mu_h0 = 30   # free hole mobility at 300K in cm^2/(V·s)
    mu_ex = 0.1  # exciton mobility in cm^2/(V·s)
    n0 = 5e17  # dark carrier concentration in cm^-3
    E_b = 15e-3 * 1.602e-19  # exciton binding energy in J (15 meV)
    dissociation_time = 1e-12  # 1 ps dissociation time

# Physical constants
k_B = 1.380649e-23  # Boltzmann constant in J/K
q = 1.602e-19  # elementary charge in C
h = 6.626e-34  # Planck constant in J·s

# Calculate photon-induced carrier density
conditions = ExperimentalConditions()
n_photon_pr = int(conditions.Ppr_avg * conditions.wavelength * 62918306.066 / conditions.a_beam / conditions.skindepth)
n_photon_pu = int(conditions.Ppu_avg * conditions.wavelength * 62918306.066 / conditions.a_beam / conditions.skindepth)
print(f"n induced per pump: {n_photon_pu/1e17:.2f} E17 cm^-3")
print(f"n induced per probe: {n_photon_pr/1e17:.2f} E17 cm^-3")

# Recombination parameters
k1 = 7.4e9  # 1/s
k2 = 3e-10  # cm^3/s
k3 = 1.3e-30  # cm^6/s

def temperature_dependent_mobility(T, mu_0):
    """Calculate temperature-dependent mobility (T^(-3/2) for phonon scattering)."""
    return mu_0 * (300 / T)**1.5

def saha_equation(T, n_total):
    """Calculate free carrier and exciton fractions using Saha equation."""
    m_red = (conditions.m_e * conditions.m_h) / (conditions.m_e + conditions.m_h)
    thermal_energy = k_B * T
    prefactor = (2 * np.pi * m_red * thermal_energy / h**2)**(3/2)
    K = prefactor * np.exp(-conditions.E_b / thermal_energy)
    a = 1
    b = K
    c = -K * n_total
    n_f = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
    n_ex = n_total - n_f
    return n_f, n_ex

def drift_velocity(E, mu):
    """Calculate drift velocity for free carriers."""
    return mu * E

def diffusion_coefficient(T, mu):
    """Calculate diffusion coefficient using Einstein relation."""
    return mu * (k_B * T) / q

def escape(n_f, Vb, T, mu):
    """Calculate escape rate using drift-diffusion framework."""
    idx = np.where(conditions.Vbias == Vb)[0][0]
    E = conditions.E_field[idx]
    mu_eff = temperature_dependent_mobility(T, mu)
    v_drift = drift_velocity(E, mu_eff)
    D = diffusion_coefficient(T, mu_eff)
    if v_drift != 0:
        tau_drift = conditions.r_beam / np.abs(v_drift)
    else:
        tau_drift = 1e10
    tau_diff = (conditions.r_beam**2) / (4 * D)
    if tau_drift > 0 and tau_diff > 0:
        tau_eff = (tau_drift * tau_diff) / (tau_drift + tau_diff)
    else:
        tau_eff = max(tau_drift, tau_diff)
    return (n_f) / tau_eff

def recomb(n):
    """Recombination rate."""
    return k1 * (n - conditions.n0) + k2 * (n**2 - conditions.n0**2) + k3 * (n - conditions.n0)**3

def single_exponential(t, A, tau, C):
    """Single exponential function for fitting: I(t) = -A * exp(-|t|/tau) + C."""
    return A * np.exp(-np.abs(t) / tau) + C

def fit_both_sides(t, I):
    """Fit both sides of zero delay to single exponentials."""
    # Split data into left (t < 0) and right (t > 0)
    zero_idx = np.argmin(np.abs(t))  # Closest to zero delay
    t_left = t[:zero_idx + 1][::-1]  # Reverse to make time positive for fitting
    I_left = I[:zero_idx + 1][::-1]
    t_right = t[zero_idx:]
    I_right = I[zero_idx:]
    
    # Initial guesses
    A_guess = max(I) - min(I)
    tau_guess = 5e-11
    C_guess = min(I)
    
    # Fit left side (pump before probe)
    try:
        t_left_rel = np.abs(t_left - t_left[-1])  # Relative to zero delay
        popt_left, _ = curve_fit(single_exponential, t_left_rel, I_left,
                               p0=[A_guess, tau_guess, C_guess],
                               bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]))
        tau_left = popt_left[1]
    except RuntimeError:
        tau_left = np.nan
        popt_left = [0, 0, 0]
    
    # Fit right side (probe before pump)
    try:
        t_right_rel = t_right - t_right[0]  # Relative to zero delay
        popt_right, _ = curve_fit(single_exponential, t_right_rel, I_right,
                                p0=[A_guess, tau_guess, C_guess],
                                bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]))
        tau_right = popt_right[1]
    except RuntimeError:
        tau_right = np.nan
        popt_right = [0, 0, 0]
    
    return tau_left, tau_right, popt_left, popt_right

# Simulation parameters
tM = 5e-10
#tM = 125e-10  # total simulation time in s
dt = 5e-13  # time step in s
N = int(tM/dt)  # total steps
td = np.arange(-1e-10, 1e-10, 5e-13)  # delay times in s

# Storage for relaxation times
tau_left_values = np.zeros(len(conditions.Vbias))
tau_right_values = np.zeros(len(conditions.Vbias))

# Plotting setup
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_title(f'Photocurrent vs Probe Delay Time (T={conditions.temperature}K)')
ax.set_xlabel('Delay Time (s)')
ax.set_ylabel('Current (A)')
colors = plt.cm.viridis(np.linspace(0, 1, len(conditions.Vbias)))

# Single temperature
T = conditions.temperature

# Progress bar for bias loop
with tqdm(total=len(conditions.Vbias), desc="Simulating", unit="bias") as pbar:
    for Vb_idx, Vb in enumerate(conditions.Vbias):
        I_total = []
        
        # Pump only
        n_total = n_photon_pu + conditions.n0
        n_f, n_ex = saha_equation(T, n_total)
        I_pu = 0
        for i in range(N):
            I_pu += q * conditions.a_beam * escape(n_f, Vb, T, conditions.mu_e0) * dt
            n_f -= (escape(n_f, Vb, T, conditions.mu_e0) + recomb(n_f)) * dt
            n_ex -= (recomb(n_ex) + n_ex / conditions.dissociation_time) * dt
            n_f += (n_ex / conditions.dissociation_time) * dt
            n_f = max(n_f, conditions.n0)
        
        # Pump + Probe
        for delay in td:
            I_pupr = 0
            if delay < 0:
                n1, n2 = n_photon_pu, n_photon_pr
                t1 = -delay
            else:
                n1, n2 = n_photon_pr, n_photon_pu
                t1 = delay
            
            N1 = int(t1/dt)
            n_total = n1 + conditions.n0
            n_f, n_ex = saha_equation(T, n_total)
            
            for j in range(N1):
                I_pupr += q * conditions.a_beam * escape(n_f, Vb, T, conditions.mu_e0) * dt
                n_f -= (escape(n_f, Vb, T, conditions.mu_e0) + recomb(n_f)) * dt
                n_ex -= (recomb(n_ex) + n_ex / conditions.dissociation_time) * dt
                n_f += (n_ex / conditions.dissociation_time) * dt
                n_f = max(n_f, conditions.n0)
            
            n_total += n2
            n_f, n_ex = saha_equation(T, n_total)
            for j in range(N1, N):
                I_pupr += q * conditions.a_beam * escape(n_f, Vb, T, conditions.mu_e0) * dt
                n_f -= (escape(n_f, Vb, T, conditions.mu_e0) + recomb(n_f)) * dt
                n_ex -= (recomb(n_ex) + n_ex / conditions.dissociation_time) * dt
                n_f += (n_ex / conditions.dissociation_time) * dt
                n_f = max(n_f, conditions.n0)
            
            I_total.append(I_pupr - I_pu)
        
        # Plot photocurrent data
        label = f'Vb={Vb}V'
        ax.plot(td, I_total, label=label, color=colors[Vb_idx])
        
        # Fit both sides and overlay fits
        tau_left, tau_right, popt_left, popt_right = fit_both_sides(td, I_total)
        tau_left_values[Vb_idx] = tau_left
        tau_right_values[Vb_idx] = tau_right
        
        # Overlay left fit
        zero_idx = np.argmin(np.abs(td))
        t_left = td[:zero_idx + 1][::-1]
        t_left_rel = np.abs(t_left - t_left[-1])
        I_fit_left = single_exponential(t_left_rel, *popt_left)
        ax.plot(t_left, I_fit_left, '--', color=colors[Vb_idx], alpha=0.7)
        
        # Overlay right fit
        t_right = td[zero_idx:]
        t_right_rel = t_right - t_right[0]
        I_fit_right = single_exponential(t_right_rel, *popt_right)
        ax.plot(t_right, I_fit_right, '--', color=colors[Vb_idx], alpha=0.7)
        
        # Update progress bar
        pbar.update(1)

ax.legend()
plt.show()

# Plot relaxation times vs bias
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.set_title(f'Relaxation Times vs Applied Bias (T={conditions.temperature}K)')
ax2.set_xlabel('Applied Bias (V)')
ax2.set_ylabel('Relaxation Time (s)')
ax2.plot(conditions.Vbias, tau_left_values, 'o-', label='τ_left (pump before probe)')
ax2.plot(conditions.Vbias, tau_right_values, 's-', label='τ_right (probe before pump)')
ax2.set_yscale('log')
ax2.legend()
plt.show()