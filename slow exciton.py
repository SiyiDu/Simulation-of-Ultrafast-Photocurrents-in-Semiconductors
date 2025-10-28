# -*- coding: utf-8 -*-
"""Pump-probe photocurrent simulation with structured outputs."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from photocurrent import (
    ExperimentalConditions,
    SimulationParameters,
    fit_delay_scan,
    plot_delay_scan,
    simulate_delay_scan,
)


conditions = ExperimentalConditions()
params = SimulationParameters(total_time=5e-10, dt=5e-13)
delays = np.arange(-1e-10, 1e-10, 5e-13)

n_photon_pr = conditions.photon_induced_density(conditions.Ppr_avg)
n_photon_pu = conditions.photon_induced_density(conditions.Ppu_avg)
print(f"n induced per pump: {n_photon_pu/1e17:.2f} E17 cm^-3")
print(f"n induced per probe: {n_photon_pr/1e17:.2f} E17 cm^-3")

pump_density = float(n_photon_pu)
probe_density = float(n_photon_pr)

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_title(f'Photocurrent vs Probe Delay Time (T={conditions.temperature}K)')
ax.set_xlabel('Delay Time (s)')
ax.set_ylabel('Differential Charge (C)')
colors = plt.cm.viridis(np.linspace(0, 1, len(conditions.Vbias)))

tau_left_values = np.zeros(len(conditions.Vbias))
tau_right_values = np.zeros(len(conditions.Vbias))

with tqdm(total=len(conditions.Vbias), desc="Simulating", unit="bias") as pbar:
    for idx, Vb in enumerate(conditions.Vbias):
        scan_result = simulate_delay_scan(
            conditions,
            params,
            Vb,
            delays,
            pump_density=pump_density,
            probe_density=probe_density,
        )

        plot_delay_scan(scan_result, ax=ax, label=f'Vb={Vb}V', color=colors[idx])

        fit_result = fit_delay_scan(scan_result)
        tau_left_values[idx] = fit_result.tau_left
        tau_right_values[idx] = fit_result.tau_right

        ax.plot(fit_result.t_left, fit_result.fit_left, '--', color=colors[idx], alpha=0.7)
        ax.plot(fit_result.t_right, fit_result.fit_right, '--', color=colors[idx], alpha=0.7)

        pbar.update(1)

ax.legend()
plt.show()

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.set_title(f'Relaxation Times vs Applied Bias (T={conditions.temperature}K)')
ax2.set_xlabel('Applied Bias (V)')
ax2.set_ylabel('Relaxation Time (s)')
ax2.plot(conditions.Vbias, tau_left_values, 'o-', label='τ_left (pump before probe)')
ax2.plot(conditions.Vbias, tau_right_values, 's-', label='τ_right (probe before pump)')
ax2.set_yscale('log')
ax2.legend()
plt.show()
