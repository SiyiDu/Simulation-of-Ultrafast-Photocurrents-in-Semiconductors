"""Experimental conditions and physical helper functions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

# Physical constants
k_B = 1.380649e-23  # Boltzmann constant in J/K
q = 1.602e-19  # elementary charge in C
h = 6.626e-34  # Planck constant in J·s

_PHOTON_CONVERSION = 62918306.066


def _ensure_array(value: Iterable[float] | float) -> np.ndarray:
    """Return *value* as a NumPy array."""
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value, dtype=float)


@dataclass
class ExperimentalConditions:
    """Container for experimental parameters with derived quantities."""

    channel_length: float = 10e-4  # 10 µm in cm
    r_beam: float = 1e-4  # beam radius in cm (stdev of gaussian)
    skindepth: float = 20e-7  # skin depth in cm
    wavelength: float = 730  # in nm
    Ppr_avg: float = 1e-6  # probe power in W
    Ppu_avg: float = 10e-6  # pump power in W
    pulse_width: float = 100e-15  # pulse width in s (100 fs)
    Vbias: np.ndarray = field(default_factory=lambda: np.array([-6, -3, -1.5, -0.7, -0.2]))
    temperature: float = 30  # in Kelvin
    m_e: float = 0.12 * 9.11e-31  # electron effective mass in kg
    m_h: float = 0.15 * 9.11e-31  # hole effective mass in kg
    mu_e0: float = 50  # free electron mobility at 300K in cm^2/(V·s)
    mu_h0: float = 30  # free hole mobility at 300K in cm^2/(V·s)
    mu_ex: float = 0.1  # exciton mobility in cm^2/(V·s)
    n0: float = 5e17  # dark carrier concentration in cm^-3
    E_b: float = 15e-3 * q  # exciton binding energy in J (15 meV)
    dissociation_time: float = 1e-12  # 1 ps dissociation time

    def __post_init__(self) -> None:
        self._update_derived()

    def _update_derived(self) -> None:
        self.a_beam = np.pi * self.r_beam * self.r_beam
        self.Vbias = _ensure_array(self.Vbias)
        self.E_field = self.Vbias / self.channel_length

    def update(self, **kwargs: float) -> None:
        """Update attributes and refresh derived quantities."""
        for key, value in kwargs.items():
            setattr(self, key, value)
        self._update_derived()

    def photon_induced_density(self, power: float) -> int:
        """Return the carrier density induced by a pulse with the given power."""
        return int(power * self.wavelength * _PHOTON_CONVERSION / self.a_beam / self.skindepth)


# Recombination parameters
k1 = 7.4e9  # 1/s
k2 = 3e-10  # cm^3/s
k3 = 1.3e-30  # cm^6/s


def temperature_dependent_mobility(T: float, mu_0: float) -> float:
    """Calculate temperature-dependent mobility (T^(-3/2) for phonon scattering)."""
    return mu_0 * (300 / T) ** 1.5


def saha_equation(conditions: ExperimentalConditions, T: float, n_total: float) -> tuple[float, float]:
    """Calculate free carrier and exciton fractions using the Saha equation."""
    m_red = (conditions.m_e * conditions.m_h) / (conditions.m_e + conditions.m_h)
    thermal_energy = k_B * T
    prefactor = (2 * np.pi * m_red * thermal_energy / h**2) ** (3 / 2)
    K = prefactor * np.exp(-conditions.E_b / thermal_energy)
    a = 1.0
    b = K
    c = -K * n_total
    discriminant = b**2 - 4 * a * c
    n_f = (-b + np.sqrt(discriminant)) / (2 * a)
    n_ex = n_total - n_f
    return n_f, n_ex


def drift_velocity(E: float, mu: float) -> float:
    """Calculate drift velocity for free carriers."""
    return mu * E


def diffusion_coefficient(T: float, mu: float) -> float:
    """Calculate diffusion coefficient using the Einstein relation."""
    return mu * (k_B * T) / q


def escape(
    n_f: float,
    Vb: float,
    T: float,
    mu: float,
    conditions: ExperimentalConditions,
) -> float:
    """Calculate the escape rate using a drift-diffusion framework."""
    if np.size(conditions.Vbias) > 1:
        matches = np.where(np.isclose(conditions.Vbias, Vb))[0]
        if matches.size:
            E = conditions.E_field[matches[0]]
        else:
            E = Vb / conditions.channel_length
    else:
        E = Vb / conditions.channel_length
    mu_eff = temperature_dependent_mobility(T, mu)
    v_drift = drift_velocity(E, mu_eff)
    D = diffusion_coefficient(T, mu_eff)
    if v_drift != 0:
        tau_drift = conditions.r_beam / abs(v_drift)
    else:
        tau_drift = 1e10
    tau_diff = (conditions.r_beam**2) / (4 * D)
    if tau_drift > 0 and tau_diff > 0:
        tau_eff = (tau_drift * tau_diff) / (tau_drift + tau_diff)
    else:
        tau_eff = max(tau_drift, tau_diff)
    return n_f / tau_eff


def recomb(n: float, conditions: ExperimentalConditions) -> float:
    """Recombination rate."""
    return k1 * (n - conditions.n0) + k2 * (n**2 - conditions.n0**2) + k3 * (n - conditions.n0) ** 3


def photon_induced_density(power: float, conditions: ExperimentalConditions) -> int:
    """Convenience wrapper to compute photon-induced carrier density."""
    return conditions.photon_induced_density(power)
