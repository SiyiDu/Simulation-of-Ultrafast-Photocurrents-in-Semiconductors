"""Dataclasses used across the photocurrent simulation workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(slots=True)
class SimulationParameters:
    """Simulation grid for the time-domain solver."""

    total_time: float
    dt: float

    @property
    def steps(self) -> int:
        return int(self.total_time / self.dt)

    @property
    def time_axis(self) -> np.ndarray:
        return np.arange(self.steps) * self.dt


@dataclass(slots=True)
class TimeEvolutionResult:
    """Time-dependent observables for a single simulation run."""

    time: np.ndarray
    current: np.ndarray
    free_carriers: np.ndarray
    excitons: np.ndarray
    dt: float

    def integrated_charge(self) -> float:
        """Return the total extracted charge (integral of current over time)."""
        return float(self.current.sum() * self.dt)


@dataclass(slots=True)
class DelayScanResult:
    """Pump-probe delay scan after baseline subtraction."""

    delays: np.ndarray
    traces: Sequence[TimeEvolutionResult]
    baseline: TimeEvolutionResult
    integrated_currents: np.ndarray
    differential_charge: np.ndarray


@dataclass(slots=True)
class DelayFitResult:
    """Parameters and fitted curves for delay-dependent exponential fits."""

    tau_left: float
    tau_right: float
    popt_left: np.ndarray
    popt_right: np.ndarray
    t_left: np.ndarray
    t_right: np.ndarray
    fit_left: np.ndarray
    fit_right: np.ndarray
