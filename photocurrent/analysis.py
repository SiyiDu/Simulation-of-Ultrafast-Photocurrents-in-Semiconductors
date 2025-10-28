"""Analysis utilities for photocurrent simulations."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from .data import DelayFitResult, DelayScanResult


def single_exponential(t: np.ndarray, A: float, tau: float, C: float) -> np.ndarray:
    """Single exponential function for fitting."""
    return A * np.exp(-np.abs(t) / tau) + C


def fit_delay_scan(
    scan: DelayScanResult,
    tau_guess: float = 5e-11,
) -> DelayFitResult:
    """Fit the left/right sides of the delay scan using single exponentials."""

    delays = scan.delays
    response = scan.differential_charge
    zero_idx = int(np.argmin(np.abs(delays)))

    t_left = delays[: zero_idx + 1][::-1]
    I_left = response[: zero_idx + 1][::-1]
    t_right = delays[zero_idx:]
    I_right = response[zero_idx:]

    A_guess = float(np.max(response) - np.min(response))
    C_guess = float(np.min(response))

    try:
        t_left_rel = np.abs(t_left - t_left[-1])
        popt_left, _ = curve_fit(
            single_exponential,
            t_left_rel,
            I_left,
            p0=[A_guess, tau_guess, C_guess],
            bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]),
        )
        tau_left = float(popt_left[1])
        fit_left = single_exponential(t_left_rel, *popt_left)
    except RuntimeError:
        tau_left = float("nan")
        popt_left = np.array([0.0, 0.0, 0.0])
        fit_left = np.zeros_like(t_left)

    try:
        t_right_rel = t_right - t_right[0]
        popt_right, _ = curve_fit(
            single_exponential,
            t_right_rel,
            I_right,
            p0=[A_guess, tau_guess, C_guess],
            bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]),
        )
        tau_right = float(popt_right[1])
        fit_right = single_exponential(t_right_rel, *popt_right)
    except RuntimeError:
        tau_right = float("nan")
        popt_right = np.array([0.0, 0.0, 0.0])
        fit_right = np.zeros_like(t_right)

    return DelayFitResult(
        tau_left=tau_left,
        tau_right=tau_right,
        popt_left=popt_left,
        popt_right=popt_right,
        t_left=t_left,
        t_right=t_right,
        fit_left=fit_left,
        fit_right=fit_right,
    )


def plot_delay_scan(
    scan: DelayScanResult,
    ax: Optional[plt.Axes] = None,
    *,
    label: Optional[str] = None,
    color: Optional[str] = None,
) -> plt.Axes:
    """Plot baseline-subtracted differential charge vs delay."""

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(scan.delays, scan.differential_charge, marker="o", label=label, color=color)
    ax.set_xlabel("Delay Time (s)")
    ax.set_ylabel("Differential Charge (C)")
    return ax
