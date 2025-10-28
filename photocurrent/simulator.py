"""Numerical solvers for the photocurrent model."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .conditions import ExperimentalConditions, escape, q, recomb, saha_equation
from .data import DelayScanResult, SimulationParameters, TimeEvolutionResult


def _propagate(
    conditions: ExperimentalConditions,
    params: SimulationParameters,
    Vb: float,
    n_f: float,
    n_ex: float,
) -> TimeEvolutionResult:
    """Run the time evolution for the provided initial populations."""

    steps = params.steps
    dt = params.dt
    time = params.time_axis
    current = np.zeros(steps)
    free_carriers = np.zeros(steps)
    excitons = np.zeros(steps)

    for idx in range(steps):
        free_carriers[idx] = n_f
        excitons[idx] = n_ex

        escape_rate = escape(n_f, Vb, conditions.temperature, conditions.mu_e0, conditions)
        current[idx] = q * conditions.a_beam * escape_rate

        n_f -= (escape_rate + recomb(n_f, conditions)) * dt
        n_ex -= (recomb(n_ex, conditions) + n_ex / conditions.dissociation_time) * dt
        n_f += (n_ex / conditions.dissociation_time) * dt
        n_f = max(n_f, conditions.n0)

    return TimeEvolutionResult(time=time, current=current, free_carriers=free_carriers, excitons=excitons, dt=dt)


def simulate_baseline_response(
    conditions: ExperimentalConditions,
    params: SimulationParameters,
    Vb: float,
    pump_density: float,
) -> TimeEvolutionResult:
    """Return the pump-only baseline response."""

    n_total = pump_density + conditions.n0
    n_f, n_ex = saha_equation(conditions, conditions.temperature, n_total)
    return _propagate(conditions, params, Vb, n_f, n_ex)


def _simulate_with_delay(
    conditions: ExperimentalConditions,
    params: SimulationParameters,
    Vb: float,
    delay: float,
    pump_density: float,
    probe_density: float,
) -> TimeEvolutionResult:
    """Simulate the pump-probe response for a single delay value."""

    dt = params.dt
    steps = params.steps
    time = params.time_axis
    current = np.zeros(steps)
    free_carriers = np.zeros(steps)
    excitons = np.zeros(steps)

    if delay < 0:
        first_density, second_density = pump_density, probe_density
        separation = -delay
    else:
        first_density, second_density = probe_density, pump_density
        separation = delay

    switch_step = int(round(separation / dt))

    n_total = first_density + conditions.n0
    n_f, n_ex = saha_equation(conditions, conditions.temperature, n_total)

    for idx in range(steps):
        if idx == switch_step and switch_step < steps:
            n_total += second_density
            n_f, n_ex = saha_equation(conditions, conditions.temperature, n_total)

        free_carriers[idx] = n_f
        excitons[idx] = n_ex

        escape_rate = escape(n_f, Vb, conditions.temperature, conditions.mu_e0, conditions)
        current[idx] = q * conditions.a_beam * escape_rate

        n_f -= (escape_rate + recomb(n_f, conditions)) * dt
        n_ex -= (recomb(n_ex, conditions) + n_ex / conditions.dissociation_time) * dt
        n_f += (n_ex / conditions.dissociation_time) * dt
        n_f = max(n_f, conditions.n0)

    return TimeEvolutionResult(time=time, current=current, free_carriers=free_carriers, excitons=excitons, dt=dt)


def simulate_delay_scan(
    conditions: ExperimentalConditions,
    params: SimulationParameters,
    Vb: float,
    delays: Sequence[float],
    pump_density: float,
    probe_density: float,
) -> DelayScanResult:
    """Run a full pump-probe delay scan and return structured results."""

    delays = np.asarray(delays, dtype=float)
    baseline = simulate_baseline_response(conditions, params, Vb, pump_density)
    traces: list[TimeEvolutionResult] = []
    integrated_currents = np.zeros_like(delays)
    differential_charge = np.zeros_like(delays)

    baseline_charge = baseline.integrated_charge()

    for idx, delay in enumerate(delays):
        trace = _simulate_with_delay(conditions, params, Vb, delay, pump_density, probe_density)
        traces.append(trace)
        integrated = trace.integrated_charge()
        integrated_currents[idx] = integrated
        differential_charge[idx] = integrated - baseline_charge

    return DelayScanResult(
        delays=delays,
        traces=traces,
        baseline=baseline,
        integrated_currents=integrated_currents,
        differential_charge=differential_charge,
    )
