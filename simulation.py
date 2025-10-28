"""Core simulation routines for ultrafast photocurrent calculations.

This module implements a one-dimensional drift-diffusion simulation with a
self-consistent electric field obtained from Poisson's equation.  It separates
electron and hole densities explicitly and provides optional diagnostics for
saving intermediate fields or enabling debug output.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.constants as const

try:  # Optional dependency used only when plotting is requested explicitly.
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - plotting is optional.
    plt = None


@dataclass
class SimulationParameters:
    """Physical and numerical parameters for the simulation."""

    length: float = 1.0e-3
    dx: float = 0.5e-5
    dt: float = 1.0e-12
    t_max: float = 1.25e-8
    temperature: float = 300.0  # Kelvin
    mobility_e: float = 300.0
    mobility_h: float = 150.0
    n0: float = 5.0e17
    p0: float = 5.0e17
    k1: float = 14.0e7
    k2: float = 5.0e-9
    k3: float = 1.0e-30
    epsilon_r: float = 12.9
    pulse_width: float = 2.0e-4
    npu0: float = 70.0e17
    npr0: float = 5.0e17
    xpu: float = 5.0e-4
    xpr: float = 5.0e-4

    @property
    def x_range(self) -> np.ndarray:
        return np.arange(0.0, self.length + self.dx, self.dx)

    @property
    def num_steps(self) -> int:
        return int(self.t_max / self.dt)


@dataclass
class DiagnosticConfig:
    """Configuration flags controlling optional diagnostic output."""

    enable_progress: bool = True
    enable_plots: bool = False
    save_fields: bool = False
    field_stride: int = 50
    field_output_dir: Path = Path("diagnostics")
    debug_interval: Optional[int] = None
    debug_enabled: bool = False

    def maybe_save_fields(
        self,
        identifier: str,
        step: int,
        phi: np.ndarray,
        electric_field: np.ndarray,
        charge_density: np.ndarray,
        n_e: np.ndarray,
        p_h: np.ndarray,
    ) -> None:
        if not self.save_fields or step % self.field_stride != 0:
            return

        self.field_output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            self.field_output_dir / f"{identifier}_step{step:06d}.npz",
            phi=phi,
            electric_field=electric_field,
            charge_density=charge_density,
            electrons=n_e,
            holes=p_h,
        )

    def maybe_debug(self, message: str, step: int) -> None:
        if self.debug_enabled and self.debug_interval and step % self.debug_interval == 0:
            print(f"[step {step}] {message}")


def gaussian(x: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-np.square((x - center) / width))


def thomas_algorithm(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Solve a tridiagonal system using the Thomas algorithm."""

    nf = len(d)  # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d))
    for it in range(1, nf):
        mc = ac[it - 1] / bc[it - 1]
        bc[it] = bc[it] - mc * cc[it - 1]
        dc[it] = dc[it] - mc * dc[it - 1]

    xc = bc
    xc[-1] = dc[-1] / bc[-1]

    for il in range(nf - 2, -1, -1):
        xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]

    return xc


def solve_poisson_dirichlet(
    charge_density: np.ndarray,
    dx: float,
    epsilon_r: float,
    phi_left: float,
    phi_right: float,
) -> np.ndarray:
    """Solve 1D Poisson equation with Dirichlet boundary conditions."""

    n = charge_density.size
    if n < 3:
        raise ValueError("Poisson solver requires at least 3 spatial points")

    eps = const.epsilon_0 * epsilon_r
    scale = -dx * dx / eps

    a = np.ones(n - 3)
    b = -2.0 * np.ones(n - 2)
    c = np.ones(n - 3)
    d = scale * charge_density[1:-1]
    d[0] -= phi_left
    d[-1] -= phi_right

    phi = np.zeros_like(charge_density)
    phi[0] = phi_left
    phi[-1] = phi_right
    phi[1:-1] = thomas_algorithm(a, b, c, d)
    return phi


def compute_electric_field(phi: np.ndarray, dx: float) -> np.ndarray:
    return -np.gradient(phi, dx, edge_order=2)


def diffusion_coefficient(mobility: float, temperature: float) -> float:
    return mobility * const.Boltzmann * temperature / const.e


def carrier_current_density(
    density: np.ndarray,
    mobility: float,
    electric_field: np.ndarray,
    diffusion_coeff: float,
    dx: float,
    sign: float,
) -> np.ndarray:
    grad_density = np.gradient(density, dx, edge_order=2)
    drift_term = mobility * density * electric_field
    diffusion_term = diffusion_coeff * grad_density
    return sign * const.e * (drift_term + diffusion_term)


def recombination_rate(
    n_e: np.ndarray,
    p_h: np.ndarray,
    params: SimulationParameters,
) -> np.ndarray:
    avg_density = 0.5 * (n_e + p_h)
    excess = avg_density - params.n0
    return (
        params.k1 * excess
        + params.k2 * avg_density * excess
        + params.k3 * np.square(avg_density) * excess
    )


def build_pulse_profiles(params: SimulationParameters) -> Tuple[np.ndarray, np.ndarray]:
    x = params.x_range
    pump_profile = params.npu0 * gaussian(x, params.xpu, params.pulse_width)
    probe_profile = params.npr0 * gaussian(x, params.xpr, params.pulse_width)
    return pump_profile, probe_profile


def evolve_system(
    voltage: float,
    pulse_schedule: Sequence[Tuple[int, np.ndarray, np.ndarray]],
    params: SimulationParameters,
    diagnostics: DiagnosticConfig,
    identifier: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = params.x_range
    n_e = np.full_like(x, params.n0, dtype=float)
    p_h = np.full_like(x, params.p0, dtype=float)

    schedule = sorted(list(pulse_schedule), key=lambda item: item[0])
    schedule_index = 0

    diffusion_e = diffusion_coefficient(params.mobility_e, params.temperature)
    diffusion_h = diffusion_coefficient(params.mobility_h, params.temperature)

    j_history = np.zeros(params.num_steps, dtype=float)

    for step in range(params.num_steps):
        while schedule_index < len(schedule) and schedule[schedule_index][0] == step:
            _, pulse_e, pulse_h = schedule[schedule_index]
            n_e += pulse_e
            p_h += pulse_h
            schedule_index += 1

        charge_density = const.e * (
            (p_h - n_e) + (params.p0 - params.n0)
        )
        phi = solve_poisson_dirichlet(
            charge_density,
            params.dx,
            params.epsilon_r,
            phi_left=0.0,
            phi_right=voltage,
        )
        electric_field = compute_electric_field(phi, params.dx)

        j_e = carrier_current_density(
            n_e,
            params.mobility_e,
            electric_field,
            diffusion_e,
            params.dx,
            sign=-1.0,
        )
        j_h = carrier_current_density(
            p_h,
            params.mobility_h,
            electric_field,
            diffusion_h,
            params.dx,
            sign=1.0,
        )

        recomb = recombination_rate(n_e, p_h, params)
        dn_dt = -np.gradient(j_e, params.dx, edge_order=2) / const.e - recomb
        dp_dt = -np.gradient(j_h, params.dx, edge_order=2) / const.e - recomb

        n_e[1:-1] += params.dt * dn_dt[1:-1]
        p_h[1:-1] += params.dt * dp_dt[1:-1]

        n_e = np.clip(n_e, 0.0, None)
        p_h = np.clip(p_h, 0.0, None)

        n_e[0] = params.n0
        n_e[-1] = params.n0
        p_h[0] = params.p0
        p_h[-1] = params.p0

        j_history[step] = j_e[0] + j_h[0]

        diagnostics.maybe_save_fields(
            identifier,
            step,
            phi,
            electric_field,
            charge_density,
            n_e,
            p_h,
        )
        diagnostics.maybe_debug(
            f"Contact current density = {j_history[step]:.3e} A/m^2",
            step,
        )

    return j_history, n_e, p_h


def integrate_current(current_trace: np.ndarray, dt: float) -> float:
    total_charge_density = np.trapz(current_trace, dx=dt)
    average_current = total_charge_density / (current_trace.size * dt)
    return average_current * 1e9  # Convert to nA assuming unit area.


def build_pulse_schedule(
    td: float,
    params: SimulationParameters,
    pump_profile: np.ndarray,
    probe_profile: np.ndarray,
) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    delay_steps = int(round(abs(td) / params.dt))

    pump = (pump_profile.copy(), pump_profile.copy())
    probe = (probe_profile.copy(), probe_profile.copy())

    if td >= 0:
        schedule = [(0, pump[0], pump[1])]
        if delay_steps > 0:
            schedule.append((delay_steps, probe[0], probe[1]))
        else:
            schedule[0] = (
                0,
                pump[0] + probe[0],
                pump[1] + probe[1],
            )
    else:
        schedule = [(0, probe[0], probe[1])]
        if delay_steps > 0:
            schedule.append((delay_steps, pump[0], pump[1]))
        else:
            schedule[0] = (
                0,
                probe[0] + pump[0],
                probe[1] + pump[1],
            )

    return schedule


def run_simulation(
    voltages: Iterable[float],
    td_values: Sequence[float],
    params: Optional[SimulationParameters] = None,
    diagnostics: Optional[DiagnosticConfig] = None,
) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
    params = params or SimulationParameters()
    diagnostics = diagnostics or DiagnosticConfig()

    pump_profile, probe_profile = build_pulse_profiles(params)
    td_array = np.asarray(td_values, dtype=float)

    results: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}

    for voltage in voltages:
        base_schedule = [(0, pump_profile.copy(), pump_profile.copy())]
        base_trace, _, _ = evolve_system(
            voltage,
            base_schedule,
            params,
            diagnostics,
            identifier=f"V{voltage:.2f}_baseline",
        )
        q_baseline = integrate_current(base_trace, params.dt)

        q_values = np.zeros_like(td_array)

        td_iterable: Iterable[float]
        if diagnostics.enable_progress:
            from tqdm import tqdm

            td_iterable = tqdm(td_array, desc=f"V={voltage} V", colour="green")
        else:
            td_iterable = td_array

        for index, td in enumerate(td_iterable):
            schedule = build_pulse_schedule(td, params, pump_profile, probe_profile)
            trace, _, _ = evolve_system(
                voltage,
                schedule,
                params,
                diagnostics,
                identifier=f"V{voltage:.2f}_td{td:+.2e}",
            )
            q_values[index] = integrate_current(trace, params.dt) - q_baseline

        results[voltage] = (td_array.copy(), q_values)

    return results


def plot_results(results: Dict[float, Tuple[np.ndarray, np.ndarray]]) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting diagnostics")

    for voltage, (tds, currents) in results.items():
        plt.plot(tds * 1e9, currents, marker="o", label=f"V = {voltage} V")

    plt.xlabel("td (ns)")
    plt.ylabel("Average current (nA)")
    plt.title("Photocurrent vs. pump-probe delay")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    default_params = SimulationParameters()
    td_values = np.arange(-0.5e-9, 0.51e-9, 0.05e-9)
    voltage_range = range(4, 8, 1)

    diagnostics = DiagnosticConfig(enable_progress=True, enable_plots=True)

    simulation_results = run_simulation(voltage_range, td_values, default_params, diagnostics)

    if diagnostics.enable_plots:
        plot_results(simulation_results)
