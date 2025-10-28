import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.optimize import curve_fit
from tqdm import tqdm

from shared_solver import (
    BoundaryCondition,
    apply_boundary_conditions,
    compute_current_density,
    compute_legacy_charge,
    integrate_current_trace,
)

kB = const.Boltzmann
q = const.e
h = const.h
c = const.c
PulsePower1st = 1

n0 = 5e17
k1 = 14e7
k2 = 1e-9
k3 = 1e-30

L = 1e-3
a = 0  # D modifier: not doing a lot at the moment
mb = 300

npu0 = 70e17
npr0 = 5e17
T = 10

t0 = 3e-11
t_max = 1.25e-8
dt = 1e-12
dx = 0.5e-5
xpu = 5e-4
xpr = 5e-4
w = 2e-4
JP = 1

voltage_range = range(4, 8, 1)

x_range = np.arange(0, L + dx, dx)
dnx = np.zeros(len(x_range))
n_initial = np.zeros(len(x_range)) + n0
Nt = int(t_max / dt)

# Lists to store Q_trapz values and td values
td_values = np.arange(-0.5e-9, 0.51e-9, 0.05e-9)


def D(n):
    return mb * kB * T / q * (n0 ** a) / np.power(n, a)


def diffusion(n):
    return D(n) * (np.roll(n, -1) - np.roll(n, 1)) / (2 * dx)


def drift(n, E):
    return n * mb * E


def gaussian(x, x_0, w=1e-4):
    return np.exp(-1 * np.square((x - x_0) / w))


def expo(x, a, b, t):
    return a + b * np.exp(x / t)


# Loop over voltage values and td values to calculate Q_avg - Q_avg0
for V in voltage_range:
    E = V / L
    Q_values = []
    n = n_initial.copy()
    n1 = npu0 * gaussian(x_range, xpu)
    n += n1

    boundary_condition = BoundaryCondition(
        kind="dirichlet",
        value=(n0, n0),
        reason=(
            "Contacts are assumed to pin the carrier density to the thermal "
            "equilibrium value, enforcing a Dirichlet boundary condition."
        ),
    )

    baseline_bc_log = {}
    J_values = []
    legacy_J_values = []
    for t in range(Nt):
        nprime = (np.roll(n, -1) - np.roll(n, 1)) / (2 * dx)
        dif = (np.roll(diffusion(n), -1) - np.roll(diffusion(n), 1)) / (2 * dx)
        driE = (np.roll(drift(n, E), -1) - np.roll(drift(n, E), 0)) / dx
        rec = k1 * (n - n0) + k2 * n * (n - n0) + k3 * n * n * (n - n0)

        dnx = (dif + driE - rec) * dt
        n += dnx

        baseline_bc_log = apply_boundary_conditions(
            n, boundary_condition, dx, record=baseline_bc_log
        )
        J_total_0 = compute_current_density(n, E, mb, D(n), dx)

        J_values.append(J_total_0)
        legacy_J_values.append(dnx[0])

    Q_avg0 = integrate_current_trace(J_values, dt)
    legacy_Q_avg0 = compute_legacy_charge(legacy_J_values, t_max)
    print(
        f"Applied {baseline_bc_log['type']} boundary condition because {baseline_bc_log['reason']}"
    )
    print(
        f"Baseline charge: legacy={legacy_Q_avg0:.3e} C, drift-diffusion={Q_avg0:.3e} C"
    )
    # Loop over different td values
    loop = 0
    td_left = []
    Q_left = []
    for td in tqdm(td_values, desc=f"Processing td values for V={V}", colour="green"):
        iteration = 1
        n = n_initial.copy()
        n1 = npu0 * gaussian(x_range, xpu)
        n2 = npr0 * gaussian(x_range, xpr)
        if td > 0:
            n1st = n1
            n2nd = n2
        else:
            n1st = n2
            n2nd = n1
        n += n1st

        J_values = []
        legacy_J_values = []
        td_bc_log = {}

        pulse_width = 1e-10  # Example pulse width
        pulse_steps = int(pulse_width / dt)

        for t in range(Nt):
            if t == int(abs(td) / dt):
                for i in range(pulse_steps):
                    n += n2nd / pulse_steps

            nprime = (np.roll(n, -1) - np.roll(n, 1)) / (2 * dx)
            dif = (np.roll(diffusion(n), -1) - np.roll(diffusion(n), 1)) / (2 * dx)
            driE = (np.roll(drift(n, E), -1) - np.roll(drift(n, E), 0)) / dx
            rec = k1 * (n - n0) + k2 * n * (n - n0) + k3 * n * n * (n - n0)

            dnx = (dif + driE - rec) * dt
            n += dnx

            td_bc_log = apply_boundary_conditions(n, boundary_condition, dx, record=td_bc_log)
            J_total_0 = compute_current_density(n, E, mb, D(n), dx)

            J_values.append(J_total_0)
            legacy_J_values.append(dnx[0])

            iteration += 1

        Q_total = integrate_current_trace(J_values, dt)
        legacy_Q_total = compute_legacy_charge(legacy_J_values, t_max)
        delta_Q = (Q_total - Q_avg0) * 1e9
        Q_values.append(delta_Q)
        print(
            "# of charges collected (legacy vs new): ",
            f"{legacy_Q_total:.3e} C -> {Q_total:.3e} C",
        )
        if td == td_values[0]:
            print(
                f"First delay boundary condition: {td_bc_log['type']} (reason: {td_bc_log['reason']})"
            )

        if loop == 10:
            Q_left = Q_values.copy()
            td_left = td_values[:11]
        loop += 1

    plt.plot(td_values, Q_values, marker='o', label=f'V={V}V')
    plt.show()

# Finalize plot
plt.xlabel('td (ns)')
plt.ylabel('ΔQ (nC)')
plt.title(f'ΔQ vs td for different V values(k1={k1}, k2={k2}, n0={n0:.1e}, mb={mb})')
plt.legend()
plt.grid(True)
plt.show()

