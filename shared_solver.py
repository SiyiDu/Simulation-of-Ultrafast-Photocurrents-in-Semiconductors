"""Shared drift-diffusion solver utilities.

This module centralises helper routines that are reused by the
experiments to evolve the carrier density, enforce boundary conditions
and evaluate the device current in a physically meaningful way.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import MutableMapping, Sequence

import numpy as np
import scipy.constants as const


@dataclass(frozen=True)
class BoundaryCondition:
    """Container describing how the domain boundaries are handled.

    Attributes
    ----------
    kind:
        Either ``"dirichlet"`` or ``"neumann"``.
    value:
        Tuple describing the boundary values.  For Dirichlet conditions
        the entries correspond to the carrier density imposed at the left
        and right contacts.  For Neumann conditions they represent the
        density gradients (\partial n / \partial x) at the contacts.
    reason:
        Human readable explanation that is emitted once per simulation to
        document why the given boundary treatment was chosen.
    """

    kind: str
    value: Sequence[float] | float
    reason: str

    def as_pair(self) -> tuple[float, float]:
        """Return the boundary values as a length-2 tuple."""
        if np.isscalar(self.value):  # type: ignore[arg-type]
            scalar = float(self.value)  # type: ignore[assignment]
            return scalar, scalar
        values = tuple(float(v) for v in self.value)
        if len(values) != 2:
            raise ValueError(
                "Boundary condition 'value' must contain two entries (left, right)."
            )
        return values  # type: ignore[return-value]


def apply_boundary_conditions(
    n: np.ndarray,
    bc: BoundaryCondition,
    dx: float,
    record: MutableMapping[str, str] | None = None,
) -> MutableMapping[str, str]:
    """Enforce the requested boundary condition on ``n``.

    The boundary metadata is returned (and optionally written into
    ``record``) so that callers can persist the justification message for
    reporting purposes.
    """

    left_value, right_value = bc.as_pair()
    kind = bc.kind.lower()

    if kind == "dirichlet":
        n[0] = left_value
        n[-1] = right_value
    elif kind == "neumann":
        n[0] = n[1] - left_value * dx
        n[-1] = n[-2] + right_value * dx
    else:
        raise ValueError(f"Unsupported boundary condition kind: {bc.kind}")

    info: MutableMapping[str, str]
    if record is None:
        info = {"type": bc.kind, "reason": bc.reason}
    else:
        record["type"] = bc.kind
        record["reason"] = bc.reason
        info = record
    return info


def compute_current_density(
    n: np.ndarray,
    electric_field: float,
    mobility: float,
    diffusion_coefficient: np.ndarray | float,
    dx: float,
    contact_index: int = 0,
) -> float:
    """Return the drift-diffusion current density at the selected contact.

    Parameters
    ----------
    n:
        Carrier density profile.
    electric_field:
        Applied macroscopic electric field (V/m).
    mobility:
        Carrier mobility used for the drift term (m^2/V/s).
    diffusion_coefficient:
        Value(s) of the diffusion coefficient corresponding to ``n``.
    dx:
        Spatial grid spacing (m).
    contact_index:
        Index of the contact at which the current is evaluated.
    """

    n = np.asarray(n)
    if n.ndim != 1:
        raise ValueError("Carrier density array must be one-dimensional.")
    if not (0 <= contact_index < n.size):
        raise IndexError("contact_index is out of bounds for the density array.")

    if np.isscalar(diffusion_coefficient):  # type: ignore[arg-type]
        D_contact = float(diffusion_coefficient)  # type: ignore[assignment]
    else:
        diff_array = np.asarray(diffusion_coefficient)
        if diff_array.shape != n.shape:
            raise ValueError(
                "Diffusion coefficient array must have the same shape as the density array."
            )
        D_contact = float(diff_array[contact_index])

    idx = contact_index
    if idx == 0:
        gradient = (n[1] - n[0]) / dx
    elif idx == n.size - 1:
        gradient = (n[-1] - n[-2]) / dx
    else:
        gradient = (n[idx + 1] - n[idx - 1]) / (2 * dx)

    drift_term = mobility * n[idx] * electric_field
    diffusion_term = D_contact * gradient
    return const.e * (drift_term - diffusion_term)


def integrate_current_trace(current_trace: Sequence[float], dt: float) -> float:
    """Integrate a current trace over time to obtain the transported charge."""

    current_array = np.asarray(current_trace, dtype=float)
    if current_array.size == 0:
        return 0.0
    time_axis = np.arange(current_array.size) * dt
    return float(np.trapz(current_array, time_axis))


def compute_legacy_charge(density_increment_trace: Sequence[float], total_time: float) -> float:
    """Reproduce the legacy charge estimate used by older scripts."""

    if not density_increment_trace:
        return 0.0
    density_array = np.asarray(density_increment_trace, dtype=float)
    return const.e * density_array.mean() * total_time
