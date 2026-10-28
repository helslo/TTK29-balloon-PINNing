"""Forward Balloon-Windkessel model for task-fMRI hemodynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


InputFunction = Callable[[float], float]


# Defaults are based on Friston et al. (2000) and SPM's DCM hemodynamic
# defaults. Double-check every exact value against the paper/SPM documentation
# before treating these values as ground truth for this project.
@dataclass
class BalloonParams:
    """Parameters for the neurovascular and Balloon-Windkessel model."""

    kappa: float = 0.65
    gamma: float = 0.41
    tau: float = 0.98
    alpha: float = 0.32
    E0: float = 0.34
    V0: float = 0.02
    eps: float = 0.5
    nu0: float = 40.3
    r0: float = 25.0
    epsilon_r: float = 0.47
    TE: float = 0.04


@dataclass
class BalloonFit:
    """Result of fitting Balloon parameters to one BOLD time series."""

    params: BalloonParams
    fitted_bold: np.ndarray
    optimizer_cost: float
    nfev: int


def build_input_function(
    onsets: Sequence[float],
    duration: float,
    dt: float,
    pulse_width: float = 1.0,
) -> InputFunction:
    """Build a sampled boxcar neuronal drive from event onset times.

    The short boxcar-pulse input is a simple task-design approximation. It is
    used as the neuronal drive in the neurovascular coupling equations from
    Friston et al. (2000), rather than being an equation from the original
    Balloon-Windkessel model in Buxton et al. (1998).

    Args:
        onsets: Event onset times in seconds.
        duration: Duration of the returned input grid in seconds.
        dt: Grid spacing in seconds.
        pulse_width: Width of each boxcar pulse in seconds.

    Returns:
        A callable that evaluates the input at arbitrary scalar times by
        linear interpolation on the sampled input grid.
    """
    if duration < 0:
        raise ValueError("duration must be non-negative")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if pulse_width <= 0:
        raise ValueError("pulse_width must be positive")

    onset_array = np.asarray(onsets, dtype=float)
    if onset_array.ndim != 1:
        raise ValueError("onsets must be a one-dimensional sequence")
    if not np.all(np.isfinite(onset_array)):
        raise ValueError("onsets must contain only finite values")

    time_grid = np.arange(0.0, duration + dt / 2.0, dt)
    input_grid = np.zeros_like(time_grid)
    for onset in onset_array:
        input_grid[(time_grid >= onset) & (time_grid < onset + pulse_width)] += 1.0

    def input_function(t: float) -> float:
        """Evaluate the interpolated neuronal drive at time ``t``."""
        return float(np.interp(t, time_grid, input_grid, left=0.0, right=0.0))

    return input_function


def balloon_ode(
    t: float,
    y: Sequence[float],
    u_func: InputFunction,
    params: BalloonParams,
) -> Tuple[float, float, float, float]:
    """Evaluate the four-state hemodynamic ODE.

    The neurovascular coupling equations for ``s`` and ``f`` follow Friston
    et al. (2000). The venous volume and deoxyhemoglobin equations follow the
    Balloon-Windkessel model of Buxton, Wong & Frank (1998), with the
    flow-dependent extraction fraction defined in Friston et al. (2000).

    Args:
        t: Time in seconds.
        y: State ``[s, f, v, q]``.
        u_func: Callable returning neuronal drive at time ``t``.
        params: Model parameters.

    Returns:
        Derivatives ``[ds/dt, df/dt, dv/dt, dq/dt]``.
    """
    s, f, v, q = y
    if f <= 0 or v <= 0:
        raise ValueError("f and v must remain positive")

    extraction = 1.0 - (1.0 - params.E0) ** (1.0 / f)
    flow_term = v ** (1.0 / params.alpha)

    ds = params.eps * u_func(t) - params.kappa * s - params.gamma * (f - 1.0)
    df = s
    dv = (f - flow_term) / params.tau
    dq = (
        f * extraction / params.E0 - (flow_term / v) * q
    ) / params.tau
    return ds, df, dv, dq


def bold_signal(v: np.ndarray, q: np.ndarray, params: BalloonParams) -> np.ndarray:
    """Compute BOLD signal from volume and deoxyhemoglobin states.

    This static observation equation is the Balloon-Windkessel BOLD equation
    from Buxton, Wong & Frank (1998), with the coefficients and extraction
    convention used by Friston et al. (2000).
    """
    k1 = 4.3 * params.nu0 * params.E0 * params.TE
    k2 = params.epsilon_r * params.r0 * params.E0 * params.TE
    k3 = 1.0 - params.epsilon_r
    return params.V0 * (
        k1 * (1.0 - q) + k2 * (1.0 - q / v) + k3 * (1.0 - v)
    )


def simulate_balloon(
    u_func: InputFunction,
    params: BalloonParams,
    t_span: Tuple[float, float],
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the model and return time, states, and BOLD signal.

    The initial state is ``[s, f, v, q] = [0, 1, 1, 1]``. Integration uses
    SciPy's RK45 solver. The ODE is the Friston et al. (2000) neurovascular
    front-end plus the Buxton, Wong & Frank (1998) Balloon-Windkessel model;
    the returned BOLD signal uses their static observation equation.

    Args:
        u_func: Callable returning neuronal drive at time ``t``.
        params: Model parameters.
        t_span: ``(start_time, end_time)`` in seconds.
        dt: Sampling interval for returned values in seconds.

    Returns:
        ``(times, states, bold)`` where states has shape ``(4, len(times))``.
    """
    start, end = t_span
    if end <= start:
        raise ValueError("t_span must have end greater than start")
    if dt <= 0:
        raise ValueError("dt must be positive")

    times = np.arange(start, end + dt / 2.0, dt)
    times = times[times <= end]
    if times[-1] < end:
        times = np.append(times, end)

    solution = solve_ivp(
        fun=lambda t, y: balloon_ode(t, y, u_func, params),
        t_span=(start, end),
        y0=[0.0, 1.0, 1.0, 1.0],
        t_eval=times,
        method="RK45",
        max_step=dt,
    )
    if not solution.success:
        raise RuntimeError(f"Balloon model integration failed: {solution.message}")

    states = solution.y
    bold = bold_signal(states[2], states[3], params)
    return solution.t, states, bold


def fit_balloon(
    times: np.ndarray,
    observed_bold: np.ndarray,
    u_func: InputFunction,
    initial_params: BalloonParams | None = None,
    max_nfev: int = 100,
) -> BalloonFit:
    """Fit selected Balloon parameters to an observed BOLD trace.

    The fitted parameters are ``kappa``, ``gamma``, ``tau``, ``alpha``,
    ``eps``, and ``V0``. Scanner-specific observation parameters remain fixed
    because they are determined by the acquisition protocol.
    """
    times = np.asarray(times, dtype=float)
    observed_bold = np.asarray(observed_bold, dtype=float)
    if times.ndim != 1 or observed_bold.ndim != 1 or len(times) != len(observed_bold):
        raise ValueError("times and observed_bold must be one-dimensional arrays of equal length")
    if len(times) < 3 or np.any(np.diff(times) <= 0):
        raise ValueError("times must contain at least three strictly increasing values")
    if not np.all(np.isfinite(observed_bold)):
        raise ValueError("observed_bold must contain only finite values")
    if max_nfev <= 0:
        raise ValueError("max_nfev must be positive")

    base = initial_params or BalloonParams()
    names = ("kappa", "gamma", "tau", "alpha", "eps", "V0")
    initial = np.array([getattr(base, name) for name in names], dtype=float)
    lower = np.array([0.01, 0.01, 0.05, 0.1, 0.01, 0.001])
    upper = np.array([5.0, 5.0, 5.0, 1.0, 5.0, 0.2])

    def predict(values: np.ndarray) -> np.ndarray:
        fitted_params = BalloonParams(
            kappa=values[0], gamma=values[1], tau=values[2], alpha=values[3],
            eps=values[4], V0=values[5], E0=base.E0, nu0=base.nu0,
            r0=base.r0, epsilon_r=base.epsilon_r, TE=base.TE,
        )
        try:
            simulated_times, _, predicted = simulate_balloon(
                u_func, fitted_params, (float(times[0]), float(times[-1])),
                float(np.median(np.diff(times))) / 2.0,
            )
        except (RuntimeError, ValueError, FloatingPointError):
            return np.full_like(observed_bold, 1e3)
        return np.interp(times, simulated_times, predicted)

    result = least_squares(
        lambda values: predict(values) - observed_bold,
        initial,
        bounds=(lower, upper),
        max_nfev=max_nfev,
        x_scale="jac",
    )
    fitted_params = BalloonParams(
        kappa=result.x[0], gamma=result.x[1], tau=result.x[2], alpha=result.x[3],
        eps=result.x[4], V0=result.x[5], E0=base.E0, nu0=base.nu0,
        r0=base.r0, epsilon_r=base.epsilon_r, TE=base.TE,
    )
    fitted_bold = predict(result.x)
    return BalloonFit(fitted_params, fitted_bold, float(result.cost), result.nfev)


if __name__ == "__main__":
    example_duration = 30.0
    example_dt = 0.05
    example_input = build_input_function(
        onsets=[2.0, 10.0, 18.0],
        duration=example_duration,
        dt=example_dt,
    )
    time, states, bold = simulate_balloon(
        example_input,
        BalloonParams(),
        t_span=(0.0, example_duration),
        dt=example_dt,
    )

    input_trace = np.array([example_input(t) for t in time])

    labels = ["u(t) (input)", "s (signal)", "f (CBF)", "v (volume)", "q (deoxyhemoglobin)", "BOLD"]
    values = [input_trace, states[0], states[1], states[2], states[3], bold]
    figure, axes = plt.subplots(6, 1, figsize=(9, 12), sharex=True)
    for axis, label, value in zip(axes, labels, values):
        axis.plot(time, value)
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    figure.tight_layout()
    plt.savefig("results/balloon_model_simulation.png")
    plt.show()