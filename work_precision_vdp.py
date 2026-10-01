#!/usr/bin/env python3
"""Work-precision study for adaptive BDF/BDFL methods on stiff van der Pol.

Place this script in the same directory as ``robertson_test.py``.  It imports
only the variable-step coefficient functions from that file and supplies its
own adaptive driver.

Main features
-------------
* corrected van der Pol Jacobian;
* cubic-Hermite extrapolation error estimator for the BDF/BDFL steps;
* no problem-dependent hard minimum step size (only a floating-point guard);
* rejection on nonlinear-solver failure;
* TR-BDF2 step-doubling for startup/restart history;
* restart when a rejected LMM step would require an excessively small step
  ratio relative to the accepted multistep history;
* RHS and Jacobian evaluation counts;
* time-integrated L2 errors against a tight Radau reference.

For a second-order LMM, the cubic-Hermite predictor is fourth-order accurate
when adjacent step ratios remain bounded.  Hence the difference between the
implicit LMM value and the predictor is asymptotically the O(h^3) local error
of the LMM itself.  We therefore use

    E_n = || y_{n+1}^{LMM} - z_{n+1}^{Hermite} ||

with the elementary controller

    h_new = safety * h * (tol/E_n)^(1/3).

Examples
--------
Single diagnostic run:
    python work_precision_vdp.py single --method BDFL5_2 --tol 1e-5 \
        --t-final 2000 --diagnostics bdfl5_diag.csv

Full work-precision sweep:
    python work_precision_vdp.py sweep --t-final 2000 \
        --tols 1e-3 3e-4 1e-4 3e-5 1e-5 3e-6 1e-6 \
        --output-dir wp_2000_hermite

The primary plotted error is

    ( integral_0^T |y_1(t)-y_{1,ref}(t)|^2 dt )^(1/2),

approximated by the trapezoidal rule on the accepted adaptive grid.  The CSV
also contains the corresponding full-state L2 error, RMS error in y1, and
maximum nodal error in y1.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from scipy.integrate import solve_ivp

try:
    import robertson_test as rt
except ImportError as exc:
    raise SystemExit(
        "Could not import robertson_test.py. Put this script in the repository "
        "directory containing robertson_test.py, or add that directory to PYTHONPATH."
    ) from exc


Array = np.ndarray
CoeffFun = Callable[[Sequence[float], int], Array]

MU = 500.0
Y0 = np.array([2.0, 0.0], dtype=float)


def vdp_rhs(t: float, u: Array, mu: float = MU) -> Array:
    y1, y2 = u
    return np.array([y2, mu * (1.0 - y1**2) * y2 - y1], dtype=float)


def vdp_jac(t: float, u: Array, mu: float = MU) -> Array:
    y1, y2 = u
    return np.array(
        [[0.0, 1.0], [-2.0 * mu * y1 * y2 - 1.0, mu * (1.0 - y1**2)]],
        dtype=float,
    )


METHODS: Dict[str, Tuple[CoeffFun, CoeffFun]] = {
    "BDF2": (rt.alpha_bdf2, rt.beta_bdf2),
    "BDFL3_2": (rt.alpha_bdfl3_poly, rt.beta_bdfl3_poly),
    "BDFL4_2": (rt.alpha_bdfl4, rt.beta_bdfl4),
    "BDFL5_2": (rt.alpha_bdfl5, rt.beta_bdfl5),
}


@dataclass
class EvalCounter:
    rhs: int = 0
    jac: int = 0


@dataclass
class RunStats:
    method: str
    tol: float
    success: bool
    message: str
    accepted_steps: int
    rejected_steps: int
    attempted_steps: int
    rhs_evals: int
    jac_evals: int
    min_h: float
    max_h: float
    restart_steps: int
    l2_error_y1: float = math.nan
    l2_error_state: float = math.nan
    rms_error_y1: float = math.nan
    max_error_y1: float = math.nan
    max_error_state: float = math.nan
    wall_seconds: float = math.nan


@dataclass
class DiagnosticRow:
    attempt: int
    accepted_index: int
    t: float
    t_trial: float
    h: float
    omega: float
    accepted: int
    reason: str
    E: float
    nonlinear_residual: float
    nonlinear_nfev: int
    proposed_h: float


class CountedProblem:
    def __init__(self, counter: EvalCounter):
        self.counter = counter

    def f(self, t: float, u: Array) -> Array:
        self.counter.rhs += 1
        return vdp_rhs(t, u)

    def jac(self, t: float, u: Array) -> Array:
        self.counter.jac += 1
        return vdp_jac(t, u)


def method_steps(alpha: CoeffFun) -> int:
    """Return k for a k-step coefficient function."""
    dummy_h = [1.0] * 8
    return len(alpha(dummy_h, -1)) - 1


def _fsolve(
    residual: Callable[[Array], Array],
    jacobian: Callable[[Array], Array],
    guess: Array,
    xtol: float = 1e-11,
) -> Tuple[Array, bool, float, int]:
    sol, info, ier, _ = opt.fsolve(
        residual,
        guess,
        fprime=jacobian,
        full_output=True,
        xtol=xtol,
    )
    resnorm = float(np.linalg.norm(info["fvec"]))
    ok = bool(ier == 1 or resnorm <= 1e-10)
    return np.asarray(sol, dtype=float), ok, resnorm, int(info.get("nfev", 0))


def tr_bdf2_step(
    t_n: float,
    y_n: Array,
    h: float,
    problem: CountedProblem,
) -> Tuple[Array, bool, float, int]:
    """One TR-BDF2 startup step with gamma=1/2."""
    t_half = t_n + 0.5 * h
    t_np1 = t_n + h
    eye = np.eye(len(y_n))

    f_n = problem.f(t_n, y_n)

    def r1(u_star: Array) -> Array:
        return u_star - y_n - 0.25 * h * (f_n + problem.f(t_half, u_star))

    def j1(u_star: Array) -> Array:
        return eye - 0.25 * h * problem.jac(t_half, u_star)

    u_star, ok1, res1, nfev1 = _fsolve(r1, j1, y_n)
    if not ok1:
        return u_star, False, res1, nfev1

    def r2(u_np1: Array) -> Array:
        return u_np1 - (4.0 * u_star - y_n + h * problem.f(t_np1, u_np1)) / 3.0

    def j2(u_np1: Array) -> Array:
        return eye - (h / 3.0) * problem.jac(t_np1, u_np1)

    u_np1, ok2, res2, nfev2 = _fsolve(r2, j2, u_star)
    return u_np1, ok2, max(res1, res2), nfev1 + nfev2


def tr_bdf2_embedded(
    t_n: float,
    y_n: Array,
    h: float,
    problem: CountedProblem,
) -> Tuple[Array, float, bool, float, int]:
    """TR-BDF2 step-doubling estimate used for startup/restart.

    The accepted value is the result of two half steps.  For a second-order
    method, ||y_fine-y_coarse||/3 estimates the local error of the fine value.
    """
    y_coarse, okc, resc, nfc = tr_bdf2_step(t_n, y_n, h, problem)
    if not okc:
        return y_coarse, math.inf, False, resc, nfc

    y_half, ok1, res1, nf1 = tr_bdf2_step(t_n, y_n, 0.5 * h, problem)
    if not ok1:
        return y_half, math.inf, False, max(resc, res1), nfc + nf1

    y_fine, ok2, res2, nf2 = tr_bdf2_step(t_n + 0.5 * h, y_half, 0.5 * h, problem)
    if not ok2:
        return y_fine, math.inf, False, max(resc, res1, res2), nfc + nf1 + nf2

    E = float(np.linalg.norm(y_fine - y_coarse) / 3.0)
    return y_fine, E, True, max(resc, res1, res2), nfc + nf1 + nf2


def cubic_hermite_predictor(
    y_nm1: Array,
    f_nm1: Array,
    y_n: Array,
    f_n: Array,
    h_prev: float,
    h_new: float,
) -> Array:
    """Extrapolate from [t_{n-1},t_n] to t_{n+1} with cubic Hermite data.

    The cubic satisfies p(t_{n-1})=y_{n-1}, p'(t_{n-1})=f_{n-1},
    p(t_n)=y_n, p'(t_n)=f_n.  With u=(t-t_{n-1})/h_prev,
    t_{n+1} corresponds to u=1+h_new/h_prev.
    """
    if h_prev <= 0.0 or h_new <= 0.0:
        raise ValueError("step sizes must be positive")

    u = 1.0 + h_new / h_prev
    h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
    h10 = u**3 - 2.0 * u**2 + u
    h01 = -2.0 * u**3 + 3.0 * u**2
    h11 = u**3 - u**2

    return (
        h00 * y_nm1
        + h10 * h_prev * f_nm1
        + h01 * y_n
        + h11 * h_prev * f_n
    )


def controller_factor(
    E: float,
    tol: float,
    safety: float,
    omega_min: float,
    omega_max: float,
) -> Tuple[float, bool]:
    """Return multiplicative step factor and acceptance decision."""
    if E == 0.0:
        return omega_max, True

    fac = safety * (tol / E) ** (1.0 / 3.0)
    fac = min(omega_max, max(omega_min, fac))
    return fac, bool(E <= tol)


def solve_adaptive(
    method: str,
    tol: float,
    t_final: float = 300.0,
    h0: float = 1e-3,
    safety: float = 0.9,
    omega_min: float = 0.1,
    omega_max: float = 1.5,
    max_attempts: int = 5_000_000,
    diagnostics_path: Optional[Path] = None,
) -> Tuple[Array, Array, RunStats, List[DiagnosticRow]]:
    """Solve van der Pol with one adaptive BDF/BDFL method.

    TR-BDF2 with step doubling supplies startup history.  The LMM local-error
    estimate is the difference between its implicit solution and a cubic
    Hermite extrapolation from the two most recent accepted solution values and
    derivatives.

    If a rejected LMM step would require h_new/h_prev < omega_min, the recent
    multistep history is rebuilt with TR-BDF2 rather than repeatedly retrying
    the LMM with an increasingly extreme step ratio.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {list(METHODS)}")
    if tol <= 0 or h0 <= 0 or t_final <= 0:
        raise ValueError("tol, h0 and t_final must be positive")
    if not (0.0 < omega_min < 1.0 <= omega_max):
        raise ValueError("require 0 < omega_min < 1 <= omega_max")

    alpha, beta = METHODS[method]
    s = method_steps(alpha)

    counter = EvalCounter()
    problem = CountedProblem(counter)
    diagnostics: List[DiagnosticRow] = []

    t_vals: List[float] = [0.0]
    y_vals: List[Array] = [Y0.copy()]
    accepted_h: List[float] = []
    f_vals: List[Array] = [problem.f(0.0, Y0)]

    rejected = 0
    attempts = 0
    min_h = math.inf
    max_h = 0.0
    restart_steps = 0
    start_clock = time.perf_counter()

    # A k-step method needs k history values, hence k-1 recent history steps.
    # We take at least two startup steps so the Hermite predictor has a clean
    # recent interval even for BDF2.
    bootstrap_target = max(s - 1, 2)
    bootstrap_remaining = bootstrap_target
    initial_bootstrap = True
    h_trial = float(h0)

    def failure_stats(message: str) -> Tuple[Array, Array, RunStats, List[DiagnosticRow]]:
        stats = RunStats(
            method=method,
            tol=tol,
            success=False,
            message=message,
            accepted_steps=len(accepted_h),
            rejected_steps=rejected,
            attempted_steps=attempts,
            rhs_evals=counter.rhs,
            jac_evals=counter.jac,
            min_h=min_h,
            max_h=max_h,
            restart_steps=restart_steps,
            wall_seconds=time.perf_counter() - start_clock,
        )
        if diagnostics_path is not None:
            write_diagnostics(diagnostics_path, diagnostics)
        return np.asarray(t_vals), np.asarray(y_vals), stats, diagnostics

    while t_vals[-1] < t_final:
        attempts += 1
        if attempts > max_attempts:
            return failure_stats("maximum number of attempted steps exceeded")

        remaining = t_final - t_vals[-1]
        h_use = min(h_trial, remaining)
        floor = 100.0 * np.finfo(float).eps * max(1.0, abs(t_vals[-1]))
        if h_use <= floor:
            return failure_stats(
                f"floating-point step-size underflow at t={t_vals[-1]:.16e}"
            )

        # --------------------------------------------------------------
        # Startup/restart mode.
        # --------------------------------------------------------------
        if bootstrap_remaining > 0:
            y_new, E_boot, ok, residual, nfev = tr_bdf2_embedded(
                t_vals[-1], y_vals[-1], h_use, problem
            )
            omega = h_use / accepted_h[-1] if accepted_h else math.nan

            if not ok:
                rejected += 1
                proposed_h = 0.5 * h_use
                diagnostics.append(DiagnosticRow(
                    attempts, len(accepted_h), t_vals[-1], t_vals[-1] + h_use,
                    h_use, omega, 0,
                    "startup nonlinear failure" if initial_bootstrap else "restart nonlinear failure",
                    E_boot, residual, nfev, proposed_h,
                ))
                h_trial = proposed_h
                continue

            fac, accept = controller_factor(
                E_boot, tol, safety, omega_min, omega_max
            )
            proposed_h = fac * h_use

            if not accept:
                rejected += 1
                diagnostics.append(DiagnosticRow(
                    attempts, len(accepted_h), t_vals[-1], t_vals[-1] + h_use,
                    h_use, omega, 0,
                    "startup error rejection" if initial_bootstrap else "restart error rejection",
                    E_boot, residual, nfev, proposed_h,
                ))
                h_trial = proposed_h
                continue

            t_new = t_vals[-1] + h_use
            t_vals.append(t_new)
            y_vals.append(y_new)
            accepted_h.append(h_use)
            f_vals.append(problem.f(t_new, y_new))
            min_h = min(min_h, h_use)
            max_h = max(max_h, h_use)
            if not initial_bootstrap:
                restart_steps += 1

            diagnostics.append(DiagnosticRow(
                attempts, len(accepted_h) - 1, t_vals[-2], t_new,
                h_use, omega, 1,
                "startup" if initial_bootstrap else "restart",
                E_boot, residual, nfev, proposed_h,
            ))
            bootstrap_remaining -= 1
            h_trial = proposed_h
            if bootstrap_remaining == 0:
                initial_bootstrap = False
            continue

        # --------------------------------------------------------------
        # Main variable-step LMM attempt.
        # --------------------------------------------------------------
        h_prev = accepted_h[-1]

        # Except for the final clipped step, enforce the ratio restriction on
        # the LMM itself.  If the proposed step is too small relative to the
        # accepted history, rebuild the recent history instead.
        if remaining > h_trial and h_use < omega_min * h_prev:
            bootstrap_remaining = bootstrap_target
            initial_bootstrap = False
            h_trial = h_use
            continue

        h_seq = accepted_h + [h_use]
        alpha_n = np.asarray(alpha(h_seq, -1), dtype=float)
        beta_n = np.asarray(beta(h_seq, -1), dtype=float)
        if len(alpha_n) != s + 1 or len(beta_n) != s + 1:
            raise RuntimeError(f"coefficient function for {method} returned wrong length")

        t_new = t_vals[-1] + h_use
        eye = np.eye(2)

        # Hermite extrapolation is both the nonlinear initial guess and the
        # independent high-order comparison value for local error estimation.
        z_np1 = cubic_hermite_predictor(
            y_vals[-2], f_vals[-2], y_vals[-1], f_vals[-1],
            h_prev, h_use,
        )

        # Cache all fixed history terms outside the nonlinear residual.
        hist = np.zeros(2, dtype=float)
        for j in range(s):
            hist += alpha_n[j] * y_vals[-s + j]
            hist -= h_use * beta_n[j] * f_vals[-s + j]

        def residual(u_new: Array) -> Array:
            return hist + alpha_n[s] * u_new - h_use * beta_n[s] * problem.f(t_new, u_new)

        def jacobian(u_new: Array) -> Array:
            return alpha_n[s] * eye - h_use * beta_n[s] * problem.jac(t_new, u_new)

        result, ok, nonlinear_residual, nonlinear_nfev = _fsolve(
            residual, jacobian, z_np1
        )
        omega = h_use / h_prev

        if not ok:
            rejected += 1
            candidate = 0.5 * h_use
            diagnostics.append(DiagnosticRow(
                attempts, len(accepted_h), t_vals[-1], t_new, h_use, omega,
                0, "nonlinear failure", math.nan,
                nonlinear_residual, nonlinear_nfev, candidate,
            ))
            if candidate < omega_min * h_prev:
                bootstrap_remaining = bootstrap_target
                initial_bootstrap = False
            h_trial = candidate
            continue

        raw_diff = float(np.linalg.norm(result - z_np1))
        roundoff = np.finfo(float).eps * max(
            1.0, float(np.linalg.norm(result)), float(np.linalg.norm(z_np1))
        )
        E = 0.0 if raw_diff <= roundoff else raw_diff

        fac, accept = controller_factor(E, tol, safety, omega_min, omega_max)
        candidate = fac * h_use

        if not accept:
            rejected += 1
            diagnostics.append(DiagnosticRow(
                attempts, len(accepted_h), t_vals[-1], t_new, h_use, omega,
                0, "error rejection", E,
                nonlinear_residual, nonlinear_nfev, candidate,
            ))
            if candidate < omega_min * h_prev:
                bootstrap_remaining = bootstrap_target
                initial_bootstrap = False
            h_trial = candidate
            continue

        # Accept LMM step.
        t_vals.append(t_new)
        y_vals.append(result)
        accepted_h.append(h_use)
        f_vals.append(problem.f(t_new, result))
        min_h = min(min_h, h_use)
        max_h = max(max_h, h_use)
        diagnostics.append(DiagnosticRow(
            attempts, len(accepted_h) - 1, t_vals[-2], t_new, h_use, omega,
            1, "accepted", E,
            nonlinear_residual, nonlinear_nfev, candidate,
        ))
        h_trial = candidate

    elapsed = time.perf_counter() - start_clock
    stats = RunStats(
        method=method,
        tol=tol,
        success=True,
        message="ok",
        accepted_steps=len(accepted_h),
        rejected_steps=rejected,
        attempted_steps=attempts,
        rhs_evals=counter.rhs,
        jac_evals=counter.jac,
        min_h=min_h,
        max_h=max_h,
        restart_steps=restart_steps,
        wall_seconds=elapsed,
    )

    if diagnostics_path is not None:
        write_diagnostics(diagnostics_path, diagnostics)

    return np.asarray(t_vals), np.asarray(y_vals), stats, diagnostics


def make_reference(t_final: float):
    ref = solve_ivp(
        vdp_rhs,
        (0.0, t_final),
        Y0,
        method="Radau",
        jac=vdp_jac,
        rtol=1e-12,
        atol=1e-14,
        dense_output=True,
    )
    if not ref.success:
        raise RuntimeError(f"Radau reference solve failed: {ref.message}")
    return ref


def _trapz(values: Array, t: Array) -> float:
    """Compatibility wrapper for NumPy versions before/after np.trapezoid."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, t))
    return float(np.trapz(values, t))


def add_errors(stats: RunStats, t: Array, y: Array, ref) -> None:
    """Add nodal max errors and time-integrated L2 errors to ``stats``."""
    y_ref = ref.sol(t).T
    err = y - y_ref

    e1_sq = err[:, 0] ** 2
    estate_sq = np.sum(err**2, axis=1)

    int_e1_sq = max(0.0, _trapz(e1_sq, t))
    int_estate_sq = max(0.0, _trapz(estate_sq, t))

    stats.l2_error_y1 = math.sqrt(int_e1_sq)
    stats.l2_error_state = math.sqrt(int_estate_sq)
    T = float(t[-1] - t[0])
    stats.rms_error_y1 = stats.l2_error_y1 / math.sqrt(T) if T > 0 else math.nan
    stats.max_error_y1 = float(np.max(np.abs(err[:, 0])))
    stats.max_error_state = float(np.max(np.linalg.norm(err, axis=1)))


def write_diagnostics(path: Path, rows: Sequence[DiagnosticRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    default_fields = [
        "attempt", "accepted_index", "t", "t_trial", "h", "omega", "accepted",
        "reason", "E", "nonlinear_residual", "nonlinear_nfev", "proposed_h",
    ]
    fields = list(asdict(rows[0]).keys()) if rows else default_fields
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_stats_csv(path: Path, rows: Sequence[RunStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(rows[0]).keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def plot_work_precision(rows: Sequence[RunStats], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    good = [r for r in rows if r.success and np.isfinite(r.l2_error_y1)]
    if not good:
        return

    for xfield, xlabel, fname in [
        ("rhs_evals", "RHS evaluations", "work_precision_rhs_l2.pdf"),
        ("attempted_steps", "Attempted time steps", "work_precision_attempts_l2.pdf"),
    ]:
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        for method in METHODS:
            mr = sorted(
                (r for r in good if r.method == method),
                key=lambda r: getattr(r, xfield),
            )
            if not mr:
                continue
            x = [getattr(r, xfield) for r in mr]
            y = [r.l2_error_y1 for r in mr]
            ax.loglog(x, y, marker="o", label=method.replace("_2", r"$_2$"))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\left(\int_0^T |y_1-y_{1,\mathrm{ref}}|^2\,dt\right)^{1/2}$")
        ax.grid(True, which="both")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / fname, dpi=200)
        plt.close(fig)


def print_stats(stats: RunStats) -> None:
    print(
        f"{stats.method:8s} tol={stats.tol:.1e} success={stats.success} "
        f"accepted={stats.accepted_steps} rejected={stats.rejected_steps} "
        f"attempts={stats.attempted_steps} rhs={stats.rhs_evals} jac={stats.jac_evals} "
        f"min_h={stats.min_h:.3e} restarts={stats.restart_steps} "
        f"L2_y1={stats.l2_error_y1:.3e} RMS_y1={stats.rms_error_y1:.3e} "
        f"max_y1={stats.max_error_y1:.3e} time={stats.wall_seconds:.2f}s"
    )
    if not stats.success:
        print("  failure:", stats.message)


def run_single(args: argparse.Namespace) -> None:
    diag = Path(args.diagnostics) if args.diagnostics else None
    t, y, stats, rows = solve_adaptive(
        args.method,
        args.tol,
        t_final=args.t_final,
        h0=args.h0,
        safety=args.safety,
        omega_min=args.omega_min,
        omega_max=args.omega_max,
        max_attempts=args.max_attempts,
        diagnostics_path=diag,
    )
    if stats.success:
        ref = make_reference(args.t_final)
        add_errors(stats, t, y, ref)
    print_stats(stats)

    if args.diagnostics and rows:
        print(f"diagnostics written to {args.diagnostics}")
        reasons: Dict[str, int] = {}
        for r in rows:
            if not r.accepted:
                reasons[r.reason] = reasons.get(r.reason, 0) + 1
        if reasons:
            print("rejection reasons:")
            for key, val in sorted(reasons.items()):
                print(f"  {key}: {val}")


def run_sweep(args: argparse.Namespace) -> None:
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    ref = make_reference(args.t_final)
    rows: List[RunStats] = []

    for method in args.methods:
        for tol in args.tols:
            print(f"Running {method}, tol={tol:.3e} ...", flush=True)
            diag_path = None
            if args.save_diagnostics:
                diag_path = outdir / f"diag_{method}_tol_{tol:.0e}.csv"
            t, y, stats, _ = solve_adaptive(
                method,
                tol,
                t_final=args.t_final,
                h0=args.h0,
                safety=args.safety,
                omega_min=args.omega_min,
                omega_max=args.omega_max,
                max_attempts=args.max_attempts,
                diagnostics_path=diag_path,
            )
            if stats.success:
                add_errors(stats, t, y, ref)
            rows.append(stats)
            print_stats(stats)
            # Persist after every run so a long sweep is recoverable.
            write_stats_csv(outdir / "work_precision.csv", rows)

    plot_work_precision(rows, outdir)
    print(f"\nWrote {outdir / 'work_precision.csv'}")
    print(f"Wrote {outdir / 'work_precision_rhs_l2.pdf'}")
    print(f"Wrote {outdir / 'work_precision_attempts_l2.pdf'}")


def add_common_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--t-final", type=float, default=300.0)
    p.add_argument("--h0", type=float, default=1e-3)
    p.add_argument("--safety", type=float, default=0.9)
    p.add_argument("--omega-min", type=float, default=0.1)
    p.add_argument("--omega-max", type=float, default=1.5)
    p.add_argument("--max-attempts", type=int, default=5_000_000)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("single", help="run one method/tolerance with optional diagnostics")
    p1.add_argument("--method", choices=list(METHODS), default="BDFL5_2")
    p1.add_argument("--tol", type=float, default=1e-5)
    p1.add_argument("--diagnostics", type=str, default="vdp_diagnostics.csv")
    add_common_options(p1)
    p1.set_defaults(func=run_single)

    p2 = sub.add_parser("sweep", help="run the work-precision sweep")
    p2.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    p2.add_argument(
        "--tols", nargs="+", type=float,
        default=[1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6],
    )
    p2.add_argument("--output-dir", type=str, default="work_precision_vdp")
    p2.add_argument("--save-diagnostics", action="store_true")
    add_common_options(p2)
    p2.set_defaults(func=run_sweep)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
