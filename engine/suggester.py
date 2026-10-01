"""Fast, offline hyperparameter scouting for SINDy.

The scout is intentionally a recommender, not an optimizer. It evaluates a
small set of real SINDy fits on blocked validation data and returns three
useful starting points: simplest, balanced, and best derivative fit.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pysindy as ps
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score


DEFAULT_THRESHOLDS = (0.005, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50)


def _library(kind, degree):
    if kind == "Polynomial":
        return ps.PolynomialLibrary(degree=int(degree))
    if kind == "Fourier":
        return ps.FourierLibrary(n_frequencies=int(degree))
    if kind == "Combined":
        return (ps.PolynomialLibrary(degree=int(degree))
                + ps.FourierLibrary(n_frequencies=int(degree)))
    raise ValueError(f"Unknown library: {kind}")


def _smooth_derivative_and_noise(df):
    """Return X, dX/dt, and a dimensionless robust noise estimate."""
    t = np.asarray(df.iloc[:, 0], dtype=float)
    X = np.asarray(df.iloc[:, 1:], dtype=float)
    if len(t) < 9:
        raise ValueError("Data Scout needs at least 9 samples per trajectory.")
    if np.any(np.diff(t) <= 0):
        raise ValueError("Time must be strictly increasing in every trajectory.")

    target = min(11, max(5, (len(t) // 10) * 2 + 1))
    window = min(target, len(t) if len(t) % 2 else len(t) - 1)
    window = max(5, window)
    polyorder = min(3, window - 2)

    smooth = np.empty_like(X)
    dXdt = np.empty_like(X)
    noise_ratios = []
    for j in range(X.shape[1]):
        smooth[:, j] = savgol_filter(
            X[:, j], window_length=window, polyorder=polyorder)
        # Supplying the actual time vector avoids silently assuming uniform dt.
        dXdt[:, j] = np.gradient(smooth[:, j], t)

        # Robust, dimensionless noise estimate. Unlike the old FFT code, this
        # does not confuse a large physical spectral peak with sensor noise.
        residual = X[:, j] - smooth[:, j]
        residual_sigma = 1.4826 * np.median(
            np.abs(residual - np.median(residual)))
        state_scale = 1.4826 * np.median(
            np.abs(X[:, j] - np.median(X[:, j])))
        if state_scale < 1e-12:
            state_scale = max(float(np.std(X[:, j])), 1e-12)
        noise_ratios.append(float(residual_sigma / state_scale))

    return X, dXdt, float(np.median(noise_ratios))


def _prepare_blocked_validation(dataframes, train_fraction=0.7):
    """Differentiate and split every trajectory independently."""
    train_X, train_dX, val_X, val_dX = [], [], [], []
    noise_levels = []
    irregular = False

    for df in dataframes:
        X, dXdt, noise = _smooth_derivative_and_noise(df)
        t = np.asarray(df.iloc[:, 0], dtype=float)
        dt = np.diff(t)
        irregular = irregular or (
            np.std(dt) / max(abs(float(np.mean(dt))), 1e-12) > 0.02)

        split = int(len(X) * train_fraction)
        split = min(max(split, 5), len(X) - 3)
        train_X.append(X[:split])
        train_dX.append(dXdt[:split])
        val_X.append(X[split:])
        val_dX.append(dXdt[split:])
        noise_levels.append(noise)

    return {
        "X_train": np.vstack(train_X),
        "dX_train": np.vstack(train_dX),
        "X_val": np.vstack(val_X),
        "dX_val": np.vstack(val_dX),
        "noise_ratio": float(np.median(noise_levels)),
        "irregular_time": irregular,
    }


def _candidate_grid(libraries, thresholds):
    for kind in libraries:
        degrees = (1, 2, 3) if kind == "Polynomial" else (1, 2)
        for degree in degrees:
            for threshold in thresholds:
                yield kind, degree, float(threshold)


def _fit_candidate(kind, degree, threshold, prepared, state_names):
    optimizer = ps.STLSQ(
        threshold=threshold,
        normalize_columns=False,
    )
    model = ps.SINDy(
        optimizer=optimizer,
        feature_library=_library(kind, degree),
        differentiation_method=ps.FiniteDifference(),
    )
    t_dummy = np.arange(len(prepared["X_train"]), dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            prepared["X_train"],
            t=t_dummy,
            x_dot=prepared["dX_train"],
            feature_names=state_names,
        )

    prediction = np.asarray(model.predict(prepared["X_val"]), dtype=float)
    truth = prepared["dX_val"]
    per_state_scale = np.std(truth, axis=0)
    per_state_scale = np.where(per_state_scale > 1e-12, per_state_scale, 1.0)
    per_state_rmse = np.sqrt(np.mean((truth - prediction) ** 2, axis=0))
    nrmse = float(np.mean(per_state_rmse / per_state_scale))
    r2 = float(r2_score(truth, prediction, multioutput="uniform_average"))
    coefficients = np.asarray(model.coefficients())
    active_terms = int(np.count_nonzero(np.abs(coefficients) > 1e-12))
    total_terms = int(coefficients.size)

    # A zero equation is never a useful recommendation, even when a nearly
    # stationary validation tail makes its raw error look deceptively small.
    zero_penalty = 1.0 if active_terms == 0 else 0.0
    terms_per_state = active_terms / max(1, truth.shape[1])
    balanced_score = nrmse + 0.015 * terms_per_state + zero_penalty
    return {
        "library": kind,
        "degree": int(degree),
        "threshold": float(threshold),
        "val_nrmse": nrmse,
        "val_r2": r2,
        "active_terms": active_terms,
        "total_terms": total_terms,
        "balanced_score": float(balanced_score),
    }


def scout_hyperparameters(
    data,
    libraries=("Polynomial", "Fourier", "Combined"),
    thresholds=DEFAULT_THRESHOLDS,
    train_fraction=0.7,
):
    """Evaluate a compact SINDy search and return three starting profiles.

    ``data`` may be one DataFrame or multiple trajectories. Each trajectory
    is differentiated and split independently, so time resets never create
    artificial derivative seams.
    """
    frames = [data] if isinstance(data, pd.DataFrame) else list(data)
    if not frames:
        raise ValueError("At least one trajectory is required.")
    state_names = list(frames[0].columns[1:])
    for frame in frames:
        if list(frame.columns[1:]) != state_names:
            raise ValueError(
                "All trajectories must have identical state columns in the same order.")

    prepared = _prepare_blocked_validation(frames, train_fraction)
    candidates = []
    failures = []
    for kind, degree, threshold in _candidate_grid(libraries, thresholds):
        try:
            result = _fit_candidate(
                kind, degree, threshold, prepared, state_names)
            if np.isfinite(result["val_nrmse"]):
                candidates.append(result)
        except Exception as exc:
            failures.append(f"{kind} d={degree} λ={threshold:g}: {exc}")

    nonzero = [item for item in candidates if item["active_terms"] > 0]
    selectable = nonzero or candidates
    if not selectable:
        detail = failures[0] if failures else "no finite candidate scores"
        raise ValueError(f"Data Scout could not fit a candidate: {detail}")

    best_fit = min(
        selectable,
        key=lambda item: (item["val_nrmse"], item["active_terms"]),
    )
    balanced = min(
        selectable,
        key=lambda item: (item["balanced_score"], item["active_terms"]),
    )
    accuracy_limit = best_fit["val_nrmse"] * 1.15 + 0.01
    near_best = [
        item for item in selectable if item["val_nrmse"] <= accuracy_limit
    ] or [best_fit]
    simplest = min(
        near_best,
        key=lambda item: (
            item["active_terms"], item["val_nrmse"], item["degree"]),
    )

    ranked = sorted(selectable, key=lambda item: item["val_nrmse"])
    distinct_alternatives = [
        item for item in ranked[1:]
        if (item["library"], item["degree"])
        != (ranked[0]["library"], ranked[0]["degree"])
    ]
    if not distinct_alternatives:
        confidence = "Low"
    else:
        gap = ((distinct_alternatives[0]["val_nrmse"] - ranked[0]["val_nrmse"])
               / max(ranked[0]["val_nrmse"], 1e-12))
        confidence = (
            "High" if gap >= 0.20 else "Moderate" if gap >= 0.05 else "Low")

    notes = []
    if prepared["irregular_time"]:
        notes.append("Time spacing is irregular; local gradients were used.")
    if prepared["noise_ratio"] >= 0.10:
        notes.append(
            "High relative noise: prefer the simpler profile and inspect residuals.")
    if best_fit["val_nrmse"] >= 0.50 or best_fit["val_r2"] < 0.60:
        notes.append(
            "Weak held-out fit: treat every profile as low-confidence and tune manually.")
    if failures:
        notes.append(f"{len(failures)} candidate fit(s) were skipped.")

    return {
        "profiles": {
            "simplest": simplest,
            "balanced": balanced,
            "best_fit": best_fit,
        },
        "candidate_count": len(candidates),
        "failed_count": len(failures),
        "noise_ratio": prepared["noise_ratio"],
        "confidence": confidence,
        "notes": notes,
    }


def analyze_data_linearity(df):
    """Backward-compatible tuple API using the balanced scout profile."""
    try:
        report = scout_hyperparameters(df)
        choice = report["profiles"]["balanced"]
        reason = (
            f"{report['confidence']} confidence; "
            f"{choice['library']}, degree/harmonics {choice['degree']}, "
            f"threshold {choice['threshold']:.3f}; "
            f"blocked-validation NRMSE {choice['val_nrmse']:.3f}; "
            f"{choice['active_terms']} active terms; "
            f"relative noise {report['noise_ratio']:.3f}."
        )
        return (
            choice["library"],
            choice["degree"],
            choice["threshold"],
            reason,
        )
    except Exception as exc:
        return "Polynomial", 1, 0.10, f"Error analyzing data: {exc}"
