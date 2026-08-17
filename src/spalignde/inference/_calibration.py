"""Private gene-specific mismatch-risk calibration.

The public API intentionally does not expose calibration knobs.  This module
contains the single implementation used by the notebook-derived fitting
kernel and by focused regression tests.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats


MISMATCH_CALIBRATION_MODE = "local_tnull_MAD_isotonic_quadratic"
MULTI_CONTRAST_CALIBRATION_MODE = (
    "per_contrast_local_calibration_then_equal_weight_huber"
)


def _pava_increasing(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted pool-adjacent-violators fit constrained to be nondecreasing."""

    y = np.asarray(values, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    if y.size != w.size:
        raise ValueError("values and weights must have the same length.")
    if y.size == 0:
        return y.copy()
    y = np.where(np.isfinite(y), y, 0.0)
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0)

    levels: list[float] = []
    block_weights: list[float] = []
    starts: list[int] = []
    ends: list[int] = []
    for index, (level, weight) in enumerate(zip(y, w)):
        levels.append(float(level))
        block_weights.append(float(weight))
        starts.append(index)
        ends.append(index)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            merged_weight = block_weights[-2] + block_weights[-1]
            merged_level = (
                levels[-2] * block_weights[-2]
                + levels[-1] * block_weights[-1]
            ) / merged_weight
            merged_start = starts[-2]
            merged_end = ends[-1]
            levels[-2:] = [merged_level]
            block_weights[-2:] = [merged_weight]
            starts[-2:] = [merged_start]
            ends[-2:] = [merged_end]

    fitted = np.zeros(y.size, dtype=float)
    for level, start, end in zip(levels, starts, ends):
        fitted[start : end + 1] = level
    return fitted


def _risk_bin_rows(
    r01: np.ndarray,
    valid: np.ndarray,
    *,
    bins: int,
    min_bin_n: int,
) -> list[dict[str, object]]:
    """Create one zero-risk bin and quantile bins over positive risk."""

    bins = max(int(bins), 2)
    min_bin_n = max(int(min_bin_n), 1)
    rows: list[dict[str, object]] = []

    zero = valid & (r01 <= 1e-12)
    if int(zero.sum()) >= min_bin_n:
        rows.append({
            "bin": "zero",
            "n": int(zero.sum()),
            "r_mid": 0.0,
            "r_min": 0.0,
            "r_max": 0.0,
            "idx": zero,
        })

    positive = valid & (r01 > 1e-12)
    n_positive = int(positive.sum())
    if n_positive >= min_bin_n:
        n_positive_bins = min(
            max(bins - len(rows), 1),
            max(1, n_positive // min_bin_n),
        )
        breaks = np.unique(
            np.nanquantile(r01[positive], np.linspace(0, 1, n_positive_bins + 1))
        )
        if breaks.size >= 2:
            for index, (lower, upper) in enumerate(zip(breaks[:-1], breaks[1:])):
                if index == breaks.size - 2:
                    in_bin = positive & (r01 >= lower) & (r01 <= upper)
                else:
                    in_bin = positive & (r01 >= lower) & (r01 < upper)
                if int(in_bin.sum()) < min_bin_n:
                    continue
                rows.append({
                    "bin": f"pos{index + 1}",
                    "n": int(in_bin.sum()),
                    "r_mid": float(np.nanmedian(r01[in_bin])),
                    "r_min": float(lower),
                    "r_max": float(upper),
                    "idx": in_bin,
                })
    return sorted(rows, key=lambda row: float(row["r_mid"]))


def calibrate_mismatch_variance(
    tvec: np.ndarray,
    use_mask: np.ndarray,
    r01: np.ndarray,
    global_score: float,
    df: float,
    *,
    bins: int = 10,
    min_bin_n: int = 200,
    p_grid: float | Sequence[float] = 2.0,
    tau_anchor_q: float = 0.80,
    slack: float = 1.0,
    lam_local_cap: float = 5e4,
    eps: float = 1e-12,
    verbose: bool = False,
    **_deprecated: object,
) -> dict[str, object]:
    """Estimate gene-specific local mismatch variance inflation.

    Initial local statistics obtained without mismatch inflation are binned by
    normalized local risk. Within each bin, the median is removed and the raw
    MAD is divided by the Student-t(df) null MAD, ``t.ppf(0.75, df)``. Positive
    excess variance is made nondecreasing with risk. A weighted quadratic fit
    through the origin is then rescaled at the risk bin nearest the requested
    anchor quantile.
    The bounded anchor factor makes the final coefficient no larger than the
    through-origin slope. The origin constraint excludes risk-independent
    excess variance, so only the spatially varying local-risk term is applied.

    Extra keyword arguments are accepted only for compatibility with the
    legacy private call site; obsolete tail/global-cap arguments have no effect.
    """

    tvec = np.asarray(tvec, dtype=float).ravel()
    use_mask = np.asarray(use_mask, dtype=bool).ravel()
    r01 = np.asarray(r01, dtype=float).ravel()
    if not (tvec.size == use_mask.size == r01.size):
        raise ValueError("tvec, use_mask, and r01 must have the same length.")
    r01 = np.clip(r01, 0.0, 1.0)
    df = float(df)
    if not np.isfinite(df) or df < 5:
        df = 30.0
    bins = int(bins)
    min_bin_n = int(min_bin_n)
    min_bins_required = 4
    valid = use_mask & np.isfinite(tvec) & np.isfinite(r01)
    n_valid = int(valid.sum())

    def empty_result(mode: str, message: str, **diag_extra: object) -> dict[str, object]:
        diag = {
            "status": "failed",
            "mode": mode,
            "msg": message,
            "n_ok": n_valid,
            "df": df,
            **diag_extra,
        }
        return {
            "tau_hat": 0.0,
            "lambda_local_hat": 0.0,
            "lambda_global_hat": 0.0,
            "global_score": float(global_score),
            "p_best": np.nan,
            "diag": diag,
        }

    if n_valid < max(500, min_bins_required * min_bin_n):
        return empty_result(
            "mad_null1_A0_calibration_too_few_ok",
            "too few valid grid locations",
        )

    rows0 = _risk_bin_rows(r01, valid, bins=bins, min_bin_n=min_bin_n)
    q75_null = float(stats.t.ppf(0.75, df=df))
    rows: list[dict[str, object]] = []
    if np.isfinite(q75_null) and q75_null > 0:
        for source in rows0:
            indices = np.asarray(source["idx"], dtype=bool)
            values = tvec[indices]
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            center = float(np.nanmedian(values))
            raw_mad = float(np.nanmedian(np.abs(values - center)))
            robust_scale = raw_mad / q75_null
            if not np.isfinite(robust_scale) or robust_scale < 0:
                continue
            at_zero = float(source["r_mid"]) <= 1e-12
            excess = 0.0 if at_zero else max(0.0, robust_scale**2 - 1.0)
            row = {key: value for key, value in source.items() if key != "idx"}
            row.update({
                # MAD-scale diagnostics.
                "scale_squared_relative_to_null": 1.0 + excess,
                "scale_excess_raw": excess,
                "center_t": center,
                "raw_mad_t": raw_mad,
                "null_mad_t": q75_null,
                "scale_ratio_to_null": robust_scale,
                # Backward-compatible aliases retained for notebook payloads.
                "tail_s2_rel": 1.0 + excess,
                "tail_excess_raw": excess,
                "tail_rate_mean": np.nan,
                "tail_excess_rate_mean": np.nan,
                "tail_alpha_grid": tuple(),
                "base_tail_rate_mean": np.nan,
                "base_tail_s2_mean": 1.0,
                "robust_center_t": center,
                "robust_MAD_t": raw_mad,
                "robust_scale_t": robust_scale,
                "theoretical_null_scale": 1.0,
                "zero_risk_target_forced_zero": bool(at_zero),
            })
            rows.append(row)

    if len(rows) < min_bins_required:
        return empty_result(
            "mad_null1_A0_calibration_too_few_bins",
            "too few risk bins retained",
            n_bins=int(len(rows)),
            bins_kept=int(len(rows)),
            df_bin=pd.DataFrame(rows),
        )

    df_bin = pd.DataFrame(rows).sort_values("r_mid").reset_index(drop=True)
    r_mid = df_bin["r_mid"].to_numpy(dtype=float)
    y_raw = np.maximum(df_bin["scale_excess_raw"].to_numpy(dtype=float), 0.0)
    weights = np.maximum(df_bin["n"].to_numpy(dtype=float), 1.0)
    y_iso = _pava_increasing(y_raw, weights)
    df_bin["scale_excess_isotonic"] = y_iso
    # Backward-compatible alias retained for existing diagnostic payloads.
    df_bin["tail_excess_iso"] = y_iso

    def fit_one_power(power: float) -> dict[str, float]:
        power = float(power)
        x = np.maximum(r_mid, 0.0) ** power
        denominator = float(np.sum(weights * x * x))
        slope = (
            max(0.0, float(np.sum(weights * x * y_iso) / denominator))
            if denominator > eps
            else 0.0
        )
        fitted = np.maximum(slope * x, 0.0)
        loss = float(
            np.sum(weights * (np.log1p(y_iso) - np.log1p(fitted)) ** 2)
        )
        return {"p": power, "A": 0.0, "B": slope, "loss": loss}

    fits = [fit_one_power(power) for power in np.atleast_1d(p_grid)]
    best = min(fits, key=lambda fit: fit["loss"])
    p_best = float(best["p"])
    local_slope = min(float(best["B"]), 1e6)
    r_anchor = float(np.nanquantile(r_mid, float(tau_anchor_q)))
    anchor_index = int(np.argmin(np.abs(r_mid - r_anchor)))
    r_anchor = float(r_mid[anchor_index])
    y_anchor = float(y_iso[anchor_index])
    denominator = local_slope * max(r_anchor, 0.0) ** p_best
    if not np.isfinite(denominator) or denominator <= 0 or not np.isfinite(y_anchor):
        tau_hat = 0.0
    else:
        tau_hat = y_anchor / denominator
        if not np.isfinite(tau_hat):
            tau_hat = 0.0
        tau_hat = min(max(0.0, tau_hat) / max(float(slack), 1.0), 1.0)

    lambda_local = min(
        max(tau_hat * local_slope, 0.0),
        float(lam_local_cap),
    )
    lambda_global = 0.0
    diag = {
        "status": "success",
        "mode": MISMATCH_CALIBRATION_MODE,
        "n_ok": n_valid,
        "df": df,
        "p_best": p_best,
        "A": 0.0,
        "B": local_slope,
        "tau": tau_hat,
        "r_anchor": r_anchor,
        "anchor_scale_squared": 1.0 + y_anchor,
        "anchor_excess": y_anchor,
        "s2_anchor": 1.0 + y_anchor,
        "tail_excess_anchor": y_anchor,
        "tail_alpha_grid": tuple(),
        "bins_kept": int(len(df_bin)),
        "df_bin": df_bin,
        "lambda_local_hat": lambda_local,
        "lambda_global_hat": lambda_global,
    }
    if verbose:
        print(
            "[risk-calib-local-MAD] bins={} df={:.1f} p={} B={:.6g} "
            "tau={:.6g} lambda_g={:.6g}".format(
                len(df_bin), df, p_best, local_slope, tau_hat, lambda_local
            )
        )
    return {
        "tau_hat": tau_hat,
        "lambda_local_hat": lambda_local,
        "lambda_global_hat": lambda_global,
        "global_score": float(global_score),
        "p_best": p_best,
        "diag": diag,
    }


def validate_provisional_calibration(
    calibration: dict[str, object] | None,
    *,
    min_bin_n: int = 200,
    lam_local_cap: float = 5e4,
) -> dict[str, object]:
    """Validate one contrast-specific provisional mismatch calibration."""

    calibration = calibration if isinstance(calibration, dict) else {}
    diag = calibration.get("diag")
    diag = diag if isinstance(diag, dict) else {}
    reasons: list[str] = []
    if diag.get("status") != "success" or diag.get("mode") != MISMATCH_CALIBRATION_MODE:
        reasons.append("calibration_not_successful")
    n_valid = int(diag.get("n_ok", 0) or 0)
    min_required = max(500, 4 * int(min_bin_n))
    if n_valid < min_required:
        reasons.append("too_few_usable_locations")
    bins_kept = int(diag.get("bins_kept", 0) or 0)
    if bins_kept < 4:
        reasons.append("too_few_risk_bins")
    frame = diag.get("df_bin")
    try:
        risk_midpoints = np.asarray(frame["r_mid"], dtype=float).ravel()
    except Exception:
        risk_midpoints = np.asarray([], dtype=float)
    risk_midpoints = risk_midpoints[np.isfinite(risk_midpoints)]
    unique_risk_bins = int(np.unique(np.round(risk_midpoints, 12)).size)
    if (
        risk_midpoints.size < 4
        or unique_risk_bins < 4
        or not np.any(risk_midpoints > 1e-12)
    ):
        reasons.append("insufficient_risk_support")

    lambda_local = float(calibration.get("lambda_local_hat", np.nan))
    lambda_global = float(calibration.get("lambda_global_hat", np.nan))
    cap = float(lam_local_cap)
    if (
        not np.isfinite(lambda_local)
        or lambda_local < 0
        or lambda_local > cap * (1.0 + 1e-10) + 1e-12
    ):
        reasons.append("invalid_local_lambda")
    if not np.isfinite(lambda_global) or abs(lambda_global) > 1e-10:
        reasons.append("nonzero_global_lambda")
    for key in ("p_best", "B", "tau"):
        if not np.isfinite(float(diag.get(key, np.nan))):
            reasons.append(f"invalid_{key}")
    tau = float(diag.get("tau", np.nan))
    slope = float(diag.get("B", np.nan))
    if np.isfinite(tau) and not 0.0 <= tau <= 1.0:
        reasons.append("tau_out_of_range")
    if np.isfinite(slope) and slope < 0:
        reasons.append("negative_B")

    return {
        "valid": not reasons,
        "reasons": tuple(dict.fromkeys(reasons)),
        "n_ok": n_valid,
        "bins_kept": bins_kept,
        "n_unique_risk_bins": unique_risk_bins,
        "lambda_local_hat": lambda_local,
        "at_local_cap": bool(
            np.isfinite(lambda_local)
            and np.isclose(lambda_local, cap, rtol=1e-10, atol=1e-12)
        ),
    }


def huber_center_nonnegative(
    values: Sequence[float],
    *,
    kappa: float = 1.345,
    scale_floor: float = 1e-8,
    max_iter: int = 100,
) -> tuple[float, dict[str, object]]:
    """Return the equal-weight Huber center of nonnegative coefficients."""

    coefficients = np.asarray(values, dtype=float).ravel()
    coefficients = coefficients[
        np.isfinite(coefficients) & (coefficients >= 0)
    ]
    if coefficients.size == 0:
        return 0.0, {"status": "no_valid_contrasts", "n_valid": 0}
    median = float(np.median(coefficients))
    mad = float(np.median(np.abs(coefficients - median)))
    scale = max(
        1.4826 * mad,
        float(scale_floor) * (1.0 + abs(median)),
    )
    lower = float(np.min(coefficients))
    upper = float(np.max(coefficients))
    if upper <= lower + np.finfo(float).eps * (1.0 + abs(lower)):
        center = lower
        iterations = 0
    else:
        def score(candidate: float) -> float:
            residual = (coefficients - float(candidate)) / scale
            return float(np.sum(np.clip(residual, -float(kappa), float(kappa))))

        iterations = 0
        for iterations in range(1, int(max_iter) + 1):
            midpoint = 0.5 * (lower + upper)
            if score(midpoint) > 0:
                lower = midpoint
            else:
                upper = midpoint
        center = 0.5 * (lower + upper)
    center = max(float(center), 0.0)
    return center, {
        "status": "success",
        "method": "equal_weight_huber_location",
        "n_valid": int(coefficients.size),
        "kappa": float(kappa),
        "median": median,
        "mad": mad,
        "robust_scale": scale,
        "iterations": int(iterations),
    }
