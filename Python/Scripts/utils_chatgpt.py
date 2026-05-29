import numpy as np
import pandas as pd


# ----------------------------------------------------------
# Utility functions
# ----------------------------------------------------------

def linear_slope(times, scores):
    """
    Compute the linear regression slope for an ALSFRS-R time series.

    Fits: score(t) = m * t + c

    :param times: Array of timestamps (numeric, irregular allowed).
    :type times: array-like
    :param scores: Array of ALSFRS-R scores corresponding to the timestamps.
    :type scores: array-like
    :return: (slope m, intercept c)
    :rtype: tuple(float, float)
    """
    if len(times) < 2:
        return np.nan, np.nan

    t = times - times.mean()
    A = np.vstack([t, np.ones_like(t)]).T
    m, c = np.linalg.lstsq(A, scores, rcond=None)[0]
    return float(m), float(c - m * times.mean())


def find_best_change_point(times, scores, min_segment=2):
    """
    Detect a single change-point in an ALSFRS-R trajectory by fitting
    two linear models on both sides of a candidate split and choosing the best one.

    :param times: Sorted array of timestamps.
    :type times: array-like
    :param scores: Sorted array of ALSFRS-R scores.
    :type scores: array-like
    :param min_segment: Minimum segment length on each side of the split.
    :type min_segment: int
    :return: (index_of_CP, improvement_ratio, ((slopeL, intL), (slopeR, intR)))
             or (None, None, None) if no improvement found.
    :rtype: tuple or None
    """
    n = len(times)
    if n < 2 * min_segment + 1:
        return None, None, None

    m0, c0 = linear_slope(times, scores)
    rss0 = np.sum((scores - (m0 * times + c0))**2)

    best_idx = None
    best_rss = None
    best_params = None

    for i in range(min_segment, n - min_segment):
        m1, c1 = linear_slope(times[:i+1], scores[:i+1])
        rss1 = np.sum((scores[:i+1] - (m1 * times[:i+1] + c1))**2)

        m2, c2 = linear_slope(times[i+1:], scores[i+1:])
        rss2 = np.sum((scores[i+1:] - (m2 * times[i+1:] + c2))**2)

        total_rss = rss1 + rss2

        if best_rss is None or total_rss < best_rss:
            best_rss = total_rss
            best_idx = i
            best_params = ((m1, c1), (m2, c2))

    improvement = (rss0 - best_rss) / (rss0 + 1e-9)
    if improvement < 0.15:
        return None, None, None

    return best_idx, improvement, best_params


def time_to_threshold(times, scores, thr):
    """
    Compute the estimated time at which ALSFRS-R score crosses a threshold
    using linear interpolation between timepoints.

    :param times: Array of timestamps.
    :type times: array-like
    :param scores: Array of scores, same length as times.
    :type scores: array-like
    :param thr: Score threshold to detect crossing.
    :type thr: float
    :return: Time of threshold crossing, or NaN if not crossed.
    :rtype: float
    """
    order = np.argsort(times)
    t = times[order]
    s = scores[order]

    for i in range(len(s) - 1):
        if s[i] >= thr and s[i+1] < thr:
            frac = (s[i] - thr) / (s[i] - s[i+1])
            return float(t[i] + frac * (t[i+1] - t[i]))

    if s[-1] < thr:
        return float(t[-1])

    return np.nan

def extract_features_single(times, scores, thresholds=[40, 30, 20, 10]):
    """
    Extract a wide set of ALSFRS-R temporal features used in ALS literature
    for clustering or progression modeling.

    :param times: Timestamps for a single patient (numeric).
    :type times: array-like
    :param scores: ALSFRS-R scores corresponding to timestamps.
    :type scores: array-like
    :param thresholds: Threshold values to compute time-to-threshold features.
    :type thresholds: list of floats
    :return: Dictionary of extracted features.
    :rtype: dict
    """
    times = np.asarray(times, float)
    scores = np.asarray(scores, float)

    idx = np.argsort(times)
    times = times[idx]
    scores = scores[idx]

    n = len(times)
    out = {}

    # --- Basic features ---
    out["n_visits"] = n
    out["time_span"] = times[-1] - times[0] if n > 1 else 0
    out["first"] = scores[0]
    out["last"] = scores[-1]
    out["mean"] = float(np.mean(scores))
    out["median"] = float(np.median(scores))
    out["min"] = float(np.min(scores))
    out["max"] = float(np.max(scores))
    out["range"] = out["max"] - out["min"]

    # --- Linear slope ---
    slope, intercept = linear_slope(times, scores)
    out["slope"] = slope
    out["intercept"] = intercept

    # --- Interval slopes ---
    if n >= 2:
        dt = np.diff(times)
        ds = np.diff(scores)
        slopes = ds / dt

        out["mean_interval_slope"] = float(np.mean(slopes))
        out["std_interval_slope"] = float(np.std(slopes))
        out["max_drop_per_time"] = float(np.min(slopes))
        out["max_increase_per_time"] = float(np.max(slopes))
        out["max_abs_change_per_time"] = float(np.max(np.abs(slopes)))
    else:
        out["mean_interval_slope"] = np.nan
        out["std_interval_slope"] = np.nan
        out["max_drop_per_time"] = np.nan
        out["max_increase_per_time"] = np.nan
        out["max_abs_change_per_time"] = np.nan

    # --- Early / late slope ---
    if n >= 3:
        out["slope_early"], _ = linear_slope(times[:3], scores[:3])
        out["slope_late"], _ = linear_slope(times[-3:], scores[-3:])
    else:
        out["slope_early"] = np.nan
        out["slope_late"] = np.nan

    # --- Curvature (quadratic) ---
    if n >= 3:
        t0 = times - times.mean()
        a, b, c = np.polyfit(t0, scores, 2)
        out["quad_a"] = float(a)
        out["quad_b"] = float(b)
        out["quad_c"] = float(c)
    else:
        out["quad_a"] = np.nan
        out["quad_b"] = np.nan
        out["quad_c"] = np.nan

    # --- Change point ---
    cp_idx, improvement, params = find_best_change_point(times, scores)
    if cp_idx is not None:
        out["change_point_index"] = cp_idx
        out["change_point_time"] = float(times[cp_idx])
        out["change_point_improvement"] = float(improvement)
        out["slope_left"] = float(params[0][0])
        out["slope_right"] = float(params[1][0])
    else:
        out["change_point_index"] = np.nan
        out["change_point_time"] = np.nan
        out["change_point_improvement"] = np.nan
        out["slope_left"] = np.nan
        out["slope_right"] = np.nan

    # --- Time-to-threshold ---
    for thr in thresholds:
        out[f"time_to_thr_{thr}"] = time_to_threshold(times, scores, thr)

    # --- Percent decline ---
    out["pct_decline"] = (out["first"] - out["last"]) / (abs(out["first"]) + 1e-9)

    return out