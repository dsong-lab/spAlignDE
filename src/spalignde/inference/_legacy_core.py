"""Legacy computational kernel extracted from the research notebook.

This module is intentionally private. Public APIs live in the sibling modules.
It remains notebook-derived, while focused components shared with tests are
maintained in dedicated private helpers.
"""

import copy
import math
import re
import time
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from scipy import sparse, stats
from scipy.interpolate import BSpline
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from ._calibration import (
    MULTI_CONTRAST_CALIBRATION_MODE,
    calibrate_mismatch_variance,
    huber_center_nonnegative,
    validate_provisional_calibration,
)

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from shapely.geometry import MultiPoint, Point, Polygon, box
    from shapely.ops import unary_union
    try:
        from shapely import concave_hull as shapely_concave_hull
    except ImportError:
        shapely_concave_hull = None
except ImportError:
    MultiPoint = None
    Point = None
    Polygon = None
    box = None
    unary_union = None
    shapely_concave_hull = None


def or_else(a, b):
    return b if a is None else a


def require_pandas():
    if pd is None:
        raise ImportError("This translated Python version requires pandas for table operations.")


def require_shapely():
    if Polygon is None or unary_union is None:
        raise ImportError("This translated Python version requires shapely for polygon and geometry operations.")


def gauss_w(d, R):
    d = np.asarray(d, dtype=float)
    return np.exp(-0.5 * (d / np.maximum(float(R), 1e-12)) ** 2)


def mad(x, center=None, constant=1.4826):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    if center is None:
        center = np.median(x)
    return constant * np.median(np.abs(x - center))


def safe_standardize(x, center="median", scale="mad", eps=1e-8):
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    z = np.zeros_like(x, dtype=float)
    if not np.any(ok):
        return z
    c0 = np.median(x[ok]) if center == "median" else float(np.mean(x[ok]))
    if scale == "mad":
        s0 = mad(x[ok], center=c0, constant=1.4826)
    else:
        s0 = float(np.std(x[ok], ddof=1)) if np.sum(ok) >= 2 else 0.0
    if not np.isfinite(s0) or s0 < eps:
        return z
    z[ok] = (x[ok] - c0) / s0
    z[~np.isfinite(z)] = 0
    return z


def safe_standardize_vec(x, eps=1e-8):
    return safe_standardize(x, center="mean", scale="sd", eps=eps)


def row_sds(X):
    X = np.asarray(X, dtype=float)
    return np.nanstd(X, axis=1, ddof=1)


def col_vars(X):
    X = np.asarray(X, dtype=float)
    return np.nanvar(X, axis=0, ddof=1)


def scale_block(X, eps=1e-8):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    out = np.zeros_like(X, dtype=float)
    for j in range(X.shape[1]):
        out[:, j] = safe_standardize_vec(X[:, j], eps=eps)
    out[~np.isfinite(out)] = 0
    return out


def scale_global(X, eps=1e-8):
    X = np.asarray(X, dtype=float)
    vv = X[np.isfinite(X)]
    if vv.size == 0:
        return np.zeros_like(X, dtype=float)
    mu = float(np.mean(vv))
    sdv = float(np.std(vv, ddof=1)) if vv.size >= 2 else 0.0
    if not np.isfinite(sdv) or sdv < eps:
        return np.zeros_like(X, dtype=float)
    out = (X - mu) / sdv
    out[~np.isfinite(out)] = 0
    return out


def zcols(X):
    return scale_block(X)


def p_adjust_bh(p):
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    ok = np.isfinite(p)
    pv = p[ok]
    if pv.size == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order]
    n = ranked.size
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    tmp = np.empty_like(q)
    tmp[order] = q
    out[ok] = tmp
    return out


def _celltype_js_adjustment(p_target, p_reference):
    """Return the normalized Jensen--Shannon cell-type adjustment.

    Rows are shared-grid locations and columns are cell-type proportions.  The
    square root of the Jensen--Shannon divergence normalized by ``log(2)`` is
    a bounded distance in ``[0, 1]``.  Its exponential is the variance factor;
    the reciprocal is the precision multiplier used by the weighted fit.
    """

    p_target = np.asarray(p_target, dtype=float)
    p_reference = np.asarray(p_reference, dtype=float)
    if p_target.shape != p_reference.shape or p_target.ndim != 2:
        raise ValueError("Cell-type proportion arrays must be two-dimensional and have equal shapes.")
    if np.any(~np.isfinite(p_target)) or np.any(~np.isfinite(p_reference)):
        raise ValueError("Cell-type proportion arrays must contain only finite values.")
    if np.any(p_target < 0) or np.any(p_reference < 0):
        raise ValueError("Cell-type proportions cannot be negative.")

    p_target = p_target / np.maximum(np.sum(p_target, axis=1, keepdims=True), 1e-12)
    p_reference = p_reference / np.maximum(np.sum(p_reference, axis=1, keepdims=True), 1e-12)
    midpoint = 0.5 * (p_target + p_reference)
    term_target = np.where(
        p_target > 0,
        p_target * np.log(np.maximum(p_target, 1e-300) / np.maximum(midpoint, 1e-300)),
        0.0,
    )
    term_reference = np.where(
        p_reference > 0,
        p_reference * np.log(np.maximum(p_reference, 1e-300) / np.maximum(midpoint, 1e-300)),
        0.0,
    )
    divergence = np.clip(
        (0.5 * np.sum(term_target, axis=1) + 0.5 * np.sum(term_reference, axis=1))
        / np.log(2.0),
        0.0,
        1.0,
    )
    distance = np.sqrt(divergence)
    variance_factor = np.exp(distance)
    precision = np.exp(-distance)
    return divergence, distance, variance_factor, precision


def ecdf_eval(sample, x):
    sample = np.sort(np.asarray(sample, dtype=float))
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return np.zeros_like(np.asarray(x, dtype=float))
    x = np.asarray(x, dtype=float)
    return np.searchsorted(sample, x, side="right") / sample.size


def infer_time_numeric(x):
    out = []
    for item in list(x):
        m = re.search(r"(\d+(?:\.\d+)?)$", str(item))
        out.append(float(m.group(1)) if m else np.nan)
    return np.asarray(out, dtype=float)


def parse_age_numeric(x):
    out = []
    for item in list(x):
        s = str(item)
        m = re.search(r"age\s*([0-9]+\.?[0-9]*)", s, flags=re.IGNORECASE)
        if m:
            out.append(float(m.group(1)))
            continue
        m2 = re.search(r"([0-9]+\.?[0-9]*)", s)
        out.append(float(m2.group(1)) if m2 else np.nan)
    return np.asarray(out, dtype=float)


def col_cor_pairwise(A, B):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.shape != B.shape:
        raise ValueError("A and B must have identical shapes.")
    n = A.shape[0]
    if n < 2:
        return np.full(A.shape[1], np.nan, dtype=float)
    muA = np.nanmean(A, axis=0)
    muB = np.nanmean(B, axis=0)
    Ac = A - muA
    Bc = B - muB
    sA = np.sqrt(np.nansum(Ac ** 2, axis=0) / max(n - 1, 1))
    sB = np.sqrt(np.nansum(Bc ** 2, axis=0) / max(n - 1, 1))
    num = np.nansum(Ac * Bc, axis=0)
    den = np.maximum((n - 1) * sA * sB, 1e-12)
    out = num / den
    bad = (~np.isfinite(sA)) | (~np.isfinite(sB)) | (sA < 1e-6) | (sB < 1e-6)
    out[bad] = np.nan
    out[~np.isfinite(out)] = np.nan
    return out


def bspline_basis(x, df=8, degree=3, intercept=True):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be one-dimensional.")
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xmax <= xmin:
        xmax = xmin + 1.0
    df_use = int(min(int(df), max(3, x.size)))
    degree_use = int(min(int(degree), max(1, x.size - 1)))
    n_internal = max(df_use - degree_use - 1, 0)
    if n_internal > 0:
        interior = np.linspace(xmin, xmax, n_internal + 2)[1:-1]
    else:
        interior = np.array([], dtype=float)
    knots = np.r_[np.repeat(xmin, degree_use + 1), interior, np.repeat(xmax, degree_use + 1)]
    n_basis = len(knots) - degree_use - 1
    eye = np.eye(n_basis)
    basis = np.column_stack([
        BSpline(knots, eye[i], degree_use, extrapolate=True)(x)
        for i in range(n_basis)
    ])
    if not intercept and basis.shape[1] > 0:
        basis = basis[:, 1:]
    return basis


def smooth_rows_bs_ridge_beta(X, df=8, degree=3, ridge=1e-2):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    Tn = X.shape[1]
    if Tn < 2:
        raise ValueError("Need at least 2 time points.")
    B = bspline_basis(np.arange(1, Tn + 1), df=df, degree=degree, intercept=True)
    K = B.T @ B + np.eye(B.shape[1]) * float(ridge)
    RHS = B.T @ X.T
    try:
        beta_t = np.linalg.solve(K, RHS)
    except np.linalg.LinAlgError:
        beta_t = np.linalg.lstsq(K, RHS, rcond=None)[0]
    beta = beta_t.T
    X_smooth = beta @ B.T
    return {"beta": beta, "X_smooth": X_smooth, "basis": B}


def spatial_contrast_score(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2:
        return 0.0
    rsd = row_sds(X)
    rsd = rsd[np.isfinite(rsd)]
    if rsd.size == 0:
        return 0.0
    out = float(np.median(rsd))
    return 0.0 if not np.isfinite(out) else out


def thread_map(items, func, core=1):
    items = list(items)
    core = max(int(core), 1)
    if core <= 1 or len(items) <= 1:
        return [func(item) for item in items]
    with ThreadPoolExecutor(max_workers=core) as pool:
        return list(pool.map(func, items))


def drop_last_if_dup(M):
    M = np.asarray(M, dtype=float)
    if M.ndim != 2:
        M = np.asarray(M, dtype=float).reshape(-1, 2)
    if M.shape[0] >= 2 and np.allclose(M[0], M[-1], equal_nan=True):
        return M[:-1]
    return M


def close_ring(M):
    M = np.asarray(M, dtype=float)
    if M.ndim != 2:
        M = np.asarray(M, dtype=float).reshape(-1, 2)
    if M.shape[0] < 3:
        return M
    if not np.allclose(M[0], M[-1], equal_nan=True):
        return np.vstack([M, M[0]])
    return M


def poly_area(M):
    M = close_ring(M)
    x = M[:, 0]
    y = M[:, 1]
    return 0.5 * np.sum(x[1:] * y[:-1] - x[:-1] * y[1:])


def ensure_orientation(M, want_ccw=True):
    M = np.asarray(M, dtype=float)
    ccw_now = poly_area(M) > 0
    if bool(ccw_now) ^ bool(want_ccw):
        return M[::-1].copy()
    return M


def largest_polygon(geom):
    require_shapely()
    if isinstance(geom, Polygon):
        return geom
    geoms = getattr(geom, "geoms", None)
    if geoms is None:
        raise TypeError("Geometry does not contain polygon components.")
    polys = [g for g in geoms if isinstance(g, Polygon)]
    if not polys:
        raise ValueError("No polygon component was found.")
    return max(polys, key=lambda g: g.area)


def detect_holes_by_empty_grid(
    df_xy,
    concave_ratio=0.65,
    nx=140,
    ny=140,
    min_area_frac=0.001,
    max_area_frac=0.15,
    drop_touching=True,
):
    require_pandas()
    require_shapely()
    from shapely.geometry import box as _shapely_box
    if not {"x", "y"}.issubset(df_xy.columns):
        raise ValueError("df_xy must contain x and y columns.")

    xy = df_xy[["x", "y"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(xy).all(axis=1)
    if not np.any(ok):
        raise ValueError("df_xy does not contain any finite coordinates.")
    xy = xy[ok]
    hull_xy = xy
    if hull_xy.shape[0] > 80000:
        rng = np.random.default_rng(1)
        keep = rng.choice(hull_xy.shape[0], 80000, replace=False)
        hull_xy = hull_xy[np.sort(keep)]
    pts = [Point(float(x), float(y)) for x, y in hull_xy]
    mp = MultiPoint(pts)
    if shapely_concave_hull is not None:
        try:
            outer = shapely_concave_hull(mp, ratio=float(concave_ratio))
        except Exception:
            outer = mp.convex_hull
    else:
        outer = mp.convex_hull
    outer = largest_polygon(outer)

    minx, miny, maxx, maxy = outer.bounds
    xs = np.linspace(minx, maxx, int(nx) + 1)
    ys = np.linspace(miny, maxy, int(ny) + 1)
    empty_boxes = []
    occupied = np.zeros((len(xs) - 1, len(ys) - 1), dtype=bool)
    xi = np.searchsorted(xs, xy[:, 0], side="right") - 1
    yi = np.searchsorted(ys, xy[:, 1], side="right") - 1
    xi = np.clip(xi, 0, len(xs) - 2)
    yi = np.clip(yi, 0, len(ys) - 2)
    occupied[xi, yi] = True
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            cell = _shapely_box(xs[i], ys[j], xs[i + 1], ys[j + 1])
            cent = cell.centroid
            in_outer = outer.contains(cent) or outer.touches(cent)
            if not in_outer:
                continue
            if not occupied[i, j]:
                empty_boxes.append(cell)

    outer_bnd = drop_last_if_dup(np.asarray(outer.exterior.coords, dtype=float)[:, :2])
    if not empty_boxes:
        poly_with_holes = Polygon(close_ring(outer_bnd))
        return {
            "outer_bnd": outer_bnd,
            "holes_mat": [],
            "poly_with_holes": poly_with_holes,
        }

    empty_union = unary_union(empty_boxes)
    polys = list(getattr(empty_union, "geoms", [empty_union]))
    boundary = outer.boundary
    outer_area = float(outer.area)

    keep_polys = []
    for poly in polys:
        poly = largest_polygon(poly) if not isinstance(poly, Polygon) else poly
        if drop_touching and poly.touches(boundary):
            continue
        area = float(poly.area)
        if area < min_area_frac * outer_area or area > max_area_frac * outer_area:
            continue
        keep_polys.append(poly)

    holes_mat = [
        drop_last_if_dup(np.asarray(poly.exterior.coords, dtype=float)[:, :2])
        for poly in keep_polys
    ]
    rings = [close_ring(outer_bnd)] + [close_ring(hole) for hole in holes_mat]
    poly_with_holes = Polygon(rings[0], holes=[ring[:-1] for ring in rings[1:]])
    return {
        "outer_bnd": outer_bnd,
        "holes_mat": holes_mat,
        "poly_with_holes": poly_with_holes,
    }


def make_grid_inside(poly, n=300):
    require_pandas()
    require_shapely()
    minx, miny, maxx, maxy = poly.bounds
    gx = np.linspace(minx, maxx, int(n))
    gy = np.linspace(miny, maxy, int(n))
    pts = []
    for x in gx:
        for y in gy:
            pt = Point(float(x), float(y))
            if poly.contains(pt) or poly.touches(pt):
                pts.append((float(x), float(y)))
    return pd.DataFrame(pts, columns=["x", "y"])


def _typical_sample_size(combined, sample_col="sample_id", x_col="x", y_col="y"):
    """Median valid observation count across samples."""

    frame = combined[[sample_col, x_col, y_col]].copy()
    valid = (
        frame[sample_col].notna()
        & np.isfinite(pd.to_numeric(frame[x_col], errors="coerce"))
        & np.isfinite(pd.to_numeric(frame[y_col], errors="coerce"))
    )
    sample_sizes = (
        frame.loc[valid]
        .groupby(sample_col, sort=False)
        .size()
        .to_numpy(dtype=float)
    )
    if sample_sizes.size == 0:
        raise ValueError("No samples contain valid spatial coordinates.")
    return float(np.nanmedian(sample_sizes))


def _resolve_grid_n_for_tissue(
    grid_n_raw,
    n_typ,
    tissue_occupancy,
    *,
    multiplier_bounds=(1.0, 2.0),
    explicit_grid_n=None,
):
    """Apply the notebook's occupancy-aware shared-grid size rule."""

    occupancy = float(tissue_occupancy)
    if not np.isfinite(occupancy) or occupancy <= 0:
        raise ValueError("tissue_occupancy must be a positive finite number.")
    occupancy = min(max(occupancy, 1e-6), 1.0)
    n_typ = float(n_typ)
    if not np.isfinite(n_typ) or n_typ <= 0:
        raise ValueError("n_typ must be a positive finite number.")
    mult_lo, mult_hi = map(float, multiplier_bounds)
    if not (np.isfinite(mult_lo) and np.isfinite(mult_hi) and 0 < mult_lo <= mult_hi):
        raise ValueError("multiplier_bounds must contain positive finite values in increasing order.")

    target_lo = max(1.0, mult_lo * n_typ)
    target_hi = max(target_lo, mult_hi * n_typ)
    grid_n_lo = max(2, int(math.ceil(math.sqrt(target_lo / occupancy))))
    grid_n_hi = max(grid_n_lo, int(math.floor(math.sqrt(target_hi / occupancy))))
    grid_n_raw = int(grid_n_raw)

    if explicit_grid_n is None:
        grid_n = int(min(max(grid_n_raw, grid_n_lo), grid_n_hi))
        selection = "r_driven" if grid_n == grid_n_raw else (
            "n_typ_lower_bound" if grid_n_raw < grid_n_lo else "two_n_typ_upper_bound"
        )
    else:
        grid_n = int(explicit_grid_n)
        if grid_n < 2:
            raise ValueError("grid_n must be an integer of at least 2.")
        selection = "explicit"

    return {
        "grid_n": grid_n,
        "grid_n_raw": grid_n_raw,
        "grid_n_bounds": (grid_n_lo, grid_n_hi),
        "target_grid_locations": (target_lo, target_hi),
        "tissue_occupancy": occupancy,
        "selection": selection,
    }


def _resolve_grid_n_for_tissue_exact(
    poly,
    grid_n_raw,
    n_typ,
    *,
    multiplier_bounds=(1.0, 2.0),
    explicit_grid_n=None,
):
    """Resolve grid resolution from actual retained tissue-grid locations.

    The R-driven resolution is retained when its actual ``make_grid_inside``
    count lies in the data-driven range. Otherwise a monotone integer search
    targets the nearest applicable boundary. An explicit resolution is never
    clamped by the automatic rule.
    """

    minx, miny, maxx, maxy = map(float, poly.bounds)
    width, height = maxx - minx, maxy - miny
    bbox_area = width * height
    if not np.isfinite(bbox_area) or bbox_area <= 0:
        raise ValueError("The tissue polygon must have a positive bounding-box area.")
    occupancy = float(poly.area) / bbox_area
    approximate = _resolve_grid_n_for_tissue(
        grid_n_raw,
        n_typ,
        occupancy,
        multiplier_bounds=multiplier_bounds,
        explicit_grid_n=explicit_grid_n,
    )

    cache = {}

    def retained_count(n):
        n = max(2, int(n))
        if n not in cache:
            cache[n] = int(len(make_grid_inside(poly, n=n)))
        return cache[n]

    grid_n_raw = int(grid_n_raw)
    target_lo, target_hi = approximate["target_grid_locations"]
    raw_count = retained_count(grid_n_raw)

    if explicit_grid_n is not None:
        final_n = int(explicit_grid_n)
        selection = "explicit"
    elif target_lo <= raw_count <= target_hi:
        final_n = grid_n_raw
        selection = "r_driven"
    else:
        target = target_lo if raw_count < target_lo else target_hi
        lo, hi = 2, max(4, int(approximate["grid_n"]), grid_n_raw)
        while retained_count(hi) < target:
            hi *= 2
        while lo < hi:
            mid = (lo + hi) // 2
            if retained_count(mid) < target:
                lo = mid + 1
            else:
                hi = mid
        candidates = sorted({max(2, lo - 1), lo, lo + 1})
        final_n = min(candidates, key=lambda n: (abs(retained_count(n) - target), n))
        selection = "n_typ_lower_bound" if raw_count < target_lo else "two_n_typ_upper_bound"

    result = dict(approximate)
    result.update({
        "grid_n": int(final_n),
        "grid_n_raw": grid_n_raw,
        "m_grid_raw": raw_count,
        "m_grid": retained_count(final_n),
        "selection": selection,
        "tissue_occupancy": float(occupancy),
        "bbox_area": float(bbox_area),
    })
    return result


def nn2_knn_chunked(data_xy, query_xy, k, batch_target=2e7):
    data_xy = np.asarray(data_xy, dtype=float)
    query_xy = np.asarray(query_xy, dtype=float)
    k = int(k)
    NQ = query_xy.shape[0]
    nn_idx = np.zeros((NQ, k), dtype=int)
    nn_dists = np.full((NQ, k), np.inf, dtype=float)
    batch = max(1, int(math.floor(batch_target / max(1, k))))
    tree = cKDTree(data_xy)
    lo = 0
    while lo < NQ:
        hi = min(NQ, lo + batch)
        dists, idx = tree.query(query_xy[lo:hi], k=k)
        if k == 1:
            dists = dists[:, None]
            idx = idx[:, None]
        nn_idx[lo:hi] = idx
        nn_dists[lo:hi] = dists
        lo = hi
    return {"nn_idx": nn_idx, "nn_dists": nn_dists}


def slice_tree_k_loc(tree, k_loc=None):
    if k_loc is None:
        return tree
    k_loc = int(k_loc)
    if k_loc <= 0:
        return tree
    k_use = min(k_loc, tree["nn_idx"].shape[1])
    return {
        "nn_idx": tree["nn_idx"][:, :k_use],
        "nn_dists": tree["nn_dists"][:, :k_use],
    }


def tree_weights_capk(tree, R, k_loc=None, hard_radius=True):
    tr = slice_tree_k_loc(tree, k_loc=k_loc)
    idx = np.asarray(tr["nn_idx"], dtype=int)
    dst = np.asarray(tr["nn_dists"], dtype=float)
    w = gauss_w(dst, R)
    if hard_radius:
        w[dst > float(R)] = 0
    return {"idx": idx, "dst": dst, "w": w}


def local_stats_from_tree(tree, x, R, k_loc=None, hard_radius=True, var_floor=1e-12):
    x = np.asarray(x, dtype=float)
    tw = tree_weights_capk(tree=tree, R=R, k_loc=k_loc, hard_radius=hard_radius)
    idx = tw["idx"]
    w = tw["w"].copy()
    x_mat = x[idx]
    bad = ~np.isfinite(x_mat)
    x_mat[bad] = 0
    w[bad] = 0
    sw = np.sum(w, axis=1)
    sw2 = np.sum(w ** 2, axis=1)
    mu = np.sum(w * x_mat, axis=1) / np.maximum(sw, 1e-12)
    xc = x_mat - mu[:, None]
    xc[~np.isfinite(xc)] = 0
    s2 = np.sum(w * xc ** 2, axis=1) / np.maximum(sw, 1e-12)
    neff = sw ** 2 / np.maximum(sw2, 1e-12)
    mu[sw <= 0] = np.nan
    s2[sw <= 0] = np.nan
    neff[sw <= 0] = 0
    s2 = np.maximum(s2, float(var_floor))
    return {"mu": mu, "s2": s2, "neff": neff}


def map_spot_to_grid_mean_gauss(spot_val, tree, R_map, k_loc=None, hard_radius=True):
    spot_val = np.asarray(spot_val, dtype=float)
    tw = tree_weights_capk(tree=tree, R=R_map, k_loc=k_loc, hard_radius=hard_radius)
    idx = tw["idx"]
    w = tw["w"].copy()
    z = spot_val[idx]
    bad = ~np.isfinite(z)
    z[bad] = 0
    w[bad] = 0
    sw = np.sum(w, axis=1)
    out = np.sum(w * z, axis=1) / np.maximum(sw, 1e-12)
    out[sw <= 0] = np.nan
    return out


def _resolve_risk_map_radius(grid_spacing, configured_radius=None, grid_multiplier=1.5):
    grid_spacing = float(grid_spacing)
    if not np.isfinite(grid_spacing) or grid_spacing <= 0:
        raise ValueError("grid_spacing must be a positive finite number.")
    if configured_radius is not None:
        configured_radius = float(configured_radius)
        if not np.isfinite(configured_radius) or configured_radius <= 0:
            raise ValueError("R_map must be None or a positive finite number.")
        return configured_radius
    grid_multiplier = float(grid_multiplier)
    if not np.isfinite(grid_multiplier) or grid_multiplier <= 0:
        raise ValueError("R_map_grid_multiplier must be a positive finite number.")
    return grid_multiplier * grid_spacing


def normalize_risk_grid(x, center="median", scale="mad", floor_zero=True):
    z = safe_standardize(x, center=center, scale=scale)
    if floor_zero:
        z = np.maximum(z, 0)
    z[~np.isfinite(z)] = 0
    return np.asarray(z, dtype=float)


def choose_gene_cols(df, gene_cols_hint=None):
    require_pandas()
    if gene_cols_hint:
        gene_cols_hint = [g for g in gene_cols_hint if g in df.columns]
        if gene_cols_hint:
            return gene_cols_hint
    meta_cols = {
        "sample_id", "x", "y", "x_aligned", "y_aligned", "celltype",
        "volume", "slide_id", "batch", "age", "rownames",
        "x_scaled", "y_scaled", "var_lddmm", ".file",
        "x_orig", "y_orig", "x_truth", "y_truth", "truth_signal",
        "cell_id",
    }
    meta_cols.update({c for c in df.columns if re.search(r"_x_aligned$|_y_aligned$", c)})
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cand_cols = [c for c in num_cols if c not in meta_cols]

    def is_count_like(v):
        arr = np.asarray(v, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return False
        if np.any(arr < 0):
            return False
        return np.nanquantile(np.abs(arr - np.round(arr)), 0.99) < 1e-6

    return [c for c in cand_cols if is_count_like(df[c].to_numpy())]


def read_batchAB_one_fast(batch_dir, i, sample_prefix="batchU"):
    require_pandas()
    meta_path = Path(batch_dir) / f"sample_{i}_metadata_res06_1106.csv"
    cnt_path = Path(batch_dir) / f"sample_{i}_counts.csv"
    if not meta_path.exists() or not cnt_path.exists():
        raise FileNotFoundError(f"Missing input files for sample_{i}.")

    meta = pd.read_csv(meta_path)
    meta = meta.rename(columns={meta.columns[0]: "cell_id"})
    cnt = pd.read_csv(cnt_path)
    cnt = cnt.rename(columns={cnt.columns[0]: "gene"})

    genes = cnt["gene"].astype(str).tolist()
    spot_cols = list(cnt.columns[1:])
    idx = pd.Index(spot_cols).get_indexer(meta["cell_id"].astype(str))
    if np.any(idx < 0):
        missing = meta.loc[idx < 0, "cell_id"].astype(str).tolist()[:10]
        raise ValueError(
            f"[sample_{i}] meta$cell_id not found in counts columns. Example: {','.join(missing)}"
        )

    mat_gs = cnt.iloc[:, idx + 1].to_numpy(dtype=float)
    mat_sg = mat_gs.T
    dat = pd.concat([meta.reset_index(drop=True), pd.DataFrame(mat_sg, columns=genes)], axis=1)

    if {"center_x", "center_y"}.issubset(dat.columns):
        dat["x"] = pd.to_numeric(dat["center_x"], errors="coerce")
        dat["y"] = pd.to_numeric(dat["center_y"], errors="coerce")
    if not {"x_aligned", "y_aligned"}.issubset(dat.columns):
        if {"V1", "V2"}.issubset(dat.columns):
            dat["x_aligned"] = pd.to_numeric(dat["V1"], errors="coerce")
            dat["y_aligned"] = pd.to_numeric(dat["V2"], errors="coerce")
        elif {"x", "y"}.issubset(dat.columns):
            dat["x_aligned"] = pd.to_numeric(dat["x"], errors="coerce")
            dat["y_aligned"] = pd.to_numeric(dat["y"], errors="coerce")
        else:
            dat["x_aligned"] = np.nan
            dat["y_aligned"] = np.nan

    if "age" not in dat.columns:
        raise ValueError(f"[sample_{i}] metadata is missing age")
    if "batch" not in dat.columns:
        dat["batch"] = "U"
    if "slide_id" not in dat.columns:
        dat["slide_id"] = f"sample{i}"
    if "celltype" not in dat.columns:
        dat["celltype"] = np.nan

    dat["age"] = pd.to_numeric(dat["age"], errors="coerce")
    dat["batch"] = dat["batch"].astype(str)
    dat["slide_id"] = dat["slide_id"].astype(str)
    dat["celltype"] = dat["celltype"].astype(str)
    dat["sample_id"] = (
        f"{sample_prefix}_sample{i}_"
        + dat["slide_id"].astype(str)
        + "_age"
        + dat["age"].astype(str)
    )
    return dat


def infer_prefix_from_meta(all_dir, i):
    require_pandas()
    meta_path = Path(all_dir) / f"sample_{i}_metadata_res06_1106.csv"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    m0 = pd.read_csv(meta_path)
    if "batch" not in m0.columns:
        return "batchU"
    b0 = str(m0["batch"].iloc[0]) if len(m0) else ""
    if not b0 or b0.lower() == "nan":
        return "batchU"
    if re.search(r"^A|batchA", b0, flags=re.IGNORECASE):
        return "batchA"
    if re.search(r"^B|batchB", b0, flags=re.IGNORECASE):
        return "batchB"
    return "batchU"


def read_all_combined_via_batchAB(all_dir, sample_indices=range(1, 21)):
    require_pandas()
    out = []
    for i in list(sample_indices):
        pref = infer_prefix_from_meta(all_dir, i)
        print(f"[READ] sample_{i} | prefix={pref}")
        di = read_batchAB_one_fast(all_dir, i, sample_prefix=pref)
        if di["sample_id"].nunique() != 1:
            raise ValueError(f"[sample_{i}] sample_id is not constant inside sample.")
        out.append(di)
    combined = pd.concat(out, axis=0, ignore_index=True)
    return {"combined": combined}


def read_all_combined(
    data,
    mode="auto",
    sample_col="sample_id",
    x_col="x",
    y_col="y",
    celltype_col="celltype",
    batch_col="batch",
    age_col="age",
    x_aligned_col=None,
    y_aligned_col=None,
    var_lddmm_col=None,
    gene_cols_hint=None,
    sample_indices=range(1, 21),
    infer_batch_from_sample_id=True,
    default_batch="batch1",
    fill_missing_celltype=False,
    default_celltype="CT1",
):
    require_pandas()

    def infer_batch(sample_id_vec):
        out = []
        for sid in list(sample_id_vec):
            s = str(sid)
            if re.search(r"^A|batchA", s, flags=re.IGNORECASE):
                out.append("batchA")
            elif re.search(r"^B|batchB", s, flags=re.IGNORECASE):
                out.append("batchB")
            else:
                out.append(default_batch)
        return out

    def format_one_df(df, sample_id_value=None):
        out = pd.DataFrame(df).copy()
        if sample_id_value is not None and "sample_id" not in out.columns:
            out["sample_id"] = sample_id_value
        if sample_col not in out.columns and "sample_id" not in out.columns:
            raise ValueError("Missing sample_id column.")
        if "sample_id" not in out.columns:
            out["sample_id"] = out[sample_col]
        if x_col not in out.columns or y_col not in out.columns:
            raise ValueError("Missing x/y columns in input data.")
        out["x"] = pd.to_numeric(out[x_col], errors="coerce")
        out["y"] = pd.to_numeric(out[y_col], errors="coerce")
        if (
            x_aligned_col is not None
            and y_aligned_col is not None
            and x_aligned_col in out.columns
            and y_aligned_col in out.columns
        ):
            out["x_aligned"] = pd.to_numeric(out[x_aligned_col], errors="coerce")
            out["y_aligned"] = pd.to_numeric(out[y_aligned_col], errors="coerce")
        elif {"x_aligned", "y_aligned"}.issubset(out.columns):
            out["x_aligned"] = pd.to_numeric(out["x_aligned"], errors="coerce")
            out["y_aligned"] = pd.to_numeric(out["y_aligned"], errors="coerce")
        else:
            out["x_aligned"] = out["x"]
            out["y_aligned"] = out["y"]
        if celltype_col in out.columns:
            out["celltype"] = out[celltype_col].astype(str)
        elif "celltype" in out.columns:
            out["celltype"] = out["celltype"].astype(str)
        elif fill_missing_celltype:
            out["celltype"] = default_celltype
        else:
            raise ValueError("Missing celltype column.")
        if batch_col in out.columns:
            out["batch"] = out[batch_col].astype(str)
        elif "batch" in out.columns:
            out["batch"] = out["batch"].astype(str)
        elif infer_batch_from_sample_id:
            out["batch"] = infer_batch(out["sample_id"])
        else:
            out["batch"] = default_batch
        if age_col in out.columns:
            out["age"] = pd.to_numeric(out[age_col], errors="coerce")
        elif "age" in out.columns:
            out["age"] = pd.to_numeric(out["age"], errors="coerce")
        else:
            out["age"] = parse_age_numeric(out["sample_id"])
        if var_lddmm_col is not None and var_lddmm_col in out.columns:
            out["var_lddmm"] = pd.to_numeric(out[var_lddmm_col], errors="coerce")
        elif "var_lddmm" in out.columns:
            out["var_lddmm"] = pd.to_numeric(out["var_lddmm"], errors="coerce")
        out["sample_id"] = out["sample_id"].astype(str)
        return out

    if mode == "auto":
        if isinstance(data, (str, os.PathLike)) and Path(data).is_dir():
            mode = "batchAB_dir"
        elif pd is not None and isinstance(data, pd.DataFrame):
            mode = "combined_df"
        elif isinstance(data, list):
            mode = "list_of_dfs"
        else:
            raise ValueError("Cannot infer mode from data.")

    if mode == "batchAB_dir":
        out = read_all_combined_via_batchAB(data, sample_indices=sample_indices)
        combined = out["combined"].copy()
    elif mode == "combined_df":
        combined = format_one_df(data)
    elif mode == "list_of_dfs":
        names = [f"sample_{i + 1}" for i in range(len(data))]
        parts = [format_one_df(df, sample_id_value=names[i]) for i, df in enumerate(data)]
        combined = pd.concat(parts, axis=0, ignore_index=True)
    else:
        raise ValueError("Unsupported mode.")

    req = ["sample_id", "x", "y", "x_aligned", "y_aligned", "celltype", "batch", "age"]
    miss = [c for c in req if c not in combined.columns]
    if miss:
        raise ValueError("combined is missing required columns: " + ", ".join(miss))
    gene_cols = choose_gene_cols(combined, gene_cols_hint=gene_cols_hint)
    meta = {
        "n_rows": int(combined.shape[0]),
        "n_samples": int(combined["sample_id"].nunique()),
        "sample_ids": combined["sample_id"].astype(str).drop_duplicates().tolist(),
    }
    return {"combined": combined, "gene_cols": gene_cols, "meta": meta}


def extract_AB_for_tid(combined, ref_sample_id, target_sample_id, gene_cols=None):
    require_pandas()
    A = combined.loc[combined["sample_id"].astype(str) == str(ref_sample_id)].copy()
    B = combined.loc[combined["sample_id"].astype(str) == str(target_sample_id)].copy()
    if A.empty or B.empty:
        raise ValueError(
            f".extract_AB_for_tid: A or B is empty (ref={ref_sample_id}, target={target_sample_id})"
        )
    id_A = [f"A_{i:06d}" for i in range(1, len(A) + 1)]
    id_B = [f"B_{i:06d}" for i in range(1, len(B) + 1)]
    ref_meta = A.copy()
    smp_align = B.copy()
    ref_meta.index = id_A
    smp_align.index = id_B
    gene_cols = choose_gene_cols(combined, gene_cols_hint=gene_cols)
    if not gene_cols:
        raise ValueError(".extract_AB_for_tid: no gene columns were identified.")
    ref_expr = pd.DataFrame(A[gene_cols].to_numpy(dtype=float).T, index=gene_cols, columns=id_A)
    smp_expr = pd.DataFrame(B[gene_cols].to_numpy(dtype=float).T, index=gene_cols, columns=id_B)
    return {
        "ref_meta": ref_meta,
        "smp_align": smp_align,
        "ref_expr": ref_expr,
        "smp_expr": smp_expr,
    }


def plot_gene_points(df, gene, vmax, point_size=0.5, title=None):
    require_pandas()
    fig, ax = plt.subplots(figsize=(5, 5))
    sc = ax.scatter(
        pd.to_numeric(df["x"], errors="coerce"),
        pd.to_numeric(df["y"], errors="coerce"),
        c=pd.to_numeric(df[gene], errors="coerce"),
        s=point_size * 10,
        cmap="Blues",
        vmin=0,
        vmax=float(vmax),
        linewidths=0,
        alpha=0.8,
    )
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title or gene)
    return fig, ax, sc


def grid_consensus_erosion(sig_vec, tr_grid, R_cons, tau=0.60, n_iter=1):
    sig = (np.asarray(sig_vec, dtype=float) > 0).astype(int)
    idx = np.asarray(tr_grid["nn_idx"], dtype=int)
    dst = np.asarray(tr_grid["nn_dists"], dtype=float)
    tau = min(max(float(tau), 0.0), 1.0)
    n_iter = max(int(n_iter), 1)
    frac_last = np.zeros(sig.shape[0], dtype=float)
    for _ in range(n_iter):
        w = np.exp(-0.5 * (dst / max(float(R_cons), 1e-12)) ** 2)
        w[(~np.isfinite(dst)) | (dst <= 0)] = 0
        w[dst > float(R_cons)] = 0
        sig_nb = sig[idx]
        den = np.sum(w, axis=1) + 1e-12
        num = np.sum(w * sig_nb, axis=1)
        frac = num / den
        frac[~np.isfinite(frac)] = 0
        frac_last = frac
        sig = ((sig > 0) & (frac >= tau)).astype(int)
    return {"sig": sig.astype(bool), "frac": frac_last}


def get_t_df_use_p_from_fit(fit, time_id):
    td = fit.get("terrain_data")
    if td is None:
        raise ValueError(".get_t_df_use_p_from_fit: fit['terrain_data'] is None.")
    needed = ["stat_by_time", "p_by_time", "use_by_time", "df_by_time"]
    if any(key not in td for key in needed):
        raise ValueError(
            ".get_t_df_use_p_from_fit: fit['terrain_data'] is missing stat_by_time/p_by_time/use_by_time/df_by_time."
        )
    if time_id not in td["stat_by_time"]:
        raise ValueError(
            f".get_t_df_use_p_from_fit: time_id={time_id} is not present in fit['terrain_data']['stat_by_time']."
        )
    tvec = np.asarray(td["stat_by_time"][time_id], dtype=float)
    pvec = np.asarray(td["p_by_time"][time_id], dtype=float)
    use = np.asarray(td["use_by_time"][time_id], dtype=bool)
    dfv = float(td["df_by_time"][time_id])
    if use.shape[0] != tvec.shape[0]:
        raise ValueError(".get_t_df_use_p_from_fit: use and stat have inconsistent lengths.")
    if pvec.shape[0] != tvec.shape[0]:
        raise ValueError(".get_t_df_use_p_from_fit: p and stat have inconsistent lengths.")
    if not np.isfinite(dfv) or dfv <= 0:
        raise ValueError(".get_t_df_use_p_from_fit: invalid df.")
    use[np.isnan(use)] = False
    return {"t": tvec, "use": use, "p": pvec, "df": dfv}


def compute_r01_and_global_score(
    risk_vec,
    use_mask=None,
    weight_norm="qcap",
    cap_q=0.95,
    global_rule="frac_above",
    global_thr=0.30,
    global_cap=0.50,
):
    r_full = np.asarray(risk_vec, dtype=float)
    r_full[~np.isfinite(r_full)] = 0
    r_full = np.maximum(r_full, 0)
    if use_mask is None:
        use_mask = np.ones(r_full.shape[0], dtype=bool)
    else:
        use_mask = np.asarray(use_mask, dtype=bool)
        if use_mask.shape[0] != r_full.shape[0]:
            use_mask = np.resize(use_mask, r_full.shape[0])
    r_use = r_full[use_mask & np.isfinite(r_full)]
    if r_use.size == 0:
        r_use = r_full[np.isfinite(r_full)]
    if r_use.size == 0:
        r_use = np.array([0.0])

    if weight_norm == "qcap":
        r_cap = float(np.nanquantile(r_use, cap_q))
        if not np.isfinite(r_cap) or r_cap <= 0:
            r_cap = float(np.nanmax(r_use)) if r_use.size else 1e-8
        if not np.isfinite(r_cap) or r_cap <= 0:
            r_cap = 1e-8
        r_cap = max(r_cap, 1e-8)
        r01 = np.minimum(r_full, r_cap) / r_cap
    elif weight_norm == "rank":
        r01 = ecdf_eval(r_use, r_full)
        r_cap = np.nan
    elif weight_norm == "none":
        mx = float(np.nanmax(r_use)) if r_use.size else 0.0
        if not np.isfinite(mx) or mx <= 0:
            r01 = np.zeros_like(r_full)
        else:
            r01 = r_full / mx
        r_cap = mx
    else:
        raise ValueError(f"Unknown weight_norm: {weight_norm}")

    r01 = np.clip(r01, 0, 1)
    if global_rule == "frac_above":
        global_score = float(np.nanmean(r_use > max(float(global_thr), 0.0)))
    elif global_rule == "mean_cap":
        cap = max(float(global_cap), 1e-8)
        global_score = float(np.nanmean(np.minimum(r_use, cap)) / cap)
    else:
        raise ValueError(f"Unknown global_rule: {global_rule}")
    if not np.isfinite(global_score):
        global_score = 0.0
    global_score = float(np.clip(global_score, 0, 1))
    return {"r01": r01, "global_score": global_score, "r_cap": r_cap}


def calibrate_lambdas_empnull_scale(
    tvec,
    use_mask,
    r01,
    global_score,
    df,
    bins=10,
    min_bin_n=200,
    trim_abs_q=0.95,
    p_grid=2.0,
    tau_anchor_q=0.80,
    slack=1,
    g_floor=0.25,
    lam_global_cap=5e4,
    lam_local_cap=5e4,
    eps=1e-12,
    verbose=True,
):
    """Compatibility wrapper for the private MAD-null1-A0 calibration.

    The trim_abs_q, g_floor and lam_global_cap arguments are retained in this
    private signature so notebook-derived callers and serialized settings can
    still be read. They do not affect the promoted local-only calibration.
    """

    return calibrate_mismatch_variance(
        tvec=tvec,
        use_mask=use_mask,
        r01=r01,
        global_score=global_score,
        df=df,
        bins=bins,
        min_bin_n=min_bin_n,
        p_grid=p_grid,
        tau_anchor_q=tau_anchor_q,
        slack=slack,
        lam_local_cap=lam_local_cap,
        eps=eps,
        verbose=verbose,
        trim_abs_q=trim_abs_q,
        g_floor=g_floor,
        lam_global_cap=lam_global_cap,
    )

def batch_prepare_once_multi(
    combined,
    ref_sample_id=None,
    control_ids=None,
    user_var_col=None,
    s=0.25,
    core=1,
    seed=None,
    use_libsize_norm=True,
    target_total=250,
    gene_cols_hint=None,
    grid_n=None,
):
    require_pandas()
    core = max(int(core), 1)
    if seed is not None:
        np.random.seed(int(seed))
    if grid_n is not None and int(grid_n) < 2:
        raise ValueError("grid_n must be an integer of at least 2.")

    def build_fixed_PARAMS(seed=None):
        return {
            "alpha": 0.05,
            "stat_s2_floor_mult": 0.10,
            "max_k_symknn": 300,
            "k_screen": 40,
            "hard_radius_loc": True,
            "alpha0_dir": 1.0,
            "USE_LIBSIZE_NORM": bool(use_libsize_norm),
            "target_total": float(target_total),
            "lambda_beta": 10.0,
            "celltype_adjustment": {
                "method": "normalized_jensen_shannon_distance",
                "variance_link": "exp",
                "log_base": "natural",
                "normalizing_constant": "log(2)",
            },
            "auto_geometry": {
                "cover_target": 18,
                "cover_quantile_R": 0.65,
                "grids_per_R": 4.5,
                "neff_quantile": 0.10,
                "rrel_bounds": (0.010, 0.040),
                "grid_location_multiplier_bounds": (1.0, 2.0),
                "neff_bounds": (2, 25),
                "k_density": 5,
                "max_k_symknn": 300,
                "max_points_per_sample": 15000,
                "seed": seed,
            },
            "grid_n_explicit": None if grid_n is None else int(grid_n),
            "uncert": {
                "R_map": None,
                "R_map_grid_multiplier": 1.5,
                "use_user_var_if_available": True,
                "user_var_col": "var_lddmm",
                "norm_center": "median",
                "norm_scale": "mad",
                "floor_zero": True,
                "swnd_cos": {
                    "coord_ref": ("x", "y"),
                    "coord_B_align": ("x_aligned", "y_aligned"),
                    "k_big": 25,
                    "k_smooth": 30,
                    "h_scale": 1.5,
                    "marker_var_q": 0.5,
                    "marker_delta_q": 0.1,
                    "marker_cor_min": 0.2,
                    "max_markers": 300,
                    "use_density_channel": True,
                    "density_weight": "auto",
                    "k_dens": 50,
                    "h_dens_scale": 1.0,
                    "verbose": True,
                },
            },
            "risk": {
                "weight_norm": "qcap",
                "cap_q": 0.95,
                "global_rule": "frac_above",
                "global_thr": 0.6,
                "global_cap": 0.50,
                "wv_floor_frac": 0,
                "wv_min": 1e-30,
                "auto": {
                    "anchor_time_id": None,
                    "bins": 10,
                    "min_bin_n": 200,
                    "tau_anchor_q": 0.80,
                    "slack": 1,
                    "lam_local_cap": 5e4,
                },
            },
        }

    PARAMS = build_fixed_PARAMS(seed=seed)
    kB = 2
    s = float(s)
    if not np.isfinite(s) or s <= 0 or s >= 1:
        raise ValueError("s must be a number in (0, 1).")

    def parallel_lapply(X, FUN, core=1):
        return thread_map(list(X), FUN, core=core)

    def tree_weights_nocap(tree, R, hard_radius):
        return tree_weights_capk(tree, R, None, hard_radius)

    def map_spot_to_grid_mean_gauss_nocap(spot_val, tree, R_map, hard_radius):
        return map_spot_to_grid_mean_gauss(spot_val, tree, R_map, None, hard_radius)

    def auto_geometry_params_v2_local(
        combined,
        sample_col="sample_id",
        x_col="x",
        y_col="y",
        cover_target=18,
        cover_quantile_R=0.65,
        grids_per_R=4.5,
        neff_quantile=0.20,
        rrel_bounds=(0.010, 0.020),
        grid_location_multiplier_bounds=(1.0, 2.0),
        neff_bounds=(2, 25),
        k_density=5,
        max_points_per_sample=15000,
        max_k_symknn=300,
        seed=None,
        core=core,
    ):
        df = combined[[sample_col, x_col, y_col]].copy()
        df.columns = ["sample_id", "x", "y"]
        df = df.loc[
            np.isfinite(pd.to_numeric(df["x"], errors="coerce"))
            & np.isfinite(pd.to_numeric(df["y"], errors="coerce"))
            & df["sample_id"].notna()
        ].copy()
        sp_list = [dat.copy() for _, dat in df.groupby("sample_id", sort=False)]
        rng = np.random.default_rng(seed)

        def pass1_fun(dat):
            XY = dat[["x", "y"]].to_numpy(dtype=float)
            if XY.shape[0] > max_points_per_sample:
                keep = rng.choice(XY.shape[0], max_points_per_sample, replace=False)
                XY = XY[keep]
            if XY.shape[0] < max(30, int(k_density) + 5):
                return None
            Ls = max(np.ptp(XY[:, 0]), np.ptp(XY[:, 1]))
            if not np.isfinite(Ls) or Ls <= 0:
                return None
            kden = min(int(k_density), XY.shape[0] - 1)
            tree = cKDTree(XY)
            dists, _ = tree.query(XY, k=kden + 1)
            dk = dists[:, kden]
            dk = dk[np.isfinite(dk) & (dk > 0)]
            if dk.size == 0:
                return None
            lambda_hat = np.median(kden / (math.pi * dk ** 2))
            if not np.isfinite(lambda_hat) or lambda_hat <= 0:
                return None
            R_t = math.sqrt(float(cover_target) / (math.pi * lambda_hat))
            Rrel_t = R_t / Ls
            if not np.isfinite(Rrel_t) or Rrel_t <= 0:
                return None
            return {"Ls": Ls, "Rrel": Rrel_t}

        pass1 = [x for x in parallel_lapply(sp_list, pass1_fun, core=core) if x is not None]
        if not pass1:
            raise ValueError("auto_geometry_params_v2: not enough valid geometry information.")
        L_all = np.asarray([item["Ls"] for item in pass1], dtype=float)
        Rrel_all = np.asarray([item["Rrel"] for item in pass1], dtype=float)
        R_rel = float(np.nanquantile(Rrel_all, cover_quantile_R))
        R_rel = min(max(R_rel, float(rrel_bounds[0])), float(rrel_bounds[1]))
        L_med = float(np.nanmedian(L_all))
        R_final = R_rel * L_med
        n_typ = _typical_sample_size(df, "sample_id", "x", "y")
        grid_n_raw = int(round(grids_per_R / R_rel))

        def pass2_fun(dat):
            XY = dat[["x", "y"]].to_numpy(dtype=float)
            if XY.shape[0] > max_points_per_sample:
                keep = rng.choice(XY.shape[0], max_points_per_sample, replace=False)
                XY = XY[keep]
            if XY.shape[0] < 30:
                return None
            kprobe = min(int(max_k_symknn), XY.shape[0])
            tree = cKDTree(XY)
            dst, idx = tree.query(XY, k=kprobe)
            if kprobe == 1:
                dst = dst[:, None]
                idx = idx[:, None]
            w = np.exp(-0.5 * (dst / max(R_final, 1e-12)) ** 2)
            w[dst > R_final] = 0
            countR = np.sum(dst <= R_final, axis=1)
            sw = np.sum(w, axis=1)
            sw2 = np.sum(w ** 2, axis=1)
            neff = sw ** 2 / np.maximum(sw2, 1e-12)
            return {
                "countR": countR[np.isfinite(countR) & (countR > 0)],
                "neff": neff[np.isfinite(neff) & (neff > 0)],
            }

        pass2 = [x for x in parallel_lapply(sp_list, pass2_fun, core=core) if x is not None]
        countR_all = np.concatenate([item["countR"] for item in pass2]) if pass2 else np.array([])
        neff_all = np.concatenate([item["neff"] for item in pass2]) if pass2 else np.array([])
        if countR_all.size == 0 or neff_all.size == 0:
            raise ValueError("auto_geometry_params_v2: failed to estimate empirical count/neff.")
        q99_countR = float(np.nanquantile(countR_all, 0.99))
        if not np.isfinite(q99_countR):
            q99_countR = float(np.nanmax(countR_all))
        max_k_symknn_auto = max(60, int(math.ceil(q99_countR + 5)))
        neff_min = int(math.floor(np.nanquantile(neff_all, neff_quantile)))
        neff_min = int(min(max(neff_min, int(neff_bounds[0])), int(neff_bounds[1])))
        return {
            "params": {
                "R_rel": round(R_rel, 4),
                "grid_n": grid_n_raw,
                "neff_min": neff_min,
                "max_k_symknn_auto": max_k_symknn_auto,
            },
            "diagnostics": {
                "L_med": L_med,
                "R_final": R_final,
                "Rrel_all": Rrel_all,
                "n_typ": n_typ,
                "grid_n_raw": grid_n_raw,
                "grid_location_multiplier_bounds": tuple(grid_location_multiplier_bounds),
                "q99_countR": q99_countR,
                "countR_summary": np.nanquantile(countR_all, [0.1, 0.5, 0.9, 0.99]),
                "neff_summary": np.nanquantile(neff_all, [0.1, 0.5, 0.9]),
            },
        }

    def prepare_once_multi_local(combined, ref_sample_id=None, PARAMS=PARAMS, gene_cols_hint=None, core=core):
        needed = {"sample_id", "x", "y", "celltype"}
        if not needed.issubset(combined.columns):
            raise ValueError("combined must contain sample_id, x, y, and celltype.")
        combined = combined.copy()

        if ref_sample_id is None:
            if "var_lddmm" in combined.columns:
                tab = (
                    combined.assign(var_lddmm_num=pd.to_numeric(combined["var_lddmm"], errors="coerce"))
                    .groupby("sample_id", sort=False)["var_lddmm_num"]
                    .apply(lambda z: np.isfinite(z).any())
                )
                cand = tab.index[~tab.to_numpy()]
                if len(cand) < 1:
                    raise ValueError("Cannot infer the reference sample automatically; please provide ref_sample_id explicitly.")
                ref_sample_id = str(cand[0])
            else:
                raise ValueError("ref_sample_id was not provided and var_lddmm is unavailable, so the reference sample cannot be inferred automatically.")

        ref_sample_id = str(ref_sample_id)
        lev_all = combined["sample_id"].astype(str).drop_duplicates().tolist()
        time_ids = [sid for sid in lev_all if sid != ref_sample_id]
        has_xa = {"x_aligned", "y_aligned"}.issubset(combined.columns)
        if has_xa:
            x_aligned = pd.to_numeric(combined["x_aligned"], errors="coerce").to_numpy(dtype=float)
            y_aligned = pd.to_numeric(combined["y_aligned"], errors="coerce").to_numpy(dtype=float)
            x_raw = pd.to_numeric(combined["x"], errors="coerce").to_numpy(dtype=float)
            y_raw = pd.to_numeric(combined["y"], errors="coerce").to_numpy(dtype=float)
            combined["x"] = np.where(np.isfinite(x_aligned), x_aligned, x_raw)
            combined["y"] = np.where(np.isfinite(y_aligned), y_aligned, y_raw)
        combined["sample_id"] = combined["sample_id"].astype(str)
        combined["celltype"] = pd.Categorical(combined["celltype"].astype(str))

        ag_cfg = or_else(PARAMS.get("auto_geometry"), {})
        # The geometry estimator performs seeded, per-sample subsampling in two
        # passes. Run only those passes serially so thread scheduling cannot
        # change which sample consumes each RNG draw. Later preparation stages
        # still use the caller-requested worker count.
        geometry_core = 1
        ag = auto_geometry_params_v2_local(
            combined=combined,
            sample_col="sample_id",
            x_col="x",
            y_col="y",
            cover_target=or_else(ag_cfg.get("cover_target"), 18),
            cover_quantile_R=or_else(ag_cfg.get("cover_quantile_R"), 0.65),
            grids_per_R=or_else(ag_cfg.get("grids_per_R"), 4.5),
            neff_quantile=or_else(ag_cfg.get("neff_quantile"), 0.20),
            rrel_bounds=or_else(ag_cfg.get("rrel_bounds"), (0.010, 0.020)),
            grid_location_multiplier_bounds=or_else(
                ag_cfg.get("grid_location_multiplier_bounds"), (1.0, 2.0)
            ),
            neff_bounds=or_else(ag_cfg.get("neff_bounds"), (2, 25)),
            k_density=or_else(ag_cfg.get("k_density"), 5),
            max_points_per_sample=or_else(ag_cfg.get("max_points_per_sample"), 15000),
            max_k_symknn=or_else(PARAMS.get("max_k_symknn"), 300),
            seed=seed if seed is not None else ag_cfg.get("seed"),
            core=geometry_core,
        )

        PARAMS["R_rel"] = ag["params"]["R_rel"]
        PARAMS["grid_n"] = ag["params"]["grid_n"]
        PARAMS["neff_min"] = ag["params"]["neff_min"]
        PARAMS["max_k_symknn"] = ag["params"]["max_k_symknn_auto"]
        PARAMS["auto_geometry_resolved"] = ag
        PARAMS["hard_radius_loc"] = True
        hard_radius_loc = True

        gene_cols = choose_gene_cols(combined, gene_cols_hint=gene_cols_hint)
        if not gene_cols:
            meta_cols_expr = {
                "sample_id", "x", "y", "x_aligned", "y_aligned", "celltype",
                "volume", "slide_id", "batch", "age", "rownames",
                "x_scaled", "y_scaled", "var_lddmm", ".file",
                "x_orig", "y_orig", "x_truth", "y_truth", "truth_signal",
                "cell_id",
            }
            meta_cols_expr.update({c for c in combined.columns if re.search(r"_x_aligned$|_y_aligned$", str(c))})
            num_cols_expr = [c for c in combined.columns if c not in meta_cols_expr and pd.api.types.is_numeric_dtype(combined[c])]
            gene_cols = []
            for c in num_cols_expr:
                vals = pd.to_numeric(combined[c], errors="coerce").to_numpy(dtype=float)
                ok = np.isfinite(vals)
                if np.any(ok) and np.nanmin(vals[ok]) >= 0 and float(np.nanmean(vals[ok] > 0)) >= 0.001:
                    gene_cols.append(c)
            if not gene_cols:
                raise ValueError("batch_prepare_once_multi: no expression columns were identified. Pass raw count-like gene columns or ensure gene columns are numeric nonnegative non-metadata columns.")
        if PARAMS.get("USE_LIBSIZE_NORM", True) and gene_cols:
            keep_gene = []
            for g in gene_cols:
                vals = pd.to_numeric(combined[g], errors="coerce").to_numpy(dtype=float, copy=True)
                keep_gene.append(float(np.nanmean(vals > 0)) >= 0.01)
            gene_cols_use = [g for g, keep in zip(gene_cols, keep_gene) if keep]
            if gene_cols_use:
                M = combined[gene_cols_use].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=True)
                M[M < 0] = 0
                libsize = np.nansum(M, axis=1)
                scf = np.zeros_like(libsize, dtype=float)
                ok_lib = np.isfinite(libsize) & (libsize > 0)
                scf[ok_lib] = float(PARAMS["target_total"]) / libsize[ok_lib]
                Msc = M * scf[:, None]
                combined = combined.astype({g: float for g in gene_cols_use}, copy=False)
                combined.loc[:, gene_cols_use] = pd.DataFrame(Msc, index=combined.index, columns=gene_cols_use)

        print(f"[INFO] detecting tissue geometry on {combined.shape[0]} spots ...", flush=True)
        auto = detect_holes_by_empty_grid(
            combined.loc[:, ["x", "y"]].copy(),
            concave_ratio=0.65,
            nx=140,
            ny=140,
            min_area_frac=0.001,
            max_area_frac=0.15,
            drop_touching=True,
        )
        outer_bnd = ensure_orientation(auto["outer_bnd"], True)
        holes_mat = [ensure_orientation(h, want_ccw=False) for h in auto["holes_mat"]] if auto["holes_mat"] else []
        poly_with_holes = Polygon(close_ring(outer_bnd), holes=[close_ring(h)[:-1] for h in holes_mat])
        minx, miny, maxx, maxy = poly_with_holes.bounds
        L = max(maxx - minx, maxy - miny)
        grid_resolution = _resolve_grid_n_for_tissue_exact(
            poly_with_holes,
            ag["diagnostics"]["grid_n_raw"],
            ag["diagnostics"]["n_typ"],
            multiplier_bounds=ag["diagnostics"]["grid_location_multiplier_bounds"],
            explicit_grid_n=PARAMS.get("grid_n_explicit"),
        )
        PARAMS["grid_n"] = grid_resolution["grid_n"]
        ag["params"]["grid_n"] = grid_resolution["grid_n"]
        ag["diagnostics"].update(grid_resolution)
        grid_eval = make_grid_inside(poly_with_holes, n=PARAMS["grid_n"])
        Q_grid = grid_eval[["x", "y"]].to_numpy(dtype=float)
        ag["diagnostics"]["m_grid"] = int(Q_grid.shape[0])
        R = float(PARAMS["R_rel"]) * float(L)
        grid_spacing = float(L) / max(int(PARAMS["grid_n"]) - 1, 1)
        risk_cfg = PARAMS.setdefault("uncert", {})
        R_map = _resolve_risk_map_radius(
            grid_spacing,
            configured_radius=risk_cfg.get("R_map"),
            grid_multiplier=or_else(risk_cfg.get("R_map_grid_multiplier"), 1.5),
        )
        risk_cfg["grid_spacing_resolved"] = grid_spacing
        risk_cfg["R_map_resolved"] = R_map
        print(
            f"[INFO] geometry ready: {len(holes_mat)} holes, "
            f"{Q_grid.shape[0]} grid points, R={R:.4g}, "
            f"h_grid={grid_spacing:.4g}, R_map={R_map:.4g}",
            flush=True,
        )
        k_screen = int(or_else(PARAMS.get("k_screen"), 40))
        k_screen = max(6, min(k_screen, Q_grid.shape[0] - 1))
        kn_g = nn2_knn_chunked(Q_grid, Q_grid, k=k_screen + 1)
        tr_grid_posthoc = {
            "nn_idx": kn_g["nn_idx"][:, 1:],
            "nn_dists": kn_g["nn_dists"][:, 1:],
        }

        sample_rows = {sid: np.flatnonzero(combined["sample_id"].astype(str).to_numpy() == sid) for sid in combined["sample_id"].astype(str).drop_duplicates()}
        REF = combined.iloc[sample_rows[ref_sample_id]][["x", "y", "celltype"]].copy()
        k_sym = max(30, int(or_else(PARAMS.get("max_k_symknn"), 300)))
        tr_ref = nn2_knn_chunked(REF[["x", "y"]].to_numpy(dtype=float), Q_grid, k=min(k_sym, REF.shape[0]))

        def build_tree_for_time(tt):
            XYt = combined.iloc[sample_rows[str(tt)]][["x", "y", "celltype"]].copy()
            return nn2_knn_chunked(XYt[["x", "y"]].to_numpy(dtype=float), Q_grid, k=min(k_sym, XYt.shape[0]))

        print(f"[INFO] building KNN trees for {len(time_ids) + 1} samples ...", flush=True)
        trees_time_res = parallel_lapply(time_ids, build_tree_for_time, core=core)
        trees_time = {str(tt): tree for tt, tree in zip(time_ids, trees_time_res)}

        def estimate_type_proportions(tr_tree, R, labels_factor, levels_all, alpha0):
            labels = pd.Categorical(pd.Series(labels_factor, dtype="string"), categories=list(levels_all))
            label_codes = labels.codes
            Cn = len(levels_all)
            tw = tree_weights_nocap(tr_tree, R, hard_radius_loc)
            idx = tw["idx"]
            w = tw["w"]
            den = np.sum(w, axis=1)
            label_mat = label_codes[idx]
            p_mat = np.full((idx.shape[0], Cn), np.nan, dtype=float)
            neff_mat = np.zeros((idx.shape[0], Cn), dtype=float)
            denom_full = den + float(alpha0) * Cn
            denom_full[(~np.isfinite(denom_full)) | (denom_full <= 0)] = float(alpha0) * Cn + 1e-12
            for c in range(Cn):
                wc = w * (label_mat == c)
                num_c = np.sum(wc, axis=1)
                p_mat[:, c] = (num_c + float(alpha0)) / denom_full
                sw2 = np.sum(wc * wc, axis=1)
                neff_mat[:, c] = np.divide(num_c * num_c, np.maximum(sw2, 1e-12), out=np.zeros_like(num_c), where=sw2 > 0)
            p_mat[~np.isfinite(p_mat)] = 1.0 / Cn
            neff_mat[~np.isfinite(neff_mat)] = 0.0
            cols = [str(level) for level in levels_all]
            return {"p": pd.DataFrame(p_mat, columns=cols), "neff_type": pd.DataFrame(neff_mat, columns=cols)}

        all_levels = list(combined["celltype"].cat.categories)
        ref_type_stats = estimate_type_proportions(
            tr_tree=tr_ref,
            R=R,
            labels_factor=REF["celltype"].astype(str).tolist(),
            levels_all=all_levels,
            alpha0=PARAMS["alpha0_dir"],
        )
        p_ref = ref_type_stats["p"].to_numpy(dtype=float)

        def build_celltype_proportions(tt):
            XYt = combined.iloc[sample_rows[str(tt)]][["x", "y", "celltype"]].copy()
            stats_t = estimate_type_proportions(
                tr_tree=trees_time[str(tt)],
                R=R,
                labels_factor=XYt["celltype"].astype(str).tolist(),
                levels_all=all_levels,
                alpha0=PARAMS["alpha0_dir"],
            )
            return stats_t["p"].to_numpy(dtype=np.float32)

        celltype_prop_res = parallel_lapply(time_ids, build_celltype_proportions, core=core)
        celltype_proportions_by_sample = {str(ref_sample_id): p_ref.astype(np.float32, copy=False)}
        celltype_proportions_by_sample.update(
            {str(tt): np.asarray(v, dtype=np.float32) for tt, v in zip(time_ids, celltype_prop_res)}
        )

        celltype_js_divergence_by_time = {}
        celltype_js_distance_by_time = {}
        celltype_variance_factor_by_time = {}
        celltype_precision_by_time = {}
        for tt in time_ids:
            js_div, js_dist, phi_cell, precision = _celltype_js_adjustment(
                celltype_proportions_by_sample[str(tt)], p_ref
            )
            celltype_js_divergence_by_time[str(tt)] = js_div.astype(np.float32, copy=False)
            celltype_js_distance_by_time[str(tt)] = js_dist.astype(np.float32, copy=False)
            celltype_variance_factor_by_time[str(tt)] = phi_cell.astype(np.float32, copy=False)
            celltype_precision_by_time[str(tt)] = precision.astype(np.float32, copy=False)
        celltype_adjustment_info = {
            "method": "normalized_jensen_shannon_distance",
            "proportion_estimator": "kernel-weighted local proportions with symmetric Dirichlet smoothing",
            "divergence": "D_JS / log(2), using natural logarithms",
            "distance": "sqrt(D_JS / log(2))",
            "variance_factor": "exp(sqrt(D_JS / log(2)))",
            "precision_factor": "exp(-sqrt(D_JS / log(2)))",
            "variance_factor_range": [1.0, float(np.e)],
            "celltype_levels": [str(x) for x in all_levels],
        }

        return {
            "combined": combined,
            "poly_with_holes": poly_with_holes,
            "grid_eval": grid_eval,
            "Q_grid": Q_grid,
            "R": R,
            "grid_spacing": grid_spacing,
            "R_map": R_map,
            "ref_sample_id": ref_sample_id,
            "time_ids": [str(tt) for tt in time_ids],
            "tr_ref": tr_ref,
            "trees_time": trees_time,
            "celltype_proportions_by_sample": celltype_proportions_by_sample,
            "celltype_js_divergence_by_time": celltype_js_divergence_by_time,
            "celltype_js_distance_by_time": celltype_js_distance_by_time,
            "celltype_variance_factor_by_time": celltype_variance_factor_by_time,
            "celltype_precision_by_time": celltype_precision_by_time,
            "celltype_adjustment_info": celltype_adjustment_info,
            "PARAMS": PARAMS,
            "tr_grid_posthoc": tr_grid_posthoc,
            "hard_radius_loc": hard_radius_loc,
            "gene_cols": gene_cols,
            "auto_geometry": ag,
        }

    def compute_swnd_cos_smooth_local(
        ref_meta,
        smp_align,
        ref_expr,
        smp_expr,
        coord_ref=("x", "y"),
        coord_B_align=("x_aligned", "y_aligned"),
        k_big=25,
        k_smooth=30,
        h_scale=1.5,
        marker_var_q=0.5,
        marker_delta_q=0.3,
        marker_cor_min=0.2,
        max_markers=300,
        use_density_channel=True,
        density_weight=1.0,
        s=0.25,
        k_dens=50,
        h_dens_scale=1.0,
        density_boundary_correct=True,
        density_hull_ratio=0.65,
        density_edge_mass_floor=0.50,
        density_bc_only_inside=True,
        verbose=True,
        return_full=False,
    ):
        require_pandas()
        s = float(s)
        if not np.isfinite(s) or s <= 0 or s >= 1:
            raise ValueError("compute_swnd_cos_smooth_local: s must be in (0, 1).")

        def make_knn(XY_ref, XY_eval, k_req):
            XY_ref = np.asarray(XY_ref, dtype=float)
            XY_eval = np.asarray(XY_eval, dtype=float)
            k_use = min(int(k_req), XY_ref.shape[0])
            if k_use < 1:
                raise ValueError("Invalid k in make_knn().")
            tree = cKDTree(XY_ref)
            dist, idx = tree.query(XY_eval, k=k_use)
            if k_use == 1:
                dist = dist[:, None]
                idx = idx[:, None]
            return {"idx": idx, "dist": dist}

        def bw_from_dist(dist, h_scale=1.0):
            dist = np.asarray(dist, dtype=float)
            k_now = dist.shape[1]
            h = np.nanmedian(dist[:, min(3, k_now - 1)]) if k_now >= 1 else np.nan
            if not np.isfinite(h) or h <= 0:
                h = np.nanmedian(dist[dist > 0])
            if not np.isfinite(h) or h <= 0:
                h = 1.0
            return float(h) * float(h_scale)

        def gauss_weights_from_knn(kn, h_scale=1.0):
            h = bw_from_dist(kn["dist"], h_scale=h_scale)
            w = np.exp(-(kn["dist"] ** 2) / (2 * h ** 2))
            return {"w": w, "h": h}

        def local_moments_from_knn(Cmat, kn, h_scale=1.5):
            Cmat = np.asarray(Cmat, dtype=float)
            G = Cmat.shape[0]
            N_eval = kn["idx"].shape[0]
            k = kn["idx"].shape[1]
            gw = gauss_weights_from_knn(kn, h_scale=h_scale)
            w = gw["w"]
            w_sum = np.sum(w, axis=1)
            w_sum[w_sum <= 0] = 1
            logmu = np.zeros((N_eval, G), dtype=float)
            logr = np.zeros((N_eval, G), dtype=float)
            logitpi = np.zeros((N_eval, G), dtype=float)
            for g in range(G):
                x_nb = Cmat[g][kn["idx"]]
                mu = np.sum(w * x_nb, axis=1) / w_sum
                xc = x_nb - mu[:, None]
                va = np.sum(w * xc ** 2, axis=1) / w_sum
                va = np.maximum(va, 1e-8)
                r = mu ** 2 / np.maximum(va - mu, 1e-8)
                r[(~np.isfinite(r)) | (r <= 0)] = 1e8
                qnb = (1 + mu / np.maximum(r, 1e-8)) ** (-r)
                p0 = np.sum(w * (x_nb == 0), axis=1) / w_sum
                pi0 = np.clip((p0 - qnb) / np.maximum(1 - qnb, 1e-8), 0, 1 - 1e-8)
                logmu[:, g] = np.log1p(mu)
                logr[:, g] = np.log1p(r)
                logitpi[:, g] = np.log(np.clip(pi0, 1e-8, 1 - 1e-8) / np.clip(1 - pi0, 1e-8, 1 - 1e-8))
            return {"logmu": logmu, "logr": logr, "logitpi": logitpi}

        def build_support_boundary(XY_ref, hull_ratio=0.65):
            require_shapely()
            pts = MultiPoint([Point(float(x), float(y)) for x, y in np.asarray(XY_ref, dtype=float)])
            if shapely_concave_hull is not None:
                try:
                    support = shapely_concave_hull(pts, ratio=float(hull_ratio))
                except Exception:
                    support = pts.convex_hull
            else:
                support = pts.convex_hull
            support = largest_polygon(support)
            return {"support": support, "boundary": support.boundary}

        def kernel_density_from_knn(
            kn,
            XY_eval,
            h_scale=1.0,
            boundary_correct=True,
            support_obj=None,
            edge_mass_floor=0.50,
            bc_only_inside=True,
            return_diag=False,
        ):
            N_eval = kn["idx"].shape[0]
            if N_eval < 1:
                out0 = np.array([], dtype=float)
                if return_diag:
                    return {"dens": out0, "dens_raw": out0, "h": np.nan, "d_bnd": out0, "inside": np.array([], dtype=bool), "mass_in": out0}
                return out0
            gw = gauss_weights_from_knn(kn, h_scale=h_scale)
            h = gw["h"]
            dens_raw = np.sum(gw["w"], axis=1)
            dens_raw[~np.isfinite(dens_raw)] = 0
            if (not boundary_correct) or support_obj is None:
                if return_diag:
                    return {
                        "dens": dens_raw,
                        "dens_raw": dens_raw,
                        "h": h,
                        "d_bnd": np.full(N_eval, np.nan),
                        "inside": np.full(N_eval, True),
                        "mass_in": np.ones(N_eval),
                    }
                return dens_raw
            pts = [Point(float(x), float(y)) for x, y in np.asarray(XY_eval, dtype=float)]
            inside = np.asarray([support_obj["support"].contains(pt) or support_obj["support"].touches(pt) for pt in pts], dtype=bool)
            d_bnd = np.asarray([pt.distance(support_obj["boundary"]) for pt in pts], dtype=float)
            d_bnd[~np.isfinite(d_bnd)] = np.inf
            mass_in = stats.norm.cdf(d_bnd / max(h, 1e-12))
            if bc_only_inside:
                mass_in[~inside] = 1
            mass_in = np.maximum(mass_in, float(edge_mass_floor))
            dens = dens_raw / mass_in
            dens[~np.isfinite(dens)] = 0
            if return_diag:
                return {"dens": dens, "dens_raw": dens_raw, "h": h, "d_bnd": d_bnd, "inside": inside, "mass_in": mass_in}
            return dens

        def gaussian_smooth_from_knn(kn, z, h_scale=1.5):
            z = np.asarray(z, dtype=float)
            gw = gauss_weights_from_knn(kn, h_scale=h_scale)
            z_mat = z[kn["idx"]]
            bad = ~np.isfinite(z_mat)
            z_mat[bad] = 0
            gw["w"][bad] = 0
            w_sum = np.sum(gw["w"], axis=1)
            w_sum[w_sum <= 0] = 1
            return np.sum(gw["w"] * z_mat, axis=1) / w_sum

        def spotwise_cos(A, B):
            A = np.asarray(A, dtype=float)
            B = np.asarray(B, dtype=float)
            num = np.sum(A * B, axis=1)
            den = np.sqrt(np.sum(A ** 2, axis=1) * np.sum(B ** 2, axis=1))
            den[(~np.isfinite(den)) | (den < 1e-8)] = 1e-8
            cosv = num / den
            cosv[~np.isfinite(cosv)] = np.nan
            return cosv

        ref_cells = [c for c in ref_expr.columns if c in ref_meta.index]
        smp_cells = [c for c in smp_expr.columns if c in smp_align.index]
        genes = [g for g in ref_expr.index if g in smp_expr.index]
        if verbose:
            print(f"[SWND] N_A={len(ref_cells)}, N_B={len(smp_cells)}, G={len(genes)}")
        if len(ref_cells) < 10 or len(smp_cells) < 10 or len(genes) < 10:
            raise ValueError("Too few cells or genes in ref/smp to compute swnd_cos risk.")

        for nm in coord_ref:
            ref_meta[nm] = pd.to_numeric(ref_meta[nm], errors="coerce")
        for nm in coord_B_align:
            smp_align[nm] = pd.to_numeric(smp_align[nm], errors="coerce")

        XY_A_ref = ref_meta.loc[ref_cells, list(coord_ref)].to_numpy(dtype=float)
        XY_B_align = smp_align.loc[smp_cells, list(coord_B_align)].to_numpy(dtype=float)
        XY_eval = XY_B_align
        CA = ref_expr.loc[genes, ref_cells].to_numpy(dtype=float)
        CB = smp_expr.loc[genes, smp_cells].to_numpy(dtype=float)
        k_AB = max(int(k_big), int(k_dens))
        k_BB = max(int(k_big), int(k_dens), int(k_smooth))
        kn_AB = make_knn(XY_A_ref, XY_eval, k_AB)
        kn_BB = make_knn(XY_B_align, XY_eval, k_BB)
        MA = local_moments_from_knn(CA, {"idx": kn_AB["idx"][:, : min(int(k_big), kn_AB["idx"].shape[1])], "dist": kn_AB["dist"][:, : min(int(k_big), kn_AB["dist"].shape[1])]}, h_scale=h_scale)
        MB = local_moments_from_knn(CB, {"idx": kn_BB["idx"][:, : min(int(k_big), kn_BB["idx"].shape[1])], "dist": kn_BB["dist"][:, : min(int(k_big), kn_BB["dist"].shape[1])]}, h_scale=h_scale)

        logmu_A = MA["logmu"]
        logmu_B = MB["logmu"]
        logr_A = MA["logr"]
        logr_B = MB["logr"]
        logitpi_A = MA["logitpi"]
        logitpi_B = MB["logitpi"]
        muA_g = np.nanmean(logmu_A, axis=0)
        muB_g = np.nanmean(logmu_B, axis=0)
        delta_mu = muA_g - muB_g
        varA_g = col_vars(logmu_A)
        varB_g = col_vars(logmu_B)
        varAB = np.maximum(varA_g, varB_g)
        stab_cor = col_cor_pairwise(logmu_A, logmu_B)
        var_thr = float(np.nanquantile(varAB, marker_var_q))
        if not np.isfinite(var_thr) or var_thr <= 0:
            v_sorted = np.sort(varAB[np.isfinite(varAB)])[::-1]
            var_thr = float(v_sorted[max(0, math.floor(0.5 * len(v_sorted)) - 1)]) if len(v_sorted) else 0.0
        keep_var = np.flatnonzero(varAB >= var_thr)
        keep_cor = np.flatnonzero(stab_cor >= float(marker_cor_min))
        abs_delta = np.abs(delta_mu)
        delta_pool = np.intersect1d(keep_var, keep_cor)
        if delta_pool.size < 20:
            delta_pool = np.flatnonzero(np.isfinite(abs_delta))
        delta_thr = float(np.nanquantile(abs_delta[delta_pool], marker_delta_q)) if delta_pool.size else np.nan
        if not np.isfinite(delta_thr):
            delta_thr = float(np.nanmax(abs_delta[np.isfinite(abs_delta)]))
        delta_thr = max(delta_thr, 0.05)
        keep_delta = np.flatnonzero(abs_delta <= delta_thr)
        keep_all = np.intersect1d(np.intersect1d(keep_var, keep_delta), keep_cor)
        keep_all = keep_all[(keep_all >= 0) & (keep_all < logmu_A.shape[1])]
        if verbose:
            print(
                "[marker-screen] var_thr={:.6g} | delta_thr={:.6g} | cor_min={:.6g} | n_var={} | n_delta={} | n_cor={} | n_all={}".format(
                    var_thr, delta_thr, marker_cor_min, keep_var.size, keep_delta.size, keep_cor.size, keep_all.size
                )
            )
        if keep_all.size < 5:
            score_fallback = np.maximum(stab_cor, 0) * varAB
            score_fallback[abs_delta > max(delta_thr, 0.25)] = -np.inf
            o = np.argsort(score_fallback)[::-1]
            keep_all = o[: min(int(max_markers), int(np.sum(np.isfinite(score_fallback) & (score_fallback > -np.inf))))]
        if keep_all.size > int(max_markers):
            score_align = np.maximum(stab_cor[keep_all], 0) * varAB[keep_all]
            o2 = np.argsort(score_align)[::-1]
            keep_all = keep_all[o2[: int(max_markers)]]
        genes_keep = [genes[i] for i in keep_all]
        if len(genes_keep) < 3:
            raise ValueError("Too few alignment marker genes (<3) to compute risk.")
        if use_density_channel and (density_weight is None or (isinstance(density_weight, str) and density_weight.lower() in {"auto", "autotune"})):
            M_used = len(keep_all)
            d = 3 * M_used
            density_weight = math.sqrt((s / (1 - s)) * d)
        A_pat = np.column_stack([logmu_A[:, keep_all], logr_A[:, keep_all], logitpi_A[:, keep_all]])
        B_pat = np.column_stack([logmu_B[:, keep_all], logr_B[:, keep_all], logitpi_B[:, keep_all]])
        A_z = zcols(A_pat)
        B_z = zcols(B_pat)
        if use_density_channel:
            support_A = build_support_boundary(XY_A_ref, hull_ratio=density_hull_ratio) if density_boundary_correct else None
            support_B = build_support_boundary(XY_B_align, hull_ratio=density_hull_ratio) if density_boundary_correct else None
            densA_obj = kernel_density_from_knn(
                {"idx": kn_AB["idx"][:, : min(int(k_dens), kn_AB["idx"].shape[1])], "dist": kn_AB["dist"][:, : min(int(k_dens), kn_AB["dist"].shape[1])]},
                XY_eval,
                h_scale=h_dens_scale,
                boundary_correct=density_boundary_correct,
                support_obj=support_A,
                edge_mass_floor=density_edge_mass_floor,
                bc_only_inside=density_bc_only_inside,
                return_diag=True,
            )
            densB_obj = kernel_density_from_knn(
                {"idx": kn_BB["idx"][:, : min(int(k_dens), kn_BB["idx"].shape[1])], "dist": kn_BB["dist"][:, : min(int(k_dens), kn_BB["dist"].shape[1])]},
                XY_eval,
                h_scale=h_dens_scale,
                boundary_correct=density_boundary_correct,
                support_obj=support_B,
                edge_mass_floor=density_edge_mass_floor,
                bc_only_inside=density_bc_only_inside,
                return_diag=True,
            )
            dens_A_raw = densA_obj["dens"]
            dens_B_raw = densB_obj["dens"]
            densA_z = zcols(dens_A_raw[:, None]).ravel()
            densB_z = zcols(dens_B_raw[:, None]).ravel()
            densA_z[~np.isfinite(densA_z)] = 0
            densB_z[~np.isfinite(densB_z)] = 0
            A_ext = np.column_stack([A_z, float(density_weight) * densA_z])
            B_ext = np.column_stack([B_z, float(density_weight) * densB_z])
        else:
            dens_A_raw = np.full(XY_eval.shape[0], np.nan)
            dens_B_raw = np.full(XY_eval.shape[0], np.nan)
            densA_z = np.full(XY_eval.shape[0], np.nan)
            densB_z = np.full(XY_eval.shape[0], np.nan)
            densA_obj = None
            densB_obj = None
            A_ext = A_z
            B_ext = B_z
        cos_shape = spotwise_cos(A_z, B_z)
        cos_comb = spotwise_cos(A_ext, B_ext)
        risk_shape = 1 - cos_shape
        risk_comb = 1 - cos_comb
        kn_smooth = {"idx": kn_BB["idx"][:, : min(int(k_smooth), kn_BB["idx"].shape[1])], "dist": kn_BB["dist"][:, : min(int(k_smooth), kn_BB["dist"].shape[1])]}
        risk_shape_smooth = gaussian_smooth_from_knn(kn_smooth, risk_shape, h_scale=h_scale)
        risk_comb_smooth = gaussian_smooth_from_knn(kn_smooth, risk_comb, h_scale=h_scale)
        if not return_full:
            return {"swnd_cos_smooth": risk_comb_smooth}
        scores = pd.DataFrame(
            {
                "swnd_cos": risk_comb,
                "swnd_cos_smooth": risk_comb_smooth,
                "risk_combined": risk_comb,
                "risk_combined_smooth": risk_comb_smooth,
                "cos_combined": cos_comb,
                "risk_shape": risk_shape,
                "risk_shape_smooth": risk_shape_smooth,
                "cos_shape": cos_shape,
                "dens_A_raw": dens_A_raw,
                "dens_B_raw": dens_B_raw,
                "densA_z": densA_z,
                "densB_z": densB_z,
                "barcode_B": smp_cells,
                "dens_A_raw_uncorrected": densA_obj["dens_raw"] if densA_obj is not None else np.nan,
                "dens_B_raw_uncorrected": densB_obj["dens_raw"] if densB_obj is not None else np.nan,
                "densA_mass_in": densA_obj["mass_in"] if densA_obj is not None else np.nan,
                "densB_mass_in": densB_obj["mass_in"] if densB_obj is not None else np.nan,
                "densA_d_bnd": densA_obj["d_bnd"] if densA_obj is not None else np.nan,
                "densB_d_bnd": densB_obj["d_bnd"] if densB_obj is not None else np.nan,
            },
            index=smp_cells,
        )
        fig, ax = plt.subplots(figsize=(5, 5))
        sc = ax.scatter(XY_B_align[:, 0], XY_B_align[:, 1], c=risk_comb_smooth, s=8, cmap="magma")
        ax.set_aspect("equal")
        ax.set_title("Shape + density alignment risk (smoothed)")
        plt.colorbar(sc, ax=ax, label="shape+density risk")
        return {
            "swnd_cos_smooth": risk_comb_smooth,
            "scores": scores,
            "XY_B_align": XY_B_align,
            "marker_genes": genes_keep,
            "plot": fig,
        }

    base = prepare_once_multi_local(combined=combined, ref_sample_id=ref_sample_id, PARAMS=PARAMS, gene_cols_hint=gene_cols_hint, core=core)
    PARAMS.setdefault("uncert", {})
    PARAMS["uncert"].setdefault("swnd_cos", {})
    swnd_cfg = PARAMS["uncert"]["swnd_cos"]
    swnd_cfg["coord_ref"] = ("x", "y")
    swnd_cfg["coord_B_align"] = ("x_aligned", "y_aligned")
    swnd_cfg["density_energy_share"] = s
    PARAMS["uncert"]["swnd_cos"] = swnd_cfg

    user_var_col = or_else(user_var_col, PARAMS["uncert"].get("user_var_col"))
    use_user_var_if_available = bool(PARAMS["uncert"].get("use_user_var_if_available", False))
    R_map = float(base["R_map"])
    norm_center = or_else(PARAMS["uncert"].get("norm_center"), "median")
    norm_scale = or_else(PARAMS["uncert"].get("norm_scale"), "mad")
    floor_zero = bool(or_else(PARAMS["uncert"].get("floor_zero"), True))

    if not control_ids:
        control_ids = [base["ref_sample_id"]]
    control_ids = [cid for cid in map(str, control_ids) if cid in list(map(str, base["time_ids"]))]
    control_ids = list(dict.fromkeys(control_ids + [base["ref_sample_id"]]))
    combined0 = base["combined"]
    sample_rows = {sid: np.flatnonzero(combined0["sample_id"].astype(str).to_numpy() == sid) for sid in combined0["sample_id"].astype(str).drop_duplicates()}
    meta_cols = {
        "sample_id", "x", "y", "x_aligned", "y_aligned", "celltype",
        "volume", "slide_id", "batch", "age", "rownames",
        "x_scaled", "y_scaled", "var_lddmm", ".file",
    }
    num_cols = [c for c in combined0.columns if pd.api.types.is_numeric_dtype(combined0[c])]
    gene_cols_auto = [c for c in num_cols if c not in meta_cols]
    genes_for_qc = [g for g in gene_cols_auto if np.nanquantile(np.abs(pd.to_numeric(combined0[g], errors="coerce") - np.round(pd.to_numeric(combined0[g], errors="coerce"))), 0.99) < 1e-6 and np.nanmin(pd.to_numeric(combined0[g], errors="coerce")) >= 0]

    def make_qc_spot_from_rows(row_idx):
        if genes_for_qc:
            M = combined0.iloc[row_idx][genes_for_qc].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=True)
            M[M < 0] = 0
            lib = np.nansum(M, axis=1)
            det = np.nanmean(M > 0, axis=1)
        else:
            if "volume" in combined0.columns:
                lib = np.maximum(pd.to_numeric(combined0.iloc[row_idx]["volume"], errors="coerce").to_numpy(dtype=float, copy=True), 0)
            else:
                lib = np.ones(len(row_idx), dtype=float)
            det = np.zeros(len(row_idx), dtype=float)
        return np.column_stack([np.log1p(lib), det])

    def accumulate_spot_qc_to_grid_numden(row_idx, tree, R, tw=None):
        hard_radius_here = bool(or_else(base.get("hard_radius_loc"), or_else(base["PARAMS"].get("hard_radius_loc"), True)))
        tw = tree_weights_nocap(tree, R, hard_radius_here) if tw is None else tw
        idx = tw["idx"]
        w = tw["w"]
        qc = make_qc_spot_from_rows(row_idx)
        den = np.sum(w, axis=1) + 1e-12
        num = np.zeros((idx.shape[0], qc.shape[1]), dtype=float)
        for j in range(qc.shape[1]):
            vals = qc[:, j]
            num[:, j] = np.sum(w * vals[idx], axis=1)
        return {"num": num, "den": den}

    sample_names = list(sample_rows.keys())
    qc_acc_res = parallel_lapply(
        sample_names,
        lambda tt: accumulate_spot_qc_to_grid_numden(
            sample_rows[tt],
            base["tr_ref"] if tt == base["ref_sample_id"] else base["trees_time"][tt],
            base["R"],
            tw=tree_weights_nocap(base["tr_ref"] if tt == base["ref_sample_id"] else base["trees_time"][tt], base["R"], bool(or_else(base.get("hard_radius_loc"), or_else(base["PARAMS"].get("hard_radius_loc"), True)))),
        ),
        core=core,
    )
    qc_acc_by_sample = {tt: acc for tt, acc in zip(sample_names, qc_acc_res)}

    def get_ctrl_grid_qc():
        mats = []
        for tt in dict.fromkeys(control_ids):
            acc = qc_acc_by_sample[str(tt)]
            mats.append(acc["num"] / np.maximum(acc["den"][:, None], 1e-12))
        return np.vstack(mats)

    ctrl_qc = get_ctrl_grid_qc()
    mu_qc = np.nanmean(ctrl_qc, axis=0)
    sd_qc = np.nanstd(ctrl_qc, axis=0, ddof=1)
    sd_qc[sd_qc < 1e-6] = 1
    if ctrl_qc.shape[1] >= 2:
        Xc = (ctrl_qc - mu_qc) / sd_qc
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        rot = vt.T[:, : min(vt.shape[0], max(1, kB))]
    else:
        rot = np.eye(1)

    if "batch" not in combined0.columns:
        raise ValueError("[batch_prepare_once_multi] combined is missing the batch column.")
    time_ids_chr = [str(tt) for tt in base["time_ids"]]
    batch_of_time = {}
    for tt in time_ids_chr:
        bset = pd.Series(combined0.iloc[sample_rows[tt]]["batch"], dtype="string").dropna().astype(str).unique().tolist()
        bset = [b for b in bset if b]
        if len(bset) == 0:
            raise ValueError(f"[batch_prepare_once_multi] sample_id={tt} has no valid batch value.")
        if len(bset) > 1:
            raise ValueError(f"[batch_prepare_once_multi] sample_id={tt} maps to multiple batch values.")
        batch_of_time[tt] = bset[0]

    batches = sorted(set(batch_of_time.values()))
    H_by_batch_res = parallel_lapply(
        batches,
        lambda bb: _build_H_for_batch(bb, batch_of_time, qc_acc_by_sample, mu_qc, sd_qc, rot, kB),
        core=core,
    )
    H_by_batch = {bb: H for bb, H in zip(batches, H_by_batch_res)}
    H_by_time = {tt: H_by_batch[batch_of_time[tt]] for tt in time_ids_chr}

    risk_core = min(core, 3)

    def risk_worker(tt):
        row_idx = sample_rows[tt]
        df_t = combined0.iloc[row_idx].copy()
        tree_t = base["trees_time"][tt]
        use_user_var = False
        raw_spot_signal = None
        if use_user_var_if_available and user_var_col in df_t.columns:
            vv = pd.to_numeric(df_t[user_var_col], errors="coerce").to_numpy(dtype=float, copy=True)
            if np.any(np.isfinite(vv)):
                raw_spot_signal = vv
                use_user_var = True
        if not use_user_var:
            AB = extract_AB_for_tid(base["combined"], base["ref_sample_id"], tt, gene_cols=base["gene_cols"])
            cfg = PARAMS["uncert"]["swnd_cos"]
            out = compute_swnd_cos_smooth_local(
                ref_meta=AB["ref_meta"],
                smp_align=AB["smp_align"],
                ref_expr=AB["ref_expr"],
                smp_expr=AB["smp_expr"],
                coord_ref=tuple(or_else(cfg.get("coord_ref"), ("x", "y"))),
                coord_B_align=tuple(or_else(cfg.get("coord_B_align"), ("x_aligned", "y_aligned"))),
                k_big=or_else(cfg.get("k_big"), 25),
                k_smooth=or_else(cfg.get("k_smooth"), 30),
                h_scale=or_else(cfg.get("h_scale"), 1.5),
                marker_var_q=or_else(cfg.get("marker_var_q"), 0.5),
                marker_delta_q=or_else(cfg.get("marker_delta_q"), 0.3),
                marker_cor_min=or_else(cfg.get("marker_cor_min"), 0.2),
                max_markers=or_else(cfg.get("max_markers"), 300),
                use_density_channel=or_else(cfg.get("use_density_channel"), True),
                density_weight=or_else(cfg.get("density_weight"), "auto"),
                s=or_else(cfg.get("density_energy_share"), s),
                k_dens=or_else(cfg.get("k_dens"), 50),
                h_dens_scale=or_else(cfg.get("h_dens_scale"), 1.0),
                verbose=or_else(cfg.get("verbose"), True),
                return_full=False,
            )
            raw_spot_signal = out["swnd_cos_smooth"]
        risk_grid_raw = map_spot_to_grid_mean_gauss_nocap(
            raw_spot_signal,
            tree_t,
            R_map=R_map,
            hard_radius=bool(or_else(base.get("hard_radius_loc"), or_else(base["PARAMS"].get("hard_radius_loc"), True))),
        )
        risk_grid = normalize_risk_grid(risk_grid_raw, center=norm_center, scale=norm_scale, floor_zero=floor_zero)
        return np.asarray(risk_grid, dtype=float)

    risk_res_list = parallel_lapply(time_ids_chr, risk_worker, core=risk_core)
    risk_res = {tt: arr for tt, arr in zip(time_ids_chr, risk_res_list)}
    base["H_by_time"] = H_by_time
    base["PARAMS"] = copy.deepcopy(base.get("PARAMS", {}))
    base["PARAMS"]["uncert"] = copy.deepcopy(PARAMS["uncert"])
    base["risk_time"] = risk_res
    return base


def _build_H_for_batch(bb, batch_of_time, qc_acc_by_sample, mu_qc, sd_qc, rot, kB):
    tids_b = [tt for tt, batch in batch_of_time.items() if batch == bb]
    num_sum = None
    den_sum = None
    for tt in tids_b:
        acc = qc_acc_by_sample[tt]
        num_sum = acc["num"].copy() if num_sum is None else num_sum + acc["num"]
        den_sum = acc["den"].copy() if den_sum is None else den_sum + acc["den"]
    qc_b = num_sum / np.maximum(den_sum[:, None], 1e-12)
    Xs = (qc_b - mu_qc) / sd_qc
    H = Xs @ rot
    k_eff = min(H.shape[1], int(kB))
    H_final = np.zeros((H.shape[0], int(kB)), dtype=float)
    H_final[:, :k_eff] = H[:, :k_eff]
    return H_final


def batch_run_one_gene_and_save_multi_conditional(
    shared,
    gene,
    alpha=None,
    do_plot=None,
    do_expr_plot=None,
    include_Psi=False,
    celltype_adjustment=False,
    include_intercept=True,
    time_contrast="vs_ref",
    risk_in_Wv=False,
    traj_pretest=False,
    screen_consensus=False,
    drawmask_cleanup=False,
    core=1,
    seed=None,
    show_progress=True,
    **kwargs,
):
    if "sampling_gap_adjust" in kwargs:
        legacy_value = bool(kwargs.pop("sampling_gap_adjust"))
        if bool(celltype_adjustment) and legacy_value != bool(celltype_adjustment):
            raise ValueError("Conflicting celltype_adjustment and legacy sampling_gap_adjust values.")
        celltype_adjustment = legacy_value
    core = max(int(core), 1)
    if seed is not None:
        np.random.seed(int(seed))
    if time_contrast not in {"vs_ref", "sequential"}:
        raise ValueError("time_contrast must be 'vs_ref' or 'sequential'.")

    score_type = "t"
    H_by_time_shared = shared.get("H_by_time") or {}
    if H_by_time_shared:
        first_key = next(iter(H_by_time_shared))
        kB_use = min(H_by_time_shared[first_key].shape[1], 2)
    else:
        kB_use = 0
    alpha = float(or_else(alpha, shared.get("PARAMS", {}).get("alpha", 0.05)))
    lambda_B = float(or_else(shared.get("PARAMS", {}).get("lambda_beta"), 10.0))
    neff_min = int(or_else(shared.get("PARAMS", {}).get("neff_min"), 30))
    prior_from_anchor = True
    ctl_q = 0.30
    ctl_tau = 0.60
    traj_lambda = 25.0
    traj_min_obs = 2
    traj_eps = 1e-8
    traj_p_dist = "norm"
    traj_df = None
    screen_tau = 0.60
    screen_n_iter = 1
    verbose = True
    traj_df_eff = 30.0 if traj_df is None or not np.isfinite(float(traj_df)) or float(traj_df) < 5 else float(traj_df)

    if isinstance(gene, str):
        gene = [gene]
    gene = [str(g) for g in gene if g is not None and str(g) != ""]
    gene = list(dict.fromkeys(gene))
    if not gene:
        raise ValueError("gene must contain at least one valid gene name.")
    combined_shared = shared.get("combined")
    gene_pool = list(dict.fromkeys(list(shared.get("gene_cols", [])) + list(combined_shared.columns if combined_shared is not None else [])))
    miss_gene = [g for g in gene if g not in gene_pool]
    if miss_gene:
        raise ValueError("Genes not found in shared['combined']: " + ", ".join(miss_gene))


    def compact_terrain_data_for_return(td):
        keep_nm = [
            "time_ids",
            "ref_sample_id",
            "risk_global_score_by_time",
            "muA_adj_by_time",
            "Wv_by_time",
            "use_by_time",
            "beta_by_time",
            "delta_by_time",
            "gamma_by_time",
            "df_by_time",
            "risk_calibration",
            "celltype_adjustment",
            "celltype_adjustment_info",
            "celltype_precision_by_time",
            "celltype_js_divergence_by_time",
            "celltype_js_distance_by_time",
            "celltype_variance_factor_by_time",
            "auto_geometry",
            "stat_by_time",
            "p_by_time",
            "q_by_time",
            "sig_mask_by_time",
            "frac_support_by_time",
            "mask_cleanup_auto",
            "mask_cleanup_meta_by_time",
            "sig_cc_summary_by_time",
        ]
        return {k: td[k] for k in keep_nm if k in td}

    def chunked_apply(X, FUN, core=1, chunk_size=None, show_progress=False, progress_label="items"):
        items = list(X)
        core = max(int(core), 1)
        if len(items) <= 1 or core <= 1:
            out = [FUN(x) for x in items]
            if show_progress and items:
                print(f"[batch_run] completed {len(items)} / {len(items)} {progress_label}")
            return out
        if chunk_size is None or not np.isfinite(chunk_size) or chunk_size <= 0:
            chunk_size = core
        chunk_size = max(1, int(chunk_size))
        out = [None] * len(items)
        for lo in range(0, len(items), chunk_size):
            hi = min(len(items), lo + chunk_size)
            idx = list(range(lo, hi))
            vals = thread_map([items[i] for i in idx], FUN, core=min(core, len(idx)))
            for j, val in zip(idx, vals):
                out[j] = val
            if show_progress:
                print(f"[batch_run] completed {hi} / {len(items)} {progress_label}")
        return out

    def build_shared_static_for_outer(shared, gene_vec):
        combined = shared.get("combined")
        if combined is None:
            return {"shared_static": shared, "gene_col_list": None}
        gene_cols_all = shared.get("gene_cols", [])
        meta_cols = [c for c in combined.columns if c not in gene_cols_all]
        shared_static = copy.copy(shared)
        shared_static["combined"] = combined.loc[:, meta_cols].copy()
        shared_static["gene_cols"] = []
        gene_col_list = {g: combined.loc[:, [g]].copy() for g in gene_vec}
        return {"shared_static": shared_static, "gene_col_list": gene_col_list}

    def build_shared_for_one_gene(shared_static, gene_col_list, g):
        sh = copy.copy(shared_static)
        if gene_col_list is not None:
            sh["combined"] = pd.concat([shared_static["combined"].reset_index(drop=True), gene_col_list[g].reset_index(drop=True)], axis=1)
            sh["gene_cols"] = [g]
        return sh

    if len(gene) > 1:
        gene_vec = gene
        outer_core = min(max(1, int(core)), len(gene_vec))
        ss = build_shared_static_for_outer(shared, gene_vec)
        shared_static = ss["shared_static"]
        gene_col_list = ss["gene_col_list"]

        def run_one_gene(ii):
            g = gene_vec[ii]
            seed_i = None if seed is None else int(seed) + ii
            shared_i = build_shared_for_one_gene(shared_static, gene_col_list, g)
            return batch_run_one_gene_and_save_multi_conditional(
                shared=shared_i,
                gene=g,
                alpha=alpha,
                include_Psi=include_Psi,
                celltype_adjustment=celltype_adjustment,
                include_intercept=include_intercept,
                time_contrast=time_contrast,
                risk_in_Wv=risk_in_Wv,
                traj_pretest=traj_pretest,
                screen_consensus=screen_consensus,
                drawmask_cleanup=drawmask_cleanup,
                core=1,
                seed=seed_i,
                show_progress=False,
            )

        out_list = chunked_apply(
            range(len(gene_vec)),
            run_one_gene,
            core=outer_core if outer_core > 1 else 1,
            chunk_size=outer_core if outer_core > 1 else 1,
            show_progress=bool(show_progress),
            progress_label="genes",
        )
        return {g: res for g, res in zip(gene_vec, out_list)}

    gene = gene[0]

    def _run_impl(
        risk_in_Wv,
        traj_pretest,
        screen_consensus,
        drawmask_cleanup,
        verbose,
        core,
        _skip_risk_autocalib=False,
        celltype_adjustment_inner=None,
    ):
        core = max(int(core), 1)
        celltype_adjustment_use = bool(celltype_adjustment) if celltype_adjustment_inner is None else bool(celltype_adjustment_inner)
        STORE_RAW_OUTPUT = bool(_skip_risk_autocalib)
        cleanup_cfg = shared.get("PARAMS", {}).get("drawmask_cleanup", {})
        if drawmask_cleanup is None:
            drawmask_cleanup = bool(or_else(cleanup_cfg.get("enable"), False))
        else:
            drawmask_cleanup = bool(drawmask_cleanup)

        def as_logical_matrix(x):
            arr = np.array(x, dtype=bool, copy=True)
            if arr.ndim == 1:
                arr = arr[:, None]
            arr[np.isnan(arr.astype(float))] = False
            return arr

        def as_numeric_matrix(x):
            arr = np.array(x, dtype=float, copy=True)
            if arr.ndim == 1:
                arr = arr[:, None]
            arr[~np.isfinite(arr)] = 0
            return arr

        def grid_index_map(grid_df):
            ux = np.sort(pd.unique(pd.to_numeric(grid_df["x"], errors="coerce")))
            uy = np.sort(pd.unique(pd.to_numeric(grid_df["y"], errors="coerce")))
            ix = np.searchsorted(ux, pd.to_numeric(grid_df["x"], errors="coerce").to_numpy(dtype=float))
            iy = np.searchsorted(uy, pd.to_numeric(grid_df["y"], errors="coerce").to_numpy(dtype=float))
            return {"ux": ux, "uy": uy, "ix": ix, "iy": iy, "nr": int(len(uy)), "nc": int(len(ux))}

        def vec_to_mask_mat(mask_vec, grid_df, gi=None):
            gi = grid_index_map(grid_df) if gi is None else gi
            vv = np.asarray(mask_vec, dtype=bool)
            if vv.shape[0] != gi["ix"].shape[0]:
                raise ValueError(f"vec_to_mask_mat: length(mask_vec)={vv.shape[0]}, expected={gi['ix'].shape[0]}")
            mat = np.zeros((gi["nr"], gi["nc"]), dtype=bool)
            mat[gi["iy"], gi["ix"]] = vv
            return {"mat": mat, "gi": gi}

        def mask_mat_to_vec(mat, gi):
            arr = np.asarray(mat, dtype=bool)
            if arr.ndim == 1:
                if arr.size != gi["nr"] * gi["nc"]:
                    raise ValueError("mask_mat_to_vec: non-matrix input length does not match nr*nc.")
                arr = arr.reshape(gi["nr"], gi["nc"])
            if arr.shape != (gi["nr"], gi["nc"]):
                raise ValueError("mask_mat_to_vec: matrix shape does not match grid index map.")
            out = arr[gi["iy"], gi["ix"]]
            out[np.isnan(out.astype(float))] = False
            return out.astype(bool)

        def box_sum2d(mat, r=1):
            r = max(0, int(r))
            X = as_numeric_matrix(mat)
            nr, nc = X.shape
            if r <= 0:
                return X
            S = X.copy()
            if nc >= 2:
                for j in range(1, nc):
                    S[:, j] += S[:, j - 1]
            if nr >= 2:
                for i in range(1, nr):
                    S[i, :] += S[i - 1, :]
            II = np.zeros((nr + 1, nc + 1), dtype=float)
            II[1:, 1:] = S
            i_lo = np.maximum(np.arange(nr) - r, 0)
            i_hi = np.minimum(np.arange(nr) + r, nr - 1)
            j_lo = np.maximum(np.arange(nc) - r, 0)
            j_hi = np.minimum(np.arange(nc) + r, nc - 1)
            out = np.zeros((nr, nc), dtype=float)
            for i in range(nr):
                r1 = i_lo[i]
                r2 = i_hi[i]
                out[i, :] = II[r2 + 1, j_hi + 1] - II[r1, j_hi + 1] - II[r2 + 1, j_lo] + II[r1, j_lo]
            return out

        def binary_dilate(mat, r=1):
            return box_sum2d(as_logical_matrix(mat), r=r) > 0 if int(r) > 0 else as_logical_matrix(mat)

        def binary_erode(mat, r=1):
            r = max(0, int(r))
            if r <= 0:
                return as_logical_matrix(mat)
            need = (2 * r + 1) ** 2
            return box_sum2d(as_logical_matrix(mat), r=r) >= need

        def binary_close(mat, r=1):
            return binary_erode(binary_dilate(mat, r=r), r=r)

        def binary_open(mat, r=1):
            return binary_dilate(binary_erode(mat, r=r), r=r)

        def cc4_scan_mat(mm):
            mm = as_logical_matrix(mm)
            nr, nc = mm.shape
            lab = np.zeros((nr, nc), dtype=int)
            n_on = int(np.sum(mm))
            if n_on <= 0:
                return {"label": lab, "size": np.array([], dtype=int), "bbox": [], "touch_boundary": np.array([], dtype=bool)}
            cur = 0
            sizes = []
            bboxs = []
            touch = []
            for i in range(nr):
                for j in range(nc):
                    if (not mm[i, j]) or lab[i, j] != 0:
                        continue
                    cur += 1
                    queue = [(i, j)]
                    lab[i, j] = cur
                    head = 0
                    imin = imax = i
                    jmin = jmax = j
                    touch_bd = i == 0 or i == nr - 1 or j == 0 or j == nc - 1
                    while head < len(queue):
                        ii, jj = queue[head]
                        head += 1
                        for ni, nj in ((ii - 1, jj), (ii + 1, jj), (ii, jj - 1), (ii, jj + 1)):
                            if 0 <= ni < nr and 0 <= nj < nc and mm[ni, nj] and lab[ni, nj] == 0:
                                lab[ni, nj] = cur
                                queue.append((ni, nj))
                                imin = min(imin, ni)
                                imax = max(imax, ni)
                                jmin = min(jmin, nj)
                                jmax = max(jmax, nj)
                                if ni == 0 or ni == nr - 1 or nj == 0 or nj == nc - 1:
                                    touch_bd = True
                    sizes.append(len(queue))
                    bboxs.append(np.array([imin, imax, jmin, jmax], dtype=int))
                    touch.append(touch_bd)
            return {
                "label": lab,
                "size": np.asarray(sizes, dtype=int),
                "bbox": bboxs,
                "touch_boundary": np.asarray(touch, dtype=bool),
            }

        def auto_cleanup_cfg(grid_df, R, neff_min, k_loc, auto_geometry=None, cfg=None):
            cfg = {} if cfg is None else cfg
            ux = np.sort(pd.unique(pd.to_numeric(grid_df["x"], errors="coerce")))
            uy = np.sort(pd.unique(pd.to_numeric(grid_df["y"], errors="coerce")))
            dx = float(np.nanmedian(np.diff(ux))) if ux.size >= 2 else 1.0
            dy = float(np.nanmedian(np.diff(uy))) if uy.size >= 2 else 1.0
            if not np.isfinite(dx) or dx <= 0:
                dx = 1.0
            if not np.isfinite(dy) or dy <= 0:
                dy = 1.0
            pix_area = dx * dy
            neigh_area = math.pi * (float(R) ** 2)
            base_cells = neigh_area / pix_area if pix_area > 0 else 1.0
            if not np.isfinite(base_cells) or base_cells <= 0:
                base_cells = 1.0
            r_px = math.sqrt(base_cells / math.pi)
            neff_summary = (((auto_geometry or {}).get("diagnostics") or {}).get("neff_summary"))
            neff_med = float(neff_summary[1]) if neff_summary is not None and len(neff_summary) >= 2 else float(neff_min)
            if not np.isfinite(neff_med) or neff_med <= 0:
                neff_med = float(neff_min) if neff_min > 0 else 1.0
            k_loc_eff = float(k_loc) if k_loc is not None else 10.0
            if not np.isfinite(k_loc_eff) or k_loc_eff <= 0:
                k_loc_eff = 10.0
            cleanup_scale = float(or_else(cfg.get("cleanup_scale"), 1.0))
            if not np.isfinite(cleanup_scale) or cleanup_scale <= 0:
                cleanup_scale = 1.0
            cc_scale = float(or_else(cfg.get("cc_scale"), 0.60))
            close_scale = float(or_else(cfg.get("close_scale"), 0.18))
            open_scale = float(or_else(cfg.get("open_scale"), 0.08))
            hole_scale = float(or_else(cfg.get("hole_scale"), 0.35))
            smooth_thr = float(or_else(cfg.get("smooth_threshold"), 0.50))
            hole_skip_bbox_cells = float(or_else(cfg.get("hole_skip_bbox_cells"), 60000))
            hole_skip_bbox_frac = float(or_else(cfg.get("hole_skip_bbox_frac"), 0.40))
            stability_fac = math.sqrt(max(neff_med, 1.0) / max(float(neff_min), 1.0))
            k_shrink = math.sqrt(10.0 / max(k_loc_eff, 10.0))
            min_cc = int(math.ceil(max(30.0, cc_scale * cleanup_scale * base_cells * stability_fac)))
            close_r = int(round(close_scale * cleanup_scale * r_px * k_shrink))
            open_r = int(round(open_scale * cleanup_scale * r_px * k_shrink))
            close_r = max(0, min(close_r, 6))
            open_r = max(0, min(open_r, close_r))
            hole_max = int(math.ceil(max(10.0, hole_scale * cleanup_scale * base_cells)))
            smooth_iter = 1 if (r_px * stability_fac) >= 4 else 0
            return {
                "dx": dx,
                "dy": dy,
                "pix_area": pix_area,
                "neigh_area": neigh_area,
                "base_cells": base_cells,
                "r_px": r_px,
                "neff_med": neff_med,
                "k_loc_eff": k_loc_eff,
                "min_cc": min_cc,
                "close_r": close_r,
                "open_r": open_r,
                "hole_max": hole_max,
                "smooth_iter": smooth_iter,
                "smooth_threshold": smooth_thr,
                "grid_cells": int(grid_df.shape[0]),
                "hole_skip_bbox_cells": hole_skip_bbox_cells,
                "hole_skip_bbox_frac": hole_skip_bbox_frac,
            }

        def fill_small_holes(mask_mat, hole_max=0, skip_if_large=True, large_bbox_cells=np.inf, large_bbox_frac=1.0, full_grid_cells=None):
            mask_mat = as_logical_matrix(mask_mat)
            hole_max = max(0, int(hole_max))
            if hole_max <= 0:
                return {"mask": mask_mat, "skipped": False, "reason": None}
            bbox_cells = int(mask_mat.size)
            frac_now = bbox_cells / full_grid_cells if full_grid_cells and np.isfinite(full_grid_cells) and full_grid_cells > 0 else 0.0
            if skip_if_large:
                cond1 = np.isfinite(large_bbox_cells) and bbox_cells >= large_bbox_cells
                cond2 = np.isfinite(large_bbox_frac) and frac_now >= large_bbox_frac
                if cond1 or cond2:
                    return {"mask": mask_mat, "skipped": True, "reason": f"large_bbox({bbox_cells} cells)"}
            bg = ~mask_mat
            scan = cc4_scan_mat(bg)
            if scan["size"].size == 0:
                return {"mask": mask_mat, "skipped": False, "reason": None}
            fill_ids = np.flatnonzero((~scan["touch_boundary"]) & (scan["size"] <= hole_max)) + 1
            if fill_ids.size > 0:
                mask_mat[np.isin(scan["label"], fill_ids)] = True
            return {"mask": mask_mat, "skipped": False, "reason": None}

        def geom_clean_mask_mat_fast(mm, auto_cfg):
            mm = as_logical_matrix(mm)
            if auto_cfg["close_r"] > 0:
                mm = binary_close(mm, r=auto_cfg["close_r"])
            if auto_cfg["open_r"] > 0:
                mm = binary_open(mm, r=auto_cfg["open_r"])
            hole_info = {"skipped": False, "reason": None}
            if auto_cfg["hole_max"] > 0:
                tmp_hole = fill_small_holes(
                    mask_mat=mm,
                    hole_max=auto_cfg["hole_max"],
                    skip_if_large=True,
                    large_bbox_cells=auto_cfg["hole_skip_bbox_cells"],
                    large_bbox_frac=auto_cfg["hole_skip_bbox_frac"],
                    full_grid_cells=auto_cfg["grid_cells"],
                )
                mm = tmp_hole["mask"]
                hole_info = {"skipped": tmp_hole["skipped"], "reason": tmp_hole["reason"]}
            smooth_field = mm.astype(float)
            if auto_cfg["smooth_iter"] > 0:
                for _ in range(auto_cfg["smooth_iter"]):
                    smooth_field = box_sum2d(smooth_field, r=1) / 9.0
            work_mm = smooth_field >= auto_cfg["smooth_threshold"]
            work_mm[np.isnan(work_mm)] = False
            cc_scan = cc4_scan_mat(work_mm)
            keep_ids = np.flatnonzero(cc_scan["size"] >= auto_cfg["min_cc"]) + 1
            out_mm = np.isin(cc_scan["label"], keep_ids) if keep_ids.size > 0 else np.zeros_like(work_mm, dtype=bool)
            cc_final = {"label": cc_scan["label"], "size": cc_scan["size"], "keep_ids": keep_ids.astype(int), "keep": out_mm}
            return {"mask_mat": out_mm, "smooth_field_mat": smooth_field, "cc": cc_final, "auto_cfg": auto_cfg, "hole_info": hole_info}

        def merge_rects(rects):
            if rects is None or len(rects) == 0:
                return np.zeros((0, 4), dtype=int)
            R = np.vstack(rects).astype(int)
            changed = True
            while changed and R.shape[0] > 1:
                changed = False
                used = np.zeros(R.shape[0], dtype=bool)
                out_rects = []
                for i in range(R.shape[0]):
                    if used[i]:
                        continue
                    cur = R[i].copy()
                    used[i] = True
                    hit = True
                    while hit:
                        hit = False
                        for j in range(R.shape[0]):
                            if used[j]:
                                continue
                            oth = R[j]
                            overlap = not (
                                cur[1] < oth[0] - 1
                                or oth[1] < cur[0] - 1
                                or cur[3] < oth[2] - 1
                                or oth[3] < cur[2] - 1
                            )
                            if overlap:
                                cur = np.array([min(cur[0], oth[0]), max(cur[1], oth[1]), min(cur[2], oth[2]), max(cur[3], oth[3])], dtype=int)
                                used[j] = True
                                hit = True
                                changed = True
                    out_rects.append(cur)
                R = np.vstack(out_rects)
            return R

        def cleanup_mask_fast(mask_vec, grid_df, gi_full, auto_cfg, bbox_pad=None, small_mask_n=0):
            mask_vec = np.asarray(mask_vec, dtype=bool)
            n_before = int(np.sum(mask_vec))
            out0 = mask_vec.copy()
            meta = {
                "applied": False,
                "skipped_reason": None,
                "n_before": n_before,
                "n_after": n_before,
                "n_local": 0,
                "n_boxes": 0,
                "bbox": None,
                "cc_sizes": np.array([], dtype=int),
                "n_hole_skip_boxes": 0,
            }
            if n_before == 0:
                meta["skipped_reason"] = "empty_mask"
                return {"mask": out0, "meta": meta}
            small_mask_n = max(0, int(small_mask_n))
            if n_before <= small_mask_n:
                meta["skipped_reason"] = "small_mask"
                return {"mask": out0, "meta": meta}
            if bbox_pad is None:
                bbox_pad = max(2, int(auto_cfg["close_r"]), int(auto_cfg["open_r"]), int(auto_cfg["smooth_iter"]) + 1)
            bbox_pad = max(0, int(bbox_pad))
            mm_full = np.zeros((gi_full["nr"], gi_full["nc"]), dtype=bool)
            mm_full[gi_full["iy"], gi_full["ix"]] = mask_vec
            scan0 = cc4_scan_mat(mm_full)
            if scan0["size"].size == 0:
                meta["skipped_reason"] = "no_component"
                return {"mask": out0, "meta": meta}
            rects = [np.array([max(0, bb[0] - bbox_pad), min(gi_full["nr"] - 1, bb[1] + bbox_pad), max(0, bb[2] - bbox_pad), min(gi_full["nc"] - 1, bb[3] + bbox_pad)], dtype=int) for bb in scan0["bbox"]]
            rects = merge_rects(rects)
            mm_out = np.zeros_like(mm_full, dtype=bool)
            cc_sizes_all = []
            n_local_total = 0
            n_hole_skip_boxes = 0
            for k in range(rects.shape[0]):
                iy_lo, iy_hi, ix_lo, ix_hi = rects[k]
                local_mm = mm_full[iy_lo : iy_hi + 1, ix_lo : ix_hi + 1]
                cln_k = geom_clean_mask_mat_fast(local_mm, auto_cfg=auto_cfg)
                mm_out[iy_lo : iy_hi + 1, ix_lo : ix_hi + 1] |= cln_k["mask_mat"]
                n_local_total += local_mm.size
                if cln_k["cc"]["size"].size > 0:
                    cc_sizes_all.extend(cln_k["cc"]["size"].tolist())
                if cln_k["hole_info"]["skipped"]:
                    n_hole_skip_boxes += 1
            out = mm_out[gi_full["iy"], gi_full["ix"]].astype(bool)
            meta["applied"] = True
            meta["n_after"] = int(np.sum(out))
            meta["n_local"] = int(n_local_total)
            meta["n_boxes"] = int(rects.shape[0])
            meta["bbox"] = rects
            meta["cc_sizes"] = np.asarray(sorted(cc_sizes_all, reverse=True), dtype=int)
            meta["n_hole_skip_boxes"] = int(n_hole_skip_boxes)
            return {"mask": out, "meta": meta}

        risk_cfg = shared.get("PARAMS", {}).get("risk", {})
        risk_weight_norm = risk_cfg.get("weight_norm", "qcap")
        risk_cap_q = float(or_else(risk_cfg.get("cap_q"), 0.95))
        risk_power_fixed = 2.0
        risk_global_rule = risk_cfg.get("global_rule", "frac_above")
        risk_global_thr = float(or_else(risk_cfg.get("global_thr"), 0.30))
        risk_global_cap = float(or_else(risk_cfg.get("global_cap"), 0.50))
        risk_wv_floor_frac = float(or_else(risk_cfg.get("wv_floor_frac"), 0))
        risk_wv_min = float(or_else(risk_cfg.get("wv_min"), 1e-30))
        risk_auto_cfg = risk_cfg.get("auto", {})
        risk_local_lambda = 0.0
        risk_global_lambda = 0.0
        risk_calibration = None

        if risk_in_Wv and not _skip_risk_autocalib:
            dry_fit = _run_impl(
                risk_in_Wv=False,
                traj_pretest=False,
                screen_consensus=False,
                drawmask_cleanup=False,
                verbose=False,
                core=1,
                _skip_risk_autocalib=True,
                celltype_adjustment_inner=False,
            )
            calib_time_ids = [
                str(value)
                for value in (dry_fit.get("terrain_data") or {}).get("time_ids", [])
            ]
            risk_time_map = shared.get("risk_time") or {}
            risk_key_by_id = {str(key): key for key in risk_time_map}
            bins_auto = int(or_else(risk_auto_cfg.get("bins"), 10))
            min_bin_n_auto = int(or_else(risk_auto_cfg.get("min_bin_n"), 200))
            tau_anchor_q_auto = float(or_else(risk_auto_cfg.get("tau_anchor_q"), 0.80))
            slack_auto = float(or_else(risk_auto_cfg.get("slack"), 1))
            lam_local_cap_auto = float(or_else(risk_auto_cfg.get("lam_local_cap"), 5e4))
            huber_kappa_auto = float(or_else(risk_auto_cfg.get("huber_kappa"), 1.345))
            provisional_by_time = {}

            # First apply the existing within-contrast calibration separately
            # to every contrast. Failed fallback calibrations are excluded;
            # a successfully estimated coefficient of zero remains valid.
            for time_id_contrast in calib_time_ids:
                record = {
                    "time_id_contrast": time_id_contrast,
                    "time_id_risk": time_id_contrast,
                    "valid": False,
                    "validity_reasons": tuple(),
                    "lambda_local_hat": np.nan,
                    "lambda_global_hat": np.nan,
                    "r_cap": np.nan,
                    "global_score": np.nan,
                    "calibration": None,
                    "empnull": None,
                }
                risk_key = risk_key_by_id.get(time_id_contrast)
                if risk_key is None:
                    record["validity_reasons"] = (
                        "missing_contrast_specific_risk_map",
                    )
                    provisional_by_time[time_id_contrast] = record
                    continue
                try:
                    off_info = get_t_df_use_p_from_fit(dry_fit, time_id_contrast)
                    normalized_risk = compute_r01_and_global_score(
                        risk_vec=risk_time_map[risk_key],
                        weight_norm=risk_weight_norm,
                        cap_q=risk_cap_q,
                        global_rule=risk_global_rule,
                        global_thr=risk_global_thr,
                        global_cap=risk_global_cap,
                    )
                    calibration = calibrate_lambdas_empnull_scale(
                        tvec=off_info["t"],
                        use_mask=off_info["use"],
                        r01=normalized_risk["r01"],
                        global_score=normalized_risk["global_score"],
                        df=off_info["df"],
                        bins=bins_auto,
                        min_bin_n=min_bin_n_auto,
                        p_grid=risk_power_fixed,
                        tau_anchor_q=tau_anchor_q_auto,
                        slack=slack_auto,
                        lam_local_cap=lam_local_cap_auto,
                        verbose=False,
                    )
                    validity = validate_provisional_calibration(
                        calibration,
                        min_bin_n=min_bin_n_auto,
                        lam_local_cap=lam_local_cap_auto,
                    )
                    record.update({
                        "valid": bool(validity["valid"]),
                        "validity_reasons": validity["reasons"],
                        "validity": validity,
                        "lambda_local_hat": float(
                            or_else(calibration.get("lambda_local_hat"), np.nan)
                        ),
                        "lambda_global_hat": float(
                            or_else(calibration.get("lambda_global_hat"), np.nan)
                        ),
                        "at_local_cap": bool(validity["at_local_cap"]),
                        "r_cap": normalized_risk.get("r_cap", np.nan),
                        "global_score": normalized_risk.get("global_score", np.nan),
                        "calibration": calibration,
                        "empnull": calibration,
                    })
                except Exception as error:
                    record.update({
                        "validity_reasons": ("calibration_exception",),
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    })
                provisional_by_time[time_id_contrast] = record

            # Then obtain one shared coefficient for the gene from an
            # equal-weight Huber center of the valid provisional estimates.
            valid_time_ids = [
                time_id for time_id in calib_time_ids
                if provisional_by_time[time_id].get("valid", False)
            ]
            invalid_time_ids = [
                time_id for time_id in calib_time_ids
                if not provisional_by_time[time_id].get("valid", False)
            ]
            provisional_values = [
                provisional_by_time[time_id]["lambda_local_hat"]
                for time_id in valid_time_ids
            ]
            risk_local_lambda, aggregation_diag = huber_center_nonnegative(
                provisional_values,
                kappa=huber_kappa_auto,
            )
            risk_local_lambda = min(
                max(float(risk_local_lambda), 0.0),
                lam_local_cap_auto,
            )
            risk_global_lambda = 0.0
            aggregation_diag.update({
                "valid_time_ids": tuple(valid_time_ids),
                "invalid_time_ids": tuple(invalid_time_ids),
                "equal_weights": (
                    {
                        time_id: 1.0 / len(valid_time_ids)
                        for time_id in valid_time_ids
                    }
                    if valid_time_ids
                    else {}
                ),
                "lambda_local_hat": risk_local_lambda,
                "lambda_global_hat": risk_global_lambda,
            })
            sole_time_id = calib_time_ids[0] if len(calib_time_ids) == 1 else None
            sole_record = (
                provisional_by_time.get(sole_time_id, {})
                if sole_time_id is not None
                else {}
            )
            sole_calibration = sole_record.get("calibration")
            risk_calibration = {
                "status": aggregation_diag.get("status"),
                "method": MULTI_CONTRAST_CALIBRATION_MODE,
                # These two fields remain populated only for a genuine
                # single-contrast fit for compatibility with older payloads.
                "time_id_anchor": sole_time_id,
                "time_id_risk": sole_time_id,
                "lambda_local_hat": risk_local_lambda,
                "lambda_global_hat": risk_global_lambda,
                "n_contrasts_total": int(len(calib_time_ids)),
                "n_contrasts_valid": int(len(valid_time_ids)),
                "valid_time_ids": tuple(valid_time_ids),
                "invalid_time_ids": tuple(invalid_time_ids),
                "provisional_by_time": provisional_by_time,
                "aggregation": aggregation_diag,
                "r_cap": sole_record.get("r_cap", np.nan),
                "global_score": sole_record.get("global_score", np.nan),
                "calibration": sole_calibration,
                "empnull": sole_calibration,
            }
            if verbose:
                print(
                    f"[risk-calib-aggregate] valid={len(valid_time_ids)}/"
                    f"{len(calib_time_ids)} method=Huber(kappa={huber_kappa_auto:g}) "
                    f"lambda_local={risk_local_lambda:.6g}"
                )

        combined = shared["combined"]
        if gene not in combined.columns:
            raise ValueError(f"gene '{gene}' is not present in combined.")
        if include_Psi and not shared.get("H_by_time"):
            raise ValueError("include_Psi=True requires shared['H_by_time'].")

        T_raw = [str(t) for t in shared["time_ids"]]
        num_key = infer_time_numeric(T_raw)
        if np.all(np.isfinite(num_key)):
            ord_idx = np.argsort(num_key)
            T_ids = [T_raw[i] for i in ord_idx]
        else:
            T_ids = T_raw
        N = shared["grid_eval"].shape[0]
        ref_id_chr = str(shared.get("ref_sample_id"))
        sample_rows = {sid: np.flatnonzero(combined["sample_id"].astype(str).to_numpy() == sid) for sid in combined["sample_id"].astype(str).drop_duplicates()}
        if time_contrast == "sequential":
            full_ids = [ref_id_chr] + T_ids
            prev_id_for_t = {}
            for k, sid in enumerate(full_ids):
                if sid == ref_id_chr:
                    continue
                prev_id_for_t[sid] = ref_id_chr if k <= 0 else full_ids[k - 1]
        else:
            prev_id_for_t = {}
        gi_grid_eval = grid_index_map(shared["grid_eval"])
        auto_cln_cfg_global = None
        cleanup_small_mask_n = 0
        cleanup_bbox_pad = None
        if drawmask_cleanup:
            auto_cln_cfg_global = auto_cleanup_cfg(
                grid_df=shared["grid_eval"],
                R=shared["R"],
                neff_min=neff_min,
                k_loc=or_else(shared.get("k_loc"), or_else(shared.get("PARAMS", {}).get("k_loc"), 10)),
                auto_geometry=or_else(shared.get("auto_geometry"), shared.get("PARAMS", {}).get("auto_geometry_resolved")),
                cfg=cleanup_cfg,
            )
            cleanup_small_mask_n = int(or_else(cleanup_cfg.get("small_mask_n"), max(0, min(25, math.ceil(auto_cln_cfg_global["min_cc"] / 8)))))
            cleanup_bbox_pad = int(or_else(cleanup_cfg.get("bbox_pad"), max(2, auto_cln_cfg_global["close_r"], auto_cln_cfg_global["open_r"], auto_cln_cfg_global["smooth_iter"] + 1)))

        zero_comp_by_time_local = {t: np.zeros((N, 0), dtype=float) for t in T_ids}
        celltype_precision_source = shared.get("celltype_precision_by_time") or {}
        celltype_proportion_source = shared.get("celltype_proportions_by_sample") or {}
        celltype_precision_local = {}
        celltype_js_divergence_local = {}
        celltype_js_distance_local = {}
        celltype_variance_factor_local = {}
        if celltype_adjustment_use and not celltype_proportion_source:
            raise ValueError("celltype_adjustment=True requires shared['celltype_proportions_by_sample']; rerun batch_prepare_once_multi.")

        def celltype_precision_for_contrast(target_id, baseline_id):
            p_t = celltype_proportion_source.get(str(target_id))
            p_r = celltype_proportion_source.get(str(baseline_id))
            if p_t is None or p_r is None:
                raise ValueError(f"Missing local cell-type proportions for contrast {target_id} vs {baseline_id}.")
            p_t = np.asarray(p_t, dtype=float)
            p_r = np.asarray(p_r, dtype=float)
            if p_t.shape != p_r.shape or p_t.ndim != 2 or p_t.shape[0] != N:
                raise ValueError(f"Invalid local cell-type proportion arrays for contrast {target_id} vs {baseline_id}.")
            js_div, js_dist, phi_cell, precision = _celltype_js_adjustment(p_t, p_r)
            return precision, js_div, js_dist, phi_cell

        for t in T_ids:
            if celltype_adjustment_use:
                baseline_id = ref_id_chr if time_contrast == "vs_ref" else prev_id_for_t.get(t, ref_id_chr)
                if baseline_id == ref_id_chr and t in celltype_precision_source:
                    precision = np.asarray(celltype_precision_source[t], dtype=float).reshape(-1)
                    js_div = np.asarray((shared.get("celltype_js_divergence_by_time") or {})[t], dtype=float).reshape(-1)
                    js_dist = np.asarray((shared.get("celltype_js_distance_by_time") or {})[t], dtype=float).reshape(-1)
                    phi_cell = np.asarray((shared.get("celltype_variance_factor_by_time") or {})[t], dtype=float).reshape(-1)
                else:
                    precision, js_div, js_dist, phi_cell = celltype_precision_for_contrast(t, baseline_id)
                if any(np.asarray(x).shape[0] != N for x in (precision, js_div, js_dist, phi_cell)):
                    raise ValueError(f"Cell-type adjustment map length does not match the shared grid for time_id={t}.")
                precision = np.clip(np.asarray(precision, dtype=float), np.exp(-1.0), 1.0)
                precision[~np.isfinite(precision)] = 1.0
                celltype_precision_local[t] = precision
                celltype_js_divergence_local[t] = np.clip(np.asarray(js_div, dtype=float), 0.0, 1.0)
                celltype_js_distance_local[t] = np.clip(np.asarray(js_dist, dtype=float), 0.0, 1.0)
                celltype_variance_factor_local[t] = np.clip(np.asarray(phi_cell, dtype=float), 1.0, np.e)
            else:
                celltype_precision_local[t] = np.ones(N, dtype=float)
                celltype_js_divergence_local[t] = np.zeros(N, dtype=float)
                celltype_js_distance_local[t] = np.zeros(N, dtype=float)
                celltype_variance_factor_local[t] = np.ones(N, dtype=float)

        Psi_raw_by_time_local = {}
        for t in T_ids:
            if not include_Psi or not shared.get("H_by_time") or kB_use <= 0 or t not in shared["H_by_time"]:
                Psi_raw_by_time_local[t] = np.zeros((N, 0), dtype=float)
            else:
                Psi_raw_by_time_local[t] = np.asarray(shared["H_by_time"][t][:, : min(shared["H_by_time"][t].shape[1], kB_use)], dtype=float)

        def get_C_t(t):
            return zero_comp_by_time_local[t]

        def get_Psi_raw_t(t):
            return Psi_raw_by_time_local[t]

        def time_worker(t):
            A_idx = sample_rows.get(t)
            if A_idx is None or len(A_idx) == 0:
                raise ValueError(f"[batch_run] Cannot find sample_id={t}.")
            if time_contrast == "vs_ref":
                B_id = ref_id_chr
            else:
                B_id = prev_id_for_t.get(t, ref_id_chr)
            B_idx = sample_rows.get(B_id)
            if B_idx is None or len(B_idx) == 0:
                raise ValueError(f"[batch_run] Cannot find B_id={B_id}.")
            xA = np.maximum(pd.to_numeric(combined.iloc[A_idx][gene], errors="coerce").to_numpy(dtype=float), 0)
            xB = np.maximum(pd.to_numeric(combined.iloc[B_idx][gene], errors="coerce").to_numpy(dtype=float), 0)
            tree_A = shared["trees_time"].get(t)
            if tree_A is None:
                raise ValueError(f"[batch_run] Cannot find the KNN tree for sample_id={t}.")
            tree_B = shared["tr_ref"] if B_id == ref_id_chr else shared["trees_time"].get(B_id)
            if tree_B is None:
                raise ValueError(f"[batch_run] Cannot find the KNN tree for B_id={B_id}.")
            k_loc_run = or_else(shared.get("k_loc"), shared.get("PARAMS", {}).get("k_loc"))
            hard_radius_run = bool(or_else(shared.get("hard_radius_loc"), or_else(shared.get("PARAMS", {}).get("hard_radius_loc"), True)))
            stA = local_stats_from_tree(tree=tree_A, x=xA, R=shared["R"], k_loc=k_loc_run, hard_radius=hard_radius_run)
            stB = local_stats_from_tree(tree=tree_B, x=xB, R=shared["R"], k_loc=k_loc_run, hard_radius=hard_radius_run)
            muA = np.asarray(stA["mu"], dtype=float)
            muB = np.asarray(stB["mu"], dtype=float)
            s2A = np.asarray(stA["s2"], dtype=float)
            s2B = np.asarray(stB["s2"], dtype=float)
            neA = np.asarray(stA["neff"], dtype=float)
            neB = np.asarray(stB["neff"], dtype=float)
            y = muA - muB
            v_base = s2A / np.maximum(neA, 1e-8) + s2B / np.maximum(neB, 1e-8)
            good_v = v_base[np.isfinite(v_base) & (v_base > 0)]
            med_v = float(np.nanmedian(good_v)) if good_v.size else 1.0
            if not np.isfinite(med_v) or med_v <= 0:
                med_v = 1.0
            v_base = np.maximum(v_base, float(or_else(shared.get("PARAMS", {}).get("stat_s2_floor_mult"), 0.10)) * med_v)
            Wv = 1 / np.maximum(v_base, 1e-12)
            ne_use = np.minimum(neA, neB)
            use = np.isfinite(y) & np.isfinite(Wv) & np.isfinite(ne_use) & (ne_use >= neff_min)
            risk_global_score = 0.0
            if risk_in_Wv and shared.get("risk_time", {}).get(t) is not None:
                rr_t = compute_r01_and_global_score(
                    risk_vec=shared["risk_time"][t],
                    use_mask=use,
                    weight_norm=risk_weight_norm,
                    cap_q=risk_cap_q,
                    global_rule=risk_global_rule,
                    global_thr=risk_global_thr,
                    global_cap=risk_global_cap,
                )
                infl_local = 1 + risk_local_lambda * (np.maximum(rr_t["r01"], 0) ** risk_power_fixed)
                # The promoted A=0 calibration is local-only. The comparison-level
                # The global score is retained for diagnostics but does not inflate variance.
                infl_global = 1.0
                v_risk = v_base * infl_local * infl_global
                v_floor_base = np.nanmedian(v_risk[use & np.isfinite(v_risk)]) if np.any(use & np.isfinite(v_risk)) else np.nan
                v_floor2 = max(risk_wv_min, risk_wv_floor_frac * v_floor_base) if np.isfinite(v_floor_base) else risk_wv_min
                if not np.isfinite(v_floor2) or v_floor2 <= 0:
                    v_floor2 = risk_wv_min
                Wv = 1 / np.maximum(v_risk, v_floor2)
                risk_global_score = rr_t["global_score"]
            if celltype_adjustment_use:
                Wv = Wv * celltype_precision_local[t]
            return {"y": y, "Wv": Wv, "use": use, "muA": muA, "muB": muB, "neff": ne_use, "risk_global_score": risk_global_score}

        time_res = chunked_apply(T_ids, time_worker, core=core, chunk_size=core)
        y_list = {t: res["y"] for t, res in zip(T_ids, time_res)}
        Wv_list = {t: res["Wv"] for t, res in zip(T_ids, time_res)}
        use_list = {t: res["use"] for t, res in zip(T_ids, time_res)}
        muA_list = {t: res["muA"] for t, res in zip(T_ids, time_res)}
        muB_list = {t: res["muB"] for t, res in zip(T_ids, time_res)}
        neff_list = {t: res["neff"] for t, res in zip(T_ids, time_res)}
        risk_global_score_by_time = {t: res["risk_global_score"] for t, res in zip(T_ids, time_res)}

        def orthonormalize_Psi(Psi, W, C):
            Psi = np.array(Psi, dtype=float, copy=True)
            if Psi.size == 0 or Psi.shape[1] == 0:
                return np.zeros((len(W), 0), dtype=float)
            w = np.sqrt(np.array(W, dtype=float, copy=True))
            w[~np.isfinite(w)] = 0
            sw = np.sum(w)
            if not np.isfinite(sw) or sw <= 0:
                sw = 1.0
            mu = np.sum(w[:, None] * Psi, axis=0) / sw
            P = Psi - mu[None, :]
            C = np.asarray(C, dtype=float)
            if C.size > 0 and C.shape[1] > 0:
                CW = C * w[:, None]
                PW = P * w[:, None]
                G = CW.T @ CW + np.eye(C.shape[1]) * 1e-8
                try:
                    coef = np.linalg.solve(G, CW.T @ PW)
                except np.linalg.LinAlgError:
                    coef = np.zeros((C.shape[1], P.shape[1]), dtype=float)
                P = P - C @ coef
            return np.asarray(P, dtype=float)

        def build_Psi_ortho_by_time(Wv_source):
            if not include_Psi:
                return {t: np.zeros((N, 0), dtype=float) for t in T_ids}
            return {t: orthonormalize_Psi(Psi_raw_by_time_local[t], Wv_source[t], zero_comp_by_time_local[t]) for t in T_ids}

        is_stable_by_time = {}
        for t in T_ids:
            y = y_list[t]
            W = Wv_list[t]
            use = use_list[t]
            t0 = np.full(len(y), np.nan, dtype=float)
            t0[use] = y[use] * np.sqrt(np.asarray(W[use], dtype=float))
            subset = np.abs(t0[use & np.isfinite(t0)])
            thr = float(np.nanquantile(subset, ctl_q)) if subset.size else np.nan
            is_stable_by_time[t] = (np.abs(t0) <= thr) & use & np.isfinite(t0)
        stab_mat = np.column_stack([is_stable_by_time[t] for t in T_ids]) if T_ids else np.zeros((N, 0), dtype=bool)
        stab_rate = np.nanmean(stab_mat.astype(float), axis=1) if stab_mat.size else np.zeros(N, dtype=float)
        S_ctl = stab_rate >= ctl_tau

        Psi_ortho_by_time = build_Psi_ortho_by_time(Wv_list)

        def fit_batch_prior_on_anchors():
            if not include_Psi:
                return {"alpha0": 0.0, "a0": np.zeros(0, dtype=float)}
            pB_target = max(int(kB_use), 0)
            n_int = 1 if include_intercept else 0
            d = n_int + pB_target
            if d == 0:
                return {"alpha0": 0.0, "a0": np.zeros(0, dtype=float)}
            partials = []
            for t in T_ids:
                use = np.asarray(use_list[t], dtype=bool) & S_ctl
                use[np.isnan(use.astype(float))] = False
                if not np.any(use):
                    continue
                Psi_t = np.asarray(Psi_ortho_by_time[t], dtype=float)
                if Psi_t.shape[1] > pB_target:
                    Psi_t = Psi_t[:, :pB_target]
                if pB_target > 0 and Psi_t.shape[1] < pB_target:
                    tmp = np.zeros((Psi_t.shape[0], pB_target), dtype=float)
                    if Psi_t.shape[1] > 0:
                        tmp[:, : Psi_t.shape[1]] = Psi_t
                    Psi_t = tmp
                y = np.asarray(y_list[t][use], dtype=float)
                w = np.asarray(Wv_list[t][use], dtype=float)
                ok = np.isfinite(y) & np.isfinite(w) & (w > 0)
                if int(np.sum(ok)) < d + 2:
                    continue
                y = y[ok]
                w = w[ok]
                X_parts = []
                if include_intercept:
                    X_parts.append(np.ones((len(y), 1), dtype=float))
                if pB_target > 0:
                    X_parts.append(Psi_t[use][ok])
                X = np.column_stack(X_parts) if X_parts else np.zeros((len(y), 0), dtype=float)
                sw = np.sqrt(w)
                Xw = X * sw[:, None]
                yw = y * sw
                partials.append({"XtWX": Xw.T @ Xw, "XtWy": Xw.T @ yw, "n_used": int(np.sum(ok))})
            if not partials:
                return {"alpha0": 0.0, "a0": np.zeros(pB_target, dtype=float)}
            XtWX = sum(item["XtWX"] for item in partials)
            XtWy = sum(item["XtWy"] for item in partials)
            n_used_total = sum(item["n_used"] for item in partials)
            if n_used_total < d + 2:
                return {"alpha0": 0.0, "a0": np.zeros(pB_target, dtype=float)}
            Pen = np.zeros((d, d), dtype=float)
            if pB_target > 0:
                idx_B_prior = list(range(1, d)) if include_intercept else list(range(d))
                for idx in idx_B_prior:
                    Pen[idx, idx] = lambda_B
            A = XtWX + Pen + np.eye(d) * 1e-8
            try:
                coef = np.linalg.solve(A, XtWy)
            except np.linalg.LinAlgError:
                coef = np.zeros(d, dtype=float)
            if include_intercept:
                return {"alpha0": float(coef[0]), "a0": np.asarray(coef[1:], dtype=float)}
            return {"alpha0": 0.0, "a0": np.asarray(coef, dtype=float)}

        batch_prior = fit_batch_prior_on_anchors() if prior_from_anchor and include_Psi else {"alpha0": 0.0, "a0": np.zeros(0, dtype=float)}

        def solve_with_blocks(y, Wv, use, C, Psi, lambda_B, delta0=None, alpha0=None, lambda_alpha=lambda_B):
            use = np.asarray(use, dtype=bool)
            idx_use = np.flatnonzero(use)
            Nall = len(y)
            t_full = np.full(Nall, np.nan, dtype=float)
            p_full = np.full(Nall, np.nan, dtype=float)
            q_full = np.full(Nall, np.nan, dtype=float)
            if idx_use.size == 0:
                return {"t": t_full, "p": p_full, "q": q_full, "df": 5.0, "idx_keep": np.array([], dtype=int), "beta": np.zeros(0), "delta": np.zeros(0), "gamma_hat": 0.0}
            y_ = np.asarray(y, dtype=float)[idx_use]
            W_ = np.array(Wv, dtype=float, copy=True)[idx_use]
            W_[~np.isfinite(W_)] = 0
            keep_row = W_ > 0
            idx_keep = idx_use[keep_row]
            if idx_keep.size < 5:
                return {"t": t_full, "p": p_full, "q": q_full, "df": 5.0, "idx_keep": idx_keep, "beta": np.zeros(0), "delta": np.zeros(0), "gamma_hat": 0.0}
            yk = y_[keep_row]
            Wk = W_[keep_row]
            C_ = np.asarray(C, dtype=float)[idx_use][keep_row] if np.asarray(C).size else np.zeros((len(yk), 0), dtype=float)
            Psi_ = np.asarray(Psi, dtype=float)[idx_use][keep_row] if np.asarray(Psi).size else np.zeros((len(yk), 0), dtype=float)
            pC = C_.shape[1] if C_.ndim == 2 else 0
            pB = Psi_.shape[1] if Psi_.ndim == 2 else 0
            alpha0_use = float(alpha0) if include_intercept and alpha0 is not None and np.isfinite(alpha0) else None
            lambda_alpha_use = float(lambda_alpha) if alpha0_use is not None and lambda_alpha is not None and np.isfinite(lambda_alpha) and lambda_alpha > 0 else 0.0
            if (pC + pB) == 0:
                if include_intercept:
                    sw = np.sum(Wk)
                    gm = float(np.sum(Wk * yk) / sw) if sw > 0 else 0.0
                    if lambda_alpha_use > 0:
                        gm = float((np.sum(Wk * yk) + lambda_alpha_use * alpha0_use) / (np.sum(Wk) + lambda_alpha_use))
                    r = yk - gm
                    df = float(max(len(r) - 1, 5))
                else:
                    gm = 0.0
                    r = yk
                    df = float(max(len(r), 5))
                tval = r * np.sqrt(Wk)
                pval = 2 * stats.t.cdf(-np.abs(tval), df=df)
                qval = p_adjust_bh(pval)
                t_full[idx_keep] = tval
                p_full[idx_keep] = pval
                q_full[idx_keep] = qval
                return {"t": t_full, "p": p_full, "q": q_full, "df": df, "idx_keep": idx_keep, "beta": np.zeros(0), "delta": np.zeros(0), "gamma_hat": gm}
            X_parts = []
            if include_intercept:
                X_parts.append(np.ones((len(yk), 1), dtype=float))
            if pC > 0:
                X_parts.append(C_)
            if pB > 0:
                X_parts.append(Psi_)
            X = np.column_stack(X_parts)
            d = X.shape[1]
            sqrtW = np.sqrt(Wk)
            Xw = X * sqrtW[:, None]
            yw = yk * sqrtW
            XtWX = Xw.T @ Xw
            XtWy = Xw.T @ yw
            Pen = np.zeros((d, d), dtype=float)
            n_int = 1 if include_intercept else 0
            idx_int = 0 if include_intercept else None
            idx_C = np.arange(n_int, n_int + pC)
            idx_B = np.arange(n_int + pC, n_int + pC + pB)
            if pB > 0:
                Pen[np.ix_(idx_B, idx_B)] = np.eye(pB) * lambda_B
                if delta0 is None or len(delta0) != pB:
                    delta0 = np.zeros(pB, dtype=float)
            if idx_int is not None and lambda_alpha_use > 0:
                Pen[idx_int, idx_int] += lambda_alpha_use
            rhs = XtWy.copy()
            if pB > 0:
                rhs[idx_B] += lambda_B * np.asarray(delta0, dtype=float)
            if idx_int is not None and lambda_alpha_use > 0:
                rhs[idx_int] += lambda_alpha_use * alpha0_use

            def robust_linear_solve(K, rhs):
                K = 0.5 * (K + K.T)
                dd = K.shape[0]
                diag_mean = float(np.mean(np.diag(K))) if dd > 0 else 1.0
                if not np.isfinite(diag_mean) or diag_mean <= 0:
                    diag_mean = 1.0
                for tau in (0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2):
                    Kt = K + np.eye(dd) * (tau * diag_mean)
                    try:
                        sol = np.linalg.solve(Kt, rhs)
                        if np.all(np.isfinite(sol)):
                            return np.asarray(sol, dtype=float)
                    except np.linalg.LinAlgError:
                        pass
                    try:
                        sol = np.linalg.lstsq(Kt, rhs, rcond=None)[0]
                        if np.all(np.isfinite(sol)):
                            return np.asarray(sol, dtype=float)
                    except np.linalg.LinAlgError:
                        pass
                raise np.linalg.LinAlgError("robust solver failed.")

            coef = robust_linear_solve(XtWX + Pen, rhs)
            yhat = X @ coef
            r = yk - yhat
            tval = r * np.sqrt(Wk)
            df = float(max(len(r) - d, 5))
            pval = 2 * stats.t.cdf(-np.abs(tval), df=df)
            qval = p_adjust_bh(pval)
            t_full[idx_keep] = tval
            p_full[idx_keep] = pval
            q_full[idx_keep] = qval
            return {
                "t": t_full,
                "p": p_full,
                "q": q_full,
                "df": df,
                "idx_keep": idx_keep,
                "beta": coef[idx_C] if pC > 0 else np.zeros(0),
                "delta": coef[idx_B] if pB > 0 else np.zeros(0),
                "gamma_hat": float(coef[idx_int]) if idx_int is not None else 0.0,
            }

        def res_worker(t):
            use_t = np.asarray(use_list[t], dtype=bool)
            Wv_t = Wv_list[t]
            y_t = y_list[t]
            C_t = get_C_t(t)
            Psi_t = Psi_ortho_by_time[t] if include_Psi else np.zeros((N, 0), dtype=float)
            pB = Psi_t.shape[1]
            if include_Psi and batch_prior.get("a0") is not None and len(batch_prior.get("a0", [])) > 0 and pB > 0:
                a0 = np.asarray(batch_prior["a0"], dtype=float)
                delta0_t = a0[:pB] if len(a0) >= pB else np.r_[a0, np.zeros(pB - len(a0), dtype=float)]
            else:
                delta0_t = np.zeros(pB, dtype=float)
            alpha0_t = float(batch_prior["alpha0"]) if include_intercept and include_Psi and np.isfinite(batch_prior.get("alpha0", np.nan)) else None
            return solve_with_blocks(
                y=y_t,
                Wv=Wv_t,
                use=use_t,
                C=C_t,
                Psi=Psi_t,
                lambda_B=lambda_B,
                delta0=delta0_t,
                alpha0=alpha0_t,
                lambda_alpha=lambda_B,
            )

        res_list_items = chunked_apply(T_ids, res_worker, core=core, chunk_size=core)
        res_list = {t: res for t, res in zip(T_ids, res_list_items)}
        t_raw_list = {t: res_list[t]["t"] for t in T_ids}
        p_raw_list = {t: res_list[t]["p"] for t in T_ids}
        q_raw_list = {t: res_list[t]["q"] for t in T_ids}
        sig_raw_list = {}
        for t in T_ids:
            q = np.asarray(q_raw_list[t], dtype=float)
            u = np.asarray(use_list[t], dtype=bool)
            out = (q <= alpha) & np.isfinite(q) & u
            sig_raw_list[t] = out.astype(bool)

        def adj_worker(t):
            res_t = res_list[t]
            C_t = get_C_t(t)
            Psi_t = Psi_ortho_by_time[t] if include_Psi else np.zeros((N, 0), dtype=float)
            fit_C = C_t @ np.asarray(res_t["beta"], dtype=float) if len(res_t["beta"]) > 0 else np.zeros(N, dtype=float)
            fit_P = Psi_t @ np.asarray(res_t["delta"], dtype=float) if len(res_t["delta"]) > 0 else np.zeros(N, dtype=float)
            nuisance = fit_C + fit_P
            return {"muA_adj": np.asarray(muA_list[t], dtype=float) - nuisance, "deltaA_adj": np.asarray(y_list[t], dtype=float) - nuisance}

        adj_res_items = [adj_worker(t) for t in T_ids]
        adj_res = {t: res for t, res in zip(T_ids, adj_res_items)}
        muA_adj_by_time = {t: adj_res[t]["muA_adj"] for t in T_ids}
        deltaA_adj_by_time = {t: adj_res[t]["deltaA_adj"] for t in T_ids}

        terrain_data = {
            "time_ids": T_ids,
            "ref_sample_id": ref_id_chr,
            "muA_by_time": muA_list,
            "muB_by_time": muB_list,
            "y_by_time": y_list,
            "neff_by_time": neff_list,
            "Wv_by_time": Wv_list,
            "use_by_time": use_list,
            "risk_global_score_by_time": risk_global_score_by_time,
            "celltype_adjustment": bool(celltype_adjustment_use),
            "celltype_precision_by_time": celltype_precision_local if celltype_adjustment_use else None,
            "celltype_js_divergence_by_time": celltype_js_divergence_local if celltype_adjustment_use else None,
            "celltype_js_distance_by_time": celltype_js_distance_local if celltype_adjustment_use else None,
            "celltype_variance_factor_by_time": celltype_variance_factor_local if celltype_adjustment_use else None,
            "celltype_adjustment_info": shared.get("celltype_adjustment_info"),
            "muA_adj_by_time": muA_adj_by_time,
            "beta_by_time": {t: res_list[t]["beta"] for t in T_ids},
            "delta_by_time": {t: res_list[t]["delta"] for t in T_ids},
            "gamma_by_time": {t: res_list[t]["gamma_hat"] for t in T_ids},
            "df_by_time": {t: res_list[t]["df"] for t in T_ids},
            "risk_calibration": risk_calibration,
            "auto_geometry": or_else(shared.get("auto_geometry"), shared.get("PARAMS", {}).get("auto_geometry_resolved")),
        }

        if traj_pretest and len(T_ids) >= 2:
            def make_rw2(Tn):
                Tn = int(Tn)
                if Tn < 3:
                    return np.zeros((0, max(0, Tn)), dtype=float)
                D2 = np.zeros((Tn - 2, Tn), dtype=float)
                for i in range(Tn - 2):
                    D2[i, i : i + 3] = np.array([1.0, -2.0, 1.0])
                return D2

            Tm = len(T_ids)
            D2 = make_rw2(Tm)
            PenT = D2.T @ D2
            Z_raw = np.column_stack([t_raw_list[t] for t in T_ids])
            U_raw = np.column_stack([use_list[t] for t in T_ids]).astype(bool)
            row_chunks = np.array_split(np.arange(N), max(1, min(N, core)))

            def smooth_chunk_worker(idx_rows):
                Z_chunk = np.full((len(idx_rows), Tm), np.nan, dtype=float)
                solve_cache = {}
                for rr, i in enumerate(idx_rows):
                    z_i = Z_raw[i, :].copy()
                    use_i = U_raw[i, :] & np.isfinite(z_i)
                    if int(np.sum(use_i)) < traj_min_obs:
                        Z_chunk[rr, :] = z_i
                    else:
                        idx = np.flatnonzero(use_i)
                        key = tuple(idx.tolist())
                        A = solve_cache.get(key)
                        if A is None:
                            Psub = PenT[np.ix_(idx, idx)]
                            A = np.eye(len(idx)) + traj_lambda * Psub + np.eye(len(idx)) * traj_eps
                            solve_cache[key] = A
                        z_obs = z_i[idx]
                        try:
                            zhat = np.linalg.solve(A, z_obs)
                        except np.linalg.LinAlgError:
                            zhat = np.linalg.lstsq(A, z_obs, rcond=None)[0]
                        Z_tmp = z_i.copy()
                        Z_tmp[idx] = zhat
                        Z_chunk[rr, :] = Z_tmp
                return Z_chunk

            Z_chunks = [smooth_chunk_worker(chunk) for chunk in row_chunks if len(chunk) > 0]
            Z_smooth = np.vstack(Z_chunks) if Z_chunks else np.zeros((N, Tm), dtype=float)

            traj_time_res = {}
            for j, t in enumerate(T_ids):
                z_j = Z_smooth[:, j]
                if traj_p_dist == "norm":
                    p_j = 2 * stats.norm.cdf(-np.abs(z_j))
                else:
                    p_j = 2 * stats.t.cdf(-np.abs(z_j), df=traj_df_eff)
                p_j[~np.isfinite(p_j)] = np.nan
                q_j = np.full(N, np.nan, dtype=float)
                ok = np.isfinite(p_j) & np.isfinite(z_j)
                if np.any(ok):
                    q_j[ok] = p_adjust_bh(p_j[ok])
                sig_j = ok & (q_j <= alpha)
                frac = None
                if screen_consensus and shared.get("tr_grid_posthoc") is not None:
                    er = grid_consensus_erosion(sig_j, shared["tr_grid_posthoc"], R_cons=shared["R"], tau=screen_tau, n_iter=screen_n_iter)
                    sig_j = np.asarray(er["sig"], dtype=bool)
                    frac = er["frac"]
                traj_time_res[t] = {"z": z_j, "p": p_j, "q": q_j, "sig": sig_j, "frac": frac}

            terrain_data["stat_by_time"] = {t: traj_time_res[t]["z"] for t in T_ids}
            terrain_data["p_by_time"] = {t: traj_time_res[t]["p"] for t in T_ids}
            terrain_data["q_by_time"] = {t: traj_time_res[t]["q"] for t in T_ids}
            terrain_data["sig_mask_by_time"] = {t: traj_time_res[t]["sig"] for t in T_ids}
            terrain_data["frac_support_by_time"] = {t: traj_time_res[t]["frac"] for t in T_ids}
        else:
            terrain_data["stat_by_time"] = t_raw_list if score_type == "t" else {t: np.sign(t_raw_list[t]) * (-np.log10(np.maximum(p_raw_list[t], 1e-300))) for t in T_ids}
            terrain_data["p_by_time"] = p_raw_list
            terrain_data["q_by_time"] = q_raw_list
            terrain_data["sig_mask_by_time"] = sig_raw_list

        if drawmask_cleanup:
            terrain_data["mask_cleanup_auto"] = auto_cln_cfg_global
            terrain_data["mask_cleanup_meta_by_time"] = {t: None for t in T_ids}
            terrain_data["sig_cc_summary_by_time"] = {t: None for t in T_ids}

        for t in T_ids:
            sig0 = np.asarray(terrain_data["sig_mask_by_time"][t], dtype=bool)
            frac_out = None
            if screen_consensus and terrain_data.get("frac_support_by_time", {}).get(t) is None and shared.get("tr_grid_posthoc") is not None:
                er = grid_consensus_erosion(sig0, shared["tr_grid_posthoc"], R_cons=shared["R"], tau=screen_tau, n_iter=screen_n_iter)
                sig0 = np.asarray(er["sig"], dtype=bool)
                frac_out = er["frac"]
            sig_final = sig0
            meta_out = None
            ccsum_out = None
            if drawmask_cleanup:
                cln_fast = cleanup_mask_fast(sig_final, shared["grid_eval"], gi_grid_eval, auto_cln_cfg_global, bbox_pad=cleanup_bbox_pad, small_mask_n=cleanup_small_mask_n)
                sig_final = cln_fast["mask"]
                meta_out = cln_fast["meta"]
                ccsum_out = cln_fast["meta"]["cc_sizes"]
            terrain_data["sig_mask_by_time"][t] = sig_final
            if frac_out is not None:
                terrain_data.setdefault("frac_support_by_time", {})[t] = frac_out
            if drawmask_cleanup:
                terrain_data["mask_cleanup_meta_by_time"][t] = meta_out
                terrain_data["sig_cc_summary_by_time"][t] = ccsum_out

        if STORE_RAW_OUTPUT:
            terrain_data["t_raw_by_time"] = t_raw_list
            terrain_data["p_raw_by_time"] = p_raw_list
            terrain_data["q_raw_by_time"] = q_raw_list
            terrain_data["sig_raw_by_time"] = sig_raw_list

        if not _skip_risk_autocalib:
            # Local P values support the spatial ACAT summary. Adjusted local
            # expression, precision and use masks are also retained because
            # the age-trend ACAT fits unsmoothed weighted slopes across
            # contrasts. Larger raw moment arrays can still be discarded.
            for nm in ["muA_by_time", "muB_by_time", "y_by_time", "neff_by_time"]:
                if nm in terrain_data:
                    terrain_data[nm] = None
            terrain_data = compact_terrain_data_for_return(terrain_data)

        return {"gene": gene, "terrain_data": terrain_data}

    return _run_impl(
        risk_in_Wv=risk_in_Wv,
        traj_pretest=traj_pretest,
        screen_consensus=screen_consensus,
        drawmask_cleanup=drawmask_cleanup,
        verbose=verbose,
        core=core,
        celltype_adjustment_inner=celltype_adjustment,
    )


import math
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def run_global_spatial_roi_trajectory_clustering(
    fit,
    shared,
    K_TRAJ=None,
    auto_k=None,
    time_values=None,
    do_plot=True,
    seed=None,
    core=1,
    spatial_weight=0.20,
    curvature_weight=0.08,
    use_stat_reliability=True,
    dtype=np.float32,
    keep_features=False,
):
    """
    Global spatial-ROI trajectory clustering.

    Global spatial-ROI trajectory clustering using complete adjusted-expression
    trajectories as the primary features. Spatial coordinates remain weak
    tie-breakers and spatial label refinement preserves coherent regions.

    When K_TRAJ is omitted, auto-K is selected in two stages without a
    hand-weighted composite score: (1) held-out-time evidence for
    cluster-specific trajectories, then (2) a fine-to-coarse scan within the
    statistically indistinguishable K plateau. Starting from the finest
    candidate, the scan coarsens while R_map-scale fragmentation decreases and
    stops at the first fine-side local minimum. If fragmentation keeps
    decreasing throughout the plateau and therefore supplies no elbow, the
    scan takes one conservative coarsening step from the finest candidate.
    """
    _ = max(int(core), 1)  # API compatibility; the implementation is vectorized.

    cfg_default = {
        "K_GRID": list(range(2, 10)),
        # Candidate K values share the feature matrix and spatial graph.
        # Full-grid candidate fits are the default because auto-K is a
        # model-selection decision. Set AUTO_SUBSAMPLE_N explicitly to use
        # a faster approximate screening fit on very large datasets.
        "AUTO_SUBSAMPLE_N": 0,
        "SPATIAL_GRAPH_K": 10,
        "GRAPH_BANDWIDTH_SCALE": 1.0,
        "BS_DF": 8,
        "BS_DEGREE": 3,
        "BS_RIDGE": 1e-2,
        "FEATURE_SMOOTH_ALPHA": 0.55,
        "FEATURE_SMOOTH_ITERS": 5,
        "KMEANS_NSTART_AUTO": 3,
        "KMEANS_NSTART_FULL": 6,
        "KMEANS_ITER_AUTO": 80,
        "KMEANS_ITER_FULL": 100,
        "AUTOK_REFINE_MAXITER": 8,
        "FINAL_LABEL_BETA": 2.8,
        "FINAL_LABEL_MAXITER": 10,
        "AUTO_K_MAX_TIME_FOLDS": 5,
        "SHOW_AUTO_K_DIAGNOSTIC": False,
        "ROWREL_CLIP": (0.60, 1.80),
        "RANDOM_STATE_FALLBACK": 1,
        "DIST_CHUNK": 50000,
    }
    cfg = dict(cfg_default)
    if auto_k is not None:
        cfg.update(dict(auto_k))

    def _or_else(a, b):
        return b if a is None else a

    def _safe_seed(seed_in):
        try:
            return int(seed_in) if seed_in is not None else int(cfg.get("RANDOM_STATE_FALLBACK", 1))
        except Exception:
            return int(cfg.get("RANDOM_STATE_FALLBACK", 1))

    rng = np.random.default_rng(_safe_seed(seed))

    def _is_single_fit(x):
        return (
            isinstance(x, dict)
            and isinstance(x.get("terrain_data"), dict)
            and isinstance(x["terrain_data"].get("muA_adj_by_time"), dict)
            and len(x["terrain_data"].get("muA_adj_by_time")) > 0
        )

    if not _is_single_fit(fit):
        raise ValueError("fit must be one fit dict with terrain_data['muA_adj_by_time'].")

    td = fit["terrain_data"]
    gene_name = str(_or_else(fit.get("gene"), "fit1"))
    mu_by_time = td["muA_adj_by_time"]
    ref_sample_id = _or_else(td.get("ref_sample_id"), shared.get("ref_sample_id"))

    def _infer_time_numeric(time_ids):
        out = []
        for t in time_ids:
            try:
                out.append(float(t))
                continue
            except Exception:
                pass
            m = re.search(r"(\d+(?:\.\d+)?)$", str(t))
            out.append(float(m.group(1)) if m is not None else np.nan)
        return np.asarray(out, dtype=float)

    def _ordered_time_ids(dct):
        tids = list(dct.keys())
        nums = _infer_time_numeric(tids)
        if np.any(np.isfinite(nums)):
            order = np.argsort(np.where(np.isfinite(nums), nums, np.inf))
            tids = [tids[i] for i in order.tolist()]
        return tids

    time_ids_input = list(mu_by_time.keys())
    time_numeric_override = None
    if time_values is None:
        time_ids_all = _ordered_time_ids(mu_by_time)
    else:
        time_numeric_input = np.asarray(time_values, dtype=float).reshape(-1)
        if time_numeric_input.size != len(time_ids_input):
            raise ValueError(
                "time_values must have one value for each trajectory time ID "
                "in insertion order."
            )
        if not np.all(np.isfinite(time_numeric_input)):
            raise ValueError("time_values must contain only finite numeric values.")
        order = np.argsort(time_numeric_input, kind="stable")
        time_ids_all = [time_ids_input[index] for index in order.tolist()]
        time_numeric_override = time_numeric_input[order]
    if len(time_ids_all) < 2:
        raise ValueError("Need at least two time points in muA_adj_by_time.")

    time_numeric_lookup = (
        {str(time_id): float(value) for time_id, value in zip(time_ids_all, time_numeric_override)}
        if time_numeric_override is not None
        else None
    )

    def _time_numeric_for_ids(time_ids):
        if time_numeric_lookup is None:
            return _infer_time_numeric(time_ids)
        return np.asarray([time_numeric_lookup[str(time_id)] for time_id in time_ids], dtype=float)

    def _resolved_time_numeric_for_ids(time_ids):
        values = _time_numeric_for_ids(time_ids)
        fallback = np.arange(len(time_ids), dtype=float)
        if not np.any(np.isfinite(values)):
            return fallback
        return np.where(np.isfinite(values), values, fallback)

    nonref = [t for t in time_ids_all if str(t) != str(ref_sample_id)]
    if nonref:
        nums_nonref = _time_numeric_for_ids(nonref)
        if np.any(np.isfinite(nums_nonref)):
            time_id_plot_map = nonref[int(np.argmin(np.where(np.isfinite(nums_nonref), nums_nonref, np.inf)))]
        else:
            time_id_plot_map = nonref[0]
    else:
        time_id_plot_map = time_ids_all[0]

    def _as_1d(v, dtype_local=None):
        arr = np.asarray(v).reshape(-1)
        if dtype_local is not None:
            arr = arr.astype(dtype_local, copy=False)
        return arr

    def _stack_time_dict(dct, time_ids, dtype_local=np.float32):
        if not isinstance(dct, dict):
            raise ValueError("Expected a dictionary keyed by time_id.")
        v0 = _as_1d(dct[time_ids[0]], None)
        X = np.empty((v0.shape[0], len(time_ids)), dtype=(dtype_local or float))
        for j, t in enumerate(time_ids):
            if t not in dct:
                raise ValueError(f"Missing time {t!r}.")
            v = _as_1d(dct[t], None)
            if v.shape[0] != v0.shape[0]:
                raise ValueError(f"Length mismatch at time {t!r}.")
            X[:, j] = v.astype(X.dtype, copy=False)
        X[~np.isfinite(X)] = np.nan
        return X

    def _fill_nan_by_row_then_col(X):
        X = np.asarray(X, dtype=np.float32).copy()
        if not np.isnan(X).any():
            return X
        finite = np.isfinite(X)
        row_count = finite.sum(axis=1)
        col_count = finite.sum(axis=0)
        row_mean = np.divide(
            np.nansum(X, axis=1), row_count,
            out=np.full(X.shape[0], np.nan, dtype=np.float32), where=row_count > 0,
        )
        col_mean = np.divide(
            np.nansum(X, axis=0), col_count,
            out=np.full(X.shape[1], np.nan, dtype=np.float32), where=col_count > 0,
        )
        total_count = int(finite.sum())
        global_mean = float(np.nansum(X) / total_count) if total_count else np.nan
        if not np.isfinite(global_mean):
            global_mean = 0.0
        row_mean[~np.isfinite(row_mean)] = global_mean
        col_mean[~np.isfinite(col_mean)] = global_mean
        ii, jj = np.where(~np.isfinite(X))
        X[ii, jj] = 0.5 * row_mean[ii] + 0.5 * col_mean[jj]
        X[~np.isfinite(X)] = global_mean
        return X.astype(np.float32, copy=False)

    def _scale_cols(X, eps=1e-8):
        X = np.asarray(X, dtype=np.float32)
        finite = np.isfinite(X)
        count = finite.sum(axis=0, keepdims=True)
        mu = np.divide(
            np.nansum(X, axis=0, keepdims=True),
            count,
            out=np.zeros((1, X.shape[1]), dtype=np.float32),
            where=count > 0,
        )
        centered = np.where(finite, X - mu, 0.0)
        denominator = np.maximum(count - 1, 1)
        variance = np.divide(
            np.sum(centered * centered, axis=0, keepdims=True),
            denominator,
            out=np.ones((1, X.shape[1]), dtype=np.float32),
            where=count > 1,
        )
        sd = np.sqrt(variance)
        mu[~np.isfinite(mu)] = 0.0
        sd[~np.isfinite(sd) | (sd < eps)] = 1.0
        Z = (X - mu) / sd
        Z[~np.isfinite(Z)] = 0.0
        return Z.astype(np.float32, copy=False)

    def _robust_scale_cols(X, q_lo=0.05, q_hi=0.95, eps=1e-8):
        X = np.asarray(X, dtype=np.float32)
        out = np.zeros_like(X, dtype=np.float32)
        for j in range(X.shape[1]):
            v = X[:, j].astype(float, copy=False)
            ok = np.isfinite(v)
            if not np.any(ok):
                continue
            lo = np.nanquantile(v[ok], q_lo)
            hi = np.nanquantile(v[ok], q_hi)
            center = np.nanmedian(v[ok])
            scale = 0.5 * (hi - lo)
            if not np.isfinite(scale) or scale < eps:
                scale = np.nanstd(v[ok], ddof=1)
            if not np.isfinite(scale) or scale < eps:
                continue
            out[ok, j] = np.clip((v[ok] - center) / scale, -4.0, 4.0).astype(np.float32, copy=False)
        return out

    def _robust_01(x, q_lo=0.05, q_hi=0.95, eps=1e-8):
        x = np.asarray(x, dtype=float).reshape(-1)
        out = np.zeros_like(x, dtype=np.float32)
        ok = np.isfinite(x)
        if not np.any(ok):
            return out
        lo = np.nanquantile(x[ok], q_lo)
        hi = np.nanquantile(x[ok], q_hi)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < eps:
            return out
        out[ok] = np.clip((x[ok] - lo) / (hi - lo), 0.0, 1.0)
        return out

    def _normalize_positive_weights(w, clip=(0.60, 1.80), eps=1e-8):
        w = np.asarray(w, dtype=float).reshape(-1)
        w[~np.isfinite(w)] = 0.0
        pos = w[w > 0]
        if pos.size == 0:
            return np.ones_like(w, dtype=np.float32)
        med = np.nanmedian(pos)
        if not np.isfinite(med) or med < eps:
            med = 1.0
        out = np.clip(w / med, float(clip[0]), float(clip[1]))
        out[~np.isfinite(out)] = 1.0
        return out.astype(np.float32, copy=False)

    ge0 = shared["grid_eval"]
    if not isinstance(ge0, pd.DataFrame) or not {"x", "y"}.issubset(ge0.columns):
        raise ValueError("shared['grid_eval'] must be a DataFrame with x/y columns.")
    grid_xy = ge0.loc[:, ["x", "y"]].to_numpy(dtype=float)
    if grid_xy.shape[0] < 2:
        raise ValueError("Need at least two grid points.")

    mu_raw = _fill_nan_by_row_then_col(_stack_time_dict(mu_by_time, time_ids_all, dtype))
    n_grid, n_time = mu_raw.shape
    if n_grid != grid_xy.shape[0]:
        raise ValueError(f"Grid length mismatch: muA has {n_grid} rows, grid_eval has {grid_xy.shape[0]} rows.")

    def _build_bspline_basis(time_ids, df=8, degree=3):
        tnum = _time_numeric_for_ids(time_ids)
        if np.all(~np.isfinite(tnum)):
            x = np.linspace(0.0, 1.0, len(time_ids))
        else:
            finite = np.isfinite(tnum)
            fill = np.arange(len(time_ids), dtype=float)
            tnum = np.where(finite, tnum, fill)
            lo, hi = float(np.min(tnum)), float(np.max(tnum))
            x = (tnum - lo) / max(hi - lo, 1e-8)
        n = len(x)
        degree = int(min(max(0, degree), max(0, n - 1)))
        n_basis = int(min(max(degree + 1, df), n))
        if n_basis <= degree + 1 or degree == 0:
            B = np.vander(x, N=n_basis, increasing=True)
            return _scale_cols(B).astype(np.float64, copy=False)
        knots = np.r_[np.zeros(degree), np.linspace(0.0, 1.0, n_basis - degree + 1), np.ones(degree)]
        B = np.zeros((n, n_basis), dtype=float)
        for j in range(n_basis):
            c = np.zeros(n_basis, dtype=float)
            c[j] = 1.0
            B[:, j] = BSpline(knots, c, degree, extrapolate=True)(x)
        B[~np.isfinite(B)] = 0.0
        return B

    def _smooth_time_bspline(X):
        B = _build_bspline_basis(time_ids_all, df=int(cfg.get("BS_DF", 8)), degree=int(cfg.get("BS_DEGREE", 3)))
        ridge = float(cfg.get("BS_RIDGE", 1e-2))
        BtB = B.T @ B + ridge * np.eye(B.shape[1], dtype=float)
        R = np.linalg.solve(BtB, B.T)  # basis x time
        coef = np.asarray(X, dtype=np.float32) @ R.T.astype(np.float32)
        X_sm = coef @ B.T.astype(np.float32)
        X_sm[~np.isfinite(X_sm)] = 0.0
        return X_sm.astype(np.float32, copy=False), B

    mu_sm, basis = _smooth_time_bspline(mu_raw)

    def _build_graph(XY, k=10, bandwidth_scale=1.0):
        XY = np.asarray(XY, dtype=float)
        n = XY.shape[0]
        k = min(max(1, int(k)), n - 1)
        dist, ind = cKDTree(XY).query(XY, k=k + 1)
        dist = np.asarray(dist, dtype=float)[:, 1:]
        ind = np.asarray(ind, dtype=int)[:, 1:]
        vals = dist[np.isfinite(dist) & (dist > 0)]
        h = float(np.nanmedian(vals)) if vals.size else 1.0
        if not np.isfinite(h) or h <= 0:
            h = 1.0
        h *= max(float(bandwidth_scale), 1e-6)
        row = np.repeat(np.arange(n), k)
        col = ind.reshape(-1)
        dd = dist.reshape(-1)
        ok = (col >= 0) & np.isfinite(dd)
        ww = np.exp(-0.5 * (dd[ok] / h) ** 2)
        W = sparse.coo_matrix((ww, (row[ok], col[ok])), shape=(n, n)).tocsr()
        W = ((W + W.T) * 0.5).tocsr()
        deg = np.asarray(W.sum(axis=1)).reshape(-1)
        inv_deg = np.zeros_like(deg, dtype=float)
        good = deg > 0
        inv_deg[good] = 1.0 / deg[good]
        P = sparse.diags(inv_deg).dot(W).tocsr()
        A = W.copy()
        A.data[:] = 1.0
        A = A.maximum(A.T).tocsr()
        return {"W": W, "P": P, "A": A, "degree": deg}

    graph = _build_graph(
        grid_xy,
        k=int(cfg.get("SPATIAL_GRAPH_K", 10)),
        bandwidth_scale=float(cfg.get("GRAPH_BANDWIDTH_SCALE", 1.0)),
    )

    def _smooth_by_neighbors(X, alpha=0.55, n_iter=5):
        X = np.asarray(X, dtype=np.float32)
        alpha = float(alpha)
        n_iter = int(max(n_iter, 0))
        if alpha <= 0 or n_iter <= 0:
            return X.astype(np.float32, copy=False)
        Z = X.astype(np.float32, copy=True)
        P = graph["P"]
        for _ in range(n_iter):
            Z = ((1.0 - alpha) * Z + alpha * (P @ Z)).astype(np.float32, copy=False)
            Z[~np.isfinite(Z)] = 0.0
        return Z

    traj_block = _scale_cols(mu_sm)
    feature_blocks = [{"block": "bspline_muA", "weight": 1.0, "n_features": int(traj_block.shape[1])}]
    parts = [traj_block]

    if n_time >= 3 and float(curvature_weight) > 0:
        curve = mu_sm[:, 2:] - 2.0 * mu_sm[:, 1:-1] + mu_sm[:, :-2]
        curve[~np.isfinite(curve)] = 0.0
        curve_block = _scale_cols(curve) * float(curvature_weight)
        parts.append(curve_block.astype(np.float32, copy=False))
        feature_blocks.append({"block": "bspline_curvature", "weight": float(curvature_weight), "n_features": int(curve_block.shape[1])})

    if float(spatial_weight) > 0:
        xy_block = _robust_scale_cols(grid_xy, 0.02, 0.98) * float(spatial_weight)
        parts.append(xy_block.astype(np.float32, copy=False))
        feature_blocks.append({"block": "spatial_xy", "weight": float(spatial_weight), "n_features": 2})

    X_feat = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
    X_feat = _smooth_by_neighbors(
        X_feat,
        alpha=float(cfg.get("FEATURE_SMOOTH_ALPHA", 0.55)),
        n_iter=int(cfg.get("FEATURE_SMOOTH_ITERS", 5)),
    )

    def _has_all_times(dct):
        return isinstance(dct, dict) and all(t in dct for t in time_ids_all)

    if use_stat_reliability and _has_all_times(td.get("stat_by_time")):
        stat_mat = _stack_time_dict(td["stat_by_time"], time_ids_all, np.float32)
        stat_mat[~np.isfinite(stat_mat)] = 0.0
        stat_strength = np.mean(np.abs(stat_mat), axis=1)
        stat_strength = _smooth_by_neighbors(stat_strength[:, None].astype(np.float32), alpha=0.55, n_iter=2).reshape(-1)
        stat01 = _robust_01(stat_strength, 0.10, 0.95)
        row_rel_raw = 0.75 + 0.25 * stat01
        if _has_all_times(td.get("sig_mask_by_time")):
            sig_mat = _stack_time_dict(td["sig_mask_by_time"], time_ids_all, np.float32)
            sig_frac = np.mean(sig_mat > 0, axis=1)
            sig_frac = _smooth_by_neighbors(sig_frac[:, None].astype(np.float32), alpha=0.55, n_iter=2).reshape(-1)
            sig01 = _robust_01(sig_frac, 0.00, 0.95)
            row_rel_raw = row_rel_raw * (0.85 + 0.15 * sig01)
        row_rel = _normalize_positive_weights(row_rel_raw, tuple(cfg.get("ROWREL_CLIP", (0.60, 1.80))))
    else:
        row_rel = np.ones(n_grid, dtype=np.float32)

    def _dist2_to_centers(X, centers, chunk=None):
        X = np.asarray(X, dtype=np.float32)
        centers = np.asarray(centers, dtype=np.float32)
        n = X.shape[0]
        K = centers.shape[0]
        chunk = int(_or_else(chunk, cfg.get("DIST_CHUNK", 50000)))
        labels = np.empty(n, dtype=np.int32)
        min_d2 = np.empty(n, dtype=np.float64)
        c_norm = np.sum(centers.astype(np.float64) ** 2, axis=1)
        C = centers.T.astype(np.float64, copy=False)
        for a in range(0, n, chunk):
            b = min(a + chunk, n)
            Xb = X[a:b].astype(np.float64, copy=False)
            d2 = np.sum(Xb ** 2, axis=1, keepdims=True) + c_norm[None, :] - 2.0 * (Xb @ C)
            d2 = np.maximum(d2, 0.0)
            labels[a:b] = np.argmin(d2, axis=1).astype(np.int32) + 1
            min_d2[a:b] = np.min(d2, axis=1)
        return labels, min_d2

    def _recompute_centers(X, labels, weights, K):
        X = np.asarray(X, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int32).reshape(-1)
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        centers = np.zeros((K, X.shape[1]), dtype=np.float64)
        for k in range(1, K + 1):
            ii = labels == k
            sw = float(np.sum(weights[ii]))
            if sw <= 0:
                centers[k - 1] = np.nan
            else:
                centers[k - 1] = np.sum(X[ii].astype(np.float64) * weights[ii, None], axis=0) / sw
        bad = ~np.isfinite(centers).all(axis=1)
        if np.any(bad):
            good = ~bad
            if np.any(good):
                centers[bad] = np.nanmean(centers[good], axis=0)
            else:
                centers[:] = np.nanmean(X.astype(np.float64), axis=0)
        centers[~np.isfinite(centers)] = 0.0
        return centers.astype(np.float32, copy=False)

    def _init_kmeans_pp(X, weights, K, rng_local):
        n = X.shape[0]
        weights = np.asarray(weights, dtype=float).reshape(-1)
        p = np.maximum(weights, 0.0)
        if np.sum(p) <= 0:
            p = np.ones(n, dtype=float)
        p = p / np.sum(p)
        centers = np.empty((K, X.shape[1]), dtype=np.float32)
        first = int(rng_local.choice(n, p=p))
        centers[0] = X[first]
        _, min_d2 = _dist2_to_centers(X, centers[:1])
        for k in range(1, K):
            pp = p * np.maximum(min_d2, 1e-12)
            s = float(np.sum(pp))
            idx = int(rng_local.choice(n, p=p if (s <= 0 or not np.isfinite(s)) else pp / s))
            centers[k] = X[idx]
            _, d2_new = _dist2_to_centers(X, centers[k:k + 1])
            min_d2 = np.minimum(min_d2, d2_new)
        return centers

    def _weighted_kmeans(X, weights, K, seed_local, nstart=4, itermax=80, init_centers=None):
        X = np.asarray(X, dtype=np.float32)
        weights = np.asarray(weights, dtype=np.float32).reshape(-1)
        best = None
        for s in range(max(int(nstart), 1)):
            rng_local = np.random.default_rng(int(seed_local) + 7919 * s)
            if s == 0 and init_centers is not None and np.asarray(init_centers).shape == (K, X.shape[1]):
                centers = np.asarray(init_centers, dtype=np.float32).copy()
            else:
                centers = _init_kmeans_pp(X, weights, K, rng_local)
            labels = None
            min_d2 = None
            for _iter in range(int(itermax)):
                new_labels, min_d2 = _dist2_to_centers(X, centers)
                if labels is not None and np.array_equal(new_labels, labels):
                    labels = new_labels
                    break
                labels = new_labels
                centers_new = _recompute_centers(X, labels, weights, K)
                if np.max(np.abs(centers_new - centers)) < 1e-5:
                    centers = centers_new
                    break
                centers = centers_new
            labels, min_d2 = _dist2_to_centers(X, centers)
            wss = float(np.sum(weights.astype(float) * min_d2))
            if best is None or wss < best["tot_withinss"]:
                size_w = np.array([np.sum(weights[labels == k]) for k in range(1, K + 1)], dtype=float)
                best = {"cluster": labels, "centers": centers, "min_d2": min_d2, "tot_withinss": wss, "weight_size": size_w}
        return best

    def _spatial_refine_labels(X, labels, weights, K, beta=2.8, max_iter=10):
        labels = np.asarray(labels, dtype=np.int32).reshape(-1).copy()
        beta = float(beta)
        max_iter = int(max(max_iter, 0))
        if beta <= 0 or max_iter <= 0:
            centers = _recompute_centers(X, labels, weights, K)
            labels, min_d2 = _dist2_to_centers(X, centers)
            return labels, centers, min_d2
        P = graph["P"]
        centers = _recompute_centers(X, labels, weights, K)
        X64 = X.astype(np.float64, copy=False)
        x_norm = np.sum(X64 ** 2, axis=1, keepdims=True)
        for _iter in range(max_iter):
            C64 = centers.astype(np.float64, copy=False)
            c_norm = np.sum(C64 ** 2, axis=1)
            d2 = x_norm + c_norm[None, :] - 2.0 * (X64 @ C64.T)
            d2 = np.maximum(d2, 0.0)
            same = np.empty((X.shape[0], K), dtype=np.float32)
            for k in range(1, K + 1):
                same[:, k - 1] = P @ (labels == k).astype(np.float32)
            obj = d2 + beta * (1.0 - same.astype(np.float64))
            new_labels = np.argmin(obj, axis=1).astype(np.int32) + 1
            if np.array_equal(new_labels, labels):
                labels = new_labels
                break
            labels = new_labels
            centers = _recompute_centers(X, labels, weights, K)
        centers = _recompute_centers(X, labels, weights, K)
        labels, min_d2 = _dist2_to_centers(X, centers)
        return labels, centers, min_d2

    def _pairwise_centroid_dist(centers):
        centers = np.asarray(centers, dtype=float)
        vals = []
        for a in range(centers.shape[0]):
            for b in range(a + 1, centers.shape[0]):
                vals.append(float(np.linalg.norm(centers[a] - centers[b])))
        return np.asarray(vals, dtype=float)

    def _component_metrics(labels, K, weights):
        """Connected-component diagnostics retained for the final partition."""
        labels = np.asarray(labels, dtype=np.int32).reshape(-1)
        weights = np.asarray(weights, dtype=float).reshape(-1)
        A = graph["A"]
        total_w = max(float(np.sum(weights)), 1e-8)
        rows = []
        fragment_mass = 0.0
        component_excess = 0.0
        for k in range(1, K + 1):
            idx_k = np.flatnonzero(labels == k)
            if idx_k.size == 0:
                rows.append({"cluster": k, "n_components": 0, "main_component_prop": 0.0, "fragment_mass": 0.0})
                continue
            if idx_k.size == 1:
                comp_w = np.array([weights[idx_k[0]]], dtype=float)
            else:
                sub = A[idx_k, :][:, idx_k]
                ncomp, comp = connected_components(sub, directed=False, return_labels=True)
                comp_w = np.bincount(comp, weights=weights[idx_k], minlength=ncomp).astype(float)
            comp_w = np.sort(comp_w)[::-1]
            mass_k = max(float(np.sum(comp_w)), 1e-8)
            fragment_mass += float(np.sum(comp_w[1:]))
            component_excess += max(int(comp_w.size) - 1, 0)
            rows.append({
                "cluster": k,
                "n_components": int(comp_w.size),
                "main_component_prop": float(comp_w[0] / mass_k),
                "fragment_mass": float(np.sum(comp_w[1:]) / mass_k) if comp_w.size > 1 else 0.0,
            })
        return {
            "fragment_mass": float(np.clip(fragment_mass / total_w, 0.0, 1.0)),
            "mean_component_excess": float(component_excess / max(K, 1)),
            "component_table": pd.DataFrame(rows),
        }

    def _sample_indices(n, target_n, weights):
        target_n = int(target_n)
        if target_n <= 0 or target_n >= n:
            return np.arange(n, dtype=int)
        weights = np.asarray(weights, dtype=float).reshape(-1)
        p = np.maximum(weights, 0.0)
        p = None if np.sum(p) <= 0 else p / np.sum(p)
        return np.sort(rng.choice(n, size=target_n, replace=False, p=p).astype(int))

    def _cluster_time_means(labels):
        """Weighted means of the complete mu_i(t) trajectory for a partition."""
        labels = np.asarray(labels, dtype=np.int32).reshape(-1)
        K = int(np.max(labels))
        means = np.full((K, n_time), np.nan, dtype=float)
        masses = np.zeros((K, n_time), dtype=float)
        for j in range(n_time):
            # Dynamic validation uses the observed trajectory at this age.
            # Missing values are excluded here; imputation above is only for
            # constructing the clustering feature matrix.
            y = _as_1d(mu_by_time[time_ids_all[j]], float)
            good = np.isfinite(y) & np.isfinite(row_rel) & (row_rel > 0)
            if not np.any(good):
                continue
            lab = labels[good] - 1
            w = row_rel[good].astype(float, copy=False)
            mass = np.bincount(lab, weights=w, minlength=K).astype(float)
            total = np.bincount(lab, weights=w * y[good], minlength=K).astype(float)
            ok = mass > 0
            means[ok, j] = total[ok] / mass[ok]
            masses[:, j] = mass
        return means, masses

    def _choose_time_cv_design():
        # The trajectory comparison is only meaningful when enough distinct
        # ages remain for an out-of-sample curve. For T=4 or 5, use the
        # simplest linear curve and leave one age out at a time.
        if n_time <= 3:
            return None
        if n_time <= 5:
            return 1, n_time
        max_folds = min(max(2, int(cfg.get("AUTO_K_MAX_TIME_FOLDS", 5))), n_time // 2)
        # A cluster-specific degree-d curve has d+1 coefficients per
        # cluster. Require at least two training ages per coefficient.
        for degree in (3, 2, 1):
            min_train = 2 * (degree + 1)
            feasible = [
                folds for folds in range(2, max_folds + 1)
                if n_time - int(np.ceil(n_time / folds)) >= min_train
            ]
            if feasible:
                return degree, max(feasible)
        return None

    def _time_basis_for_cv(degree):
        raw = _time_numeric_for_ids(time_ids_all)
        if not np.any(np.isfinite(raw)):
            raw = np.arange(n_time, dtype=float)
        else:
            raw = np.where(np.isfinite(raw), raw, np.arange(n_time, dtype=float))
        x = (raw - float(np.min(raw))) / max(float(np.max(raw) - np.min(raw)), 1e-8)
        degree = min(max(int(degree), 1), max(n_time - 2, 1))
        return np.column_stack([x ** power for power in range(1, degree + 1)])

    def _weighted_lstsq_predict(X_train, y_train, w_train, X_pred):
        good = np.isfinite(y_train) & np.isfinite(w_train) & (w_train > 0)
        if int(np.sum(good)) < X_train.shape[1]:
            return np.full(X_pred.shape[0], np.nan, dtype=float)
        root_w = np.sqrt(w_train[good])
        coef, *_ = np.linalg.lstsq(X_train[good] * root_w[:, None], y_train[good] * root_w, rcond=None)
        return X_pred @ coef

    def _dynamic_fold_loss(cluster_means, cluster_masses, basis_cv, train_idx, test_idx):
        """Held-out loss for shared versus cluster-specific age trajectories."""
        n_cluster = cluster_means.shape[0]
        n_basis = basis_cv.shape[1]

        def _design(indices, interaction):
            n_rows = n_cluster * len(indices)
            n_cols = n_cluster + (n_cluster * n_basis if interaction else n_basis)
            D = np.zeros((n_rows, n_cols), dtype=float)
            row = 0
            for c in range(n_cluster):
                for t in indices:
                    D[row, c] = 1.0
                    if interaction:
                        D[row, n_cluster + c * n_basis:n_cluster + (c + 1) * n_basis] = basis_cv[t]
                    else:
                        D[row, n_cluster:] = basis_cv[t]
                    row += 1
            return D

        y_train = cluster_means[:, train_idx].reshape(-1)
        w_train = cluster_masses[:, train_idx].reshape(-1)
        y_test = cluster_means[:, test_idx].reshape(-1)
        w_test = cluster_masses[:, test_idx].reshape(-1)
        pred_shared = _weighted_lstsq_predict(_design(train_idx, False), y_train, w_train, _design(test_idx, False))
        pred_cluster = _weighted_lstsq_predict(_design(train_idx, True), y_train, w_train, _design(test_idx, True))
        good_shared = np.isfinite(y_test) & np.isfinite(pred_shared) & np.isfinite(w_test) & (w_test > 0)
        good_cluster = np.isfinite(y_test) & np.isfinite(pred_cluster) & np.isfinite(w_test) & (w_test > 0)
        loss_shared = float(np.sum(w_test[good_shared] * (y_test[good_shared] - pred_shared[good_shared]) ** 2))
        loss_cluster = float(np.sum(w_test[good_cluster] * (y_test[good_cluster] - pred_cluster[good_cluster]) ** 2))
        gain = (loss_shared - loss_cluster) / max(loss_shared, np.finfo(float).tiny)
        return loss_shared, loss_cluster, float(gain)

    def _rmap_footprint_size():
        r_map = shared.get("R_map", None)
        if r_map is None:
            r_map = 1.5 * shared.get("grid_spacing", np.nan)
        try:
            r_map = float(r_map)
        except Exception:
            r_map = np.nan
        if not np.isfinite(r_map) or r_map <= 0:
            raise ValueError("auto-K needs shared['R_map'] or a positive shared['grid_spacing'].")
        tree = cKDTree(grid_xy)
        try:
            counts = np.asarray(tree.query_ball_point(grid_xy, r=r_map, return_length=True), dtype=int)
        except TypeError:
            counts = np.fromiter((len(x) for x in tree.query_ball_point(grid_xy, r=r_map)), dtype=int, count=n_grid)
        return int(max(1, np.ceil(np.median(counts)))), r_map

    def _resolution_support(labels, K, min_footprint):
        """Grid fraction and count of components below one R_map footprint."""
        labels = np.asarray(labels, dtype=np.int32).reshape(-1)
        small_n = 0
        small_components = 0
        n_components = 0
        min_size = n_grid
        for k in range(1, K + 1):
            idx_k = np.flatnonzero(labels == k)
            if idx_k.size == 0:
                continue
            sub = graph["A"][idx_k, :][:, idx_k]
            n_comp, membership = connected_components(sub, directed=False, return_labels=True)
            sizes = np.bincount(membership, minlength=n_comp).astype(int)
            n_components += int(n_comp)
            min_size = min(min_size, int(sizes.min()))
            small = sizes < min_footprint
            small_components += int(np.sum(small))
            small_n += int(np.sum(sizes[small]))
        return {
            "n_connected_components": int(n_components),
            "minimum_component_grid_locations": int(min_size),
            "small_components_below_one_Rmap_footprint": int(small_components),
            "grid_fraction_in_small_components": float(small_n / max(n_grid, 1)),
        }

    def _eval_auto_k():
        K_grid = sorted({int(k) for k in list(cfg.get("K_GRID", range(2, 10))) if 2 <= int(k) <= max(2, n_grid - 1)})
        if not K_grid:
            raise ValueError("No valid K values in K_GRID.")
        cv_design = _choose_time_cv_design()
        if cv_design is None:
            table = pd.DataFrame({"K": K_grid})
            table["selected_final_k"] = table["K"].eq(min(K_grid))
            return {
                "best_k": int(min(K_grid)), "coarse_k": int(min(K_grid)), "table": table,
                "models": {}, "sample_n": 0,
                "selection": {
                    "mode": "minimum_K",
                    "dynamic_candidates": [int(min(K_grid))],
                },
            }

        # Feature construction and graph building occur once above. Candidate
        # labels are released immediately; only small center matrices are kept
        # to initialize the single final fit. Full-grid fitting is the default.
        sample_idx = _sample_indices(n_grid, int(cfg.get("AUTO_SUBSAMPLE_N", 25000)), row_rel)
        Xs = X_feat[sample_idx]
        ws = row_rel[sample_idx]
        min_footprint, r_map = _rmap_footprint_size()
        degree_cv, n_folds = cv_design
        all_time_idx = np.arange(n_time, dtype=int)
        heldout_folds = [np.arange(fold, n_time, n_folds, dtype=int) for fold in range(n_folds)]
        basis_cv = _time_basis_for_cv(degree_cv)
        rows = []
        models = {}

        for K in K_grid:
            km_s = _weighted_kmeans(Xs, ws, K, seed_local=_safe_seed(seed) + 90001 + K,
                                    nstart=int(cfg.get("KMEANS_NSTART_AUTO", 3)),
                                    itermax=int(cfg.get("KMEANS_ITER_AUTO", 70)))
            labels_full, _ = _dist2_to_centers(X_feat, km_s["centers"])
            labels_ref, centers_ref, _ = _spatial_refine_labels(
                X_feat, labels_full, row_rel, K,
                beta=float(cfg.get("FINAL_LABEL_BETA", 2.8)),
                max_iter=int(cfg.get("AUTOK_REFINE_MAXITER", 3)),
            )
            means, masses = _cluster_time_means(labels_ref)
            support = _resolution_support(labels_ref, K, min_footprint)
            fold_gain, fold_shared, fold_cluster = [], [], []
            for test_idx in heldout_folds:
                train_idx = np.setdiff1d(all_time_idx, test_idx, assume_unique=True)
                loss_shared, loss_cluster, gain = _dynamic_fold_loss(means, masses, basis_cv, train_idx, test_idx)
                fold_gain.append(gain)
                fold_shared.append(loss_shared)
                fold_cluster.append(loss_cluster)
            gain_arr = np.asarray(fold_gain, dtype=float)
            n_gain = int(np.sum(np.isfinite(gain_arr)))
            se_gain = float(np.nanstd(gain_arr, ddof=1) / np.sqrt(n_gain)) if n_gain > 1 else 0.0
            rows.append({
                "K": int(K),
                "mean_dynamic_gain": float(np.nanmean(gain_arr)),
                "se_dynamic_gain": se_gain,
                "mean_shared_time_loss": float(np.nanmean(fold_shared)),
                "mean_cluster_specific_time_loss": float(np.nanmean(fold_cluster)),
                "R_map": float(r_map),
                "R_map_footprint_grid_locations": int(min_footprint),
                **support,
            })
            models[int(K)] = {"centers": centers_ref.astype(np.float32, copy=True)}
            del labels_full, labels_ref, centers_ref, means, masses

        df = pd.DataFrame(rows).sort_values("K").reset_index(drop=True)
        best_row = df.loc[df["mean_dynamic_gain"].idxmax()]
        lower_1se = float(best_row["mean_dynamic_gain"] - best_row["se_dynamic_gain"])
        k_min = int(df["K"].min())
        df["within_one_SE_of_best_dynamic_gain"] = df["mean_dynamic_gain"] >= lower_1se - 1e-12
        fine_to_coarse_scan = []
        fragmentation_stop_at_k = np.nan
        rejected_coarser_k = np.nan
        no_elbow_fallback_k = np.nan
        if lower_1se <= 0:
            selected_k = k_min
            dynamic_candidates = [k_min]
            mode = "no_reliable_cluster_specific_time_evidence"
            reason = "The best held-out cluster-specific trajectory gain is not positive after its one-SE uncertainty bound."
        else:
            dynamic_candidates = df.loc[df["within_one_SE_of_best_dynamic_gain"], "K"].astype(int).tolist()
            candidate_table = df.loc[
                df["K"].isin(dynamic_candidates),
                ["K", "grid_fraction_in_small_components"],
            ].sort_values("K", ascending=False).reset_index(drop=True)
            if candidate_table.shape[0] == 1:
                selected_k = int(candidate_table["K"].iloc[0])
                mode = "single_dynamic_candidate"
                reason = "Only one K lies within one SE of the best held-out dynamic gain."
                fragmentation_stop_at_k = int(selected_k)
            else:
                kvals_desc = candidate_table["K"].to_numpy(dtype=int)
                frag_desc = candidate_table[
                    "grid_fraction_in_small_components"
                ].to_numpy(dtype=float)
                selected_k = int(kvals_desc[0])
                selected_frag = float(frag_desc[0])
                stopped = False
                for next_k, next_frag in zip(kvals_desc[1:], frag_desc[1:]):
                    next_k = int(next_k)
                    next_frag = float(next_frag)
                    should_coarsen = bool(next_frag < selected_frag)
                    fine_to_coarse_scan.append({
                        "from_k": int(selected_k),
                        "to_k": next_k,
                        "fragmentation_from": selected_frag,
                        "fragmentation_to": next_frag,
                        "fragmentation_change_on_coarsening": next_frag - selected_frag,
                        "decision": "coarsen" if should_coarsen else "stop",
                    })
                    if should_coarsen:
                        selected_k = next_k
                        selected_frag = next_frag
                    else:
                        fragmentation_stop_at_k = int(selected_k)
                        rejected_coarser_k = next_k
                        stopped = True
                        break
                if stopped:
                    mode = "dynamic_plateau_then_fine_to_coarse_local_minimum"
                    reason = (
                        "Starting from the finest dynamic candidate, coarsening "
                        "reduced sub-footprint fragmentation until the next "
                        "coarser candidate increased it; retain the first "
                        "fine-side local minimum."
                    )
                else:
                    # Fragmentation locates an elbow; it is not an objective to
                    # minimize all the way to the coarsest supported partition.
                    # When no elbow exists, take one conservative coarsening
                    # step from the finest dynamically supported candidate.
                    selected_k = int(kvals_desc[1])
                    selected_frag = float(frag_desc[1])
                    fragmentation_stop_at_k = int(selected_k)
                    no_elbow_fallback_k = int(selected_k)
                    mode = "dynamic_plateau_no_fragmentation_elbow_one_step_coarsening"
                    reason = (
                        "Sub-footprint fragmentation supplied no fine-to-coarse "
                        "elbow within the dynamic-evidence plateau; retain one "
                        "conservative coarsening step below its finest candidate."
                    )
        df["dynamic_candidate"] = df["K"].isin(dynamic_candidates)
        fine_to_coarse_order = sorted(dynamic_candidates, reverse=True)
        df["fine_to_coarse_rank"] = df["K"].map({
            k_value: rank + 1
            for rank, k_value in enumerate(fine_to_coarse_order)
        })
        visited_k = {
            int(step["from_k"])
            for step in fine_to_coarse_scan
        }
        visited_k.update(
            int(step["to_k"])
            for step in fine_to_coarse_scan
        )
        if len(dynamic_candidates) == 1:
            visited_k.add(int(dynamic_candidates[0]))
        df["fine_to_coarse_visited"] = df["K"].astype(int).isin(visited_k)
        df["selected_final_k"] = df["K"].astype(int).eq(int(selected_k))
        selection = {
            "mode": mode,
            "recommended_k": int(selected_k),
            "best_dynamic_gain": float(best_row["mean_dynamic_gain"]),
            "best_dynamic_gain_lower_1SE": float(lower_1se),
            "dynamic_candidates": dynamic_candidates,
            "fine_to_coarse_order": fine_to_coarse_order,
            "fine_to_coarse_scan": fine_to_coarse_scan,
            "fragmentation_stop_at_k": fragmentation_stop_at_k,
            "rejected_coarser_k": rejected_coarser_k,
            "no_elbow_fallback_k": no_elbow_fallback_k,
            "fragmentation_jump_from": np.nan,
            "fragmentation_jump_to": np.nan,
            "reason": reason,
            "rule": (
                "held-out complete-trajectory evidence, then fine-to-coarse "
                "first local minimum of R_map-footprint fragmentation, with "
                "a one-step no-elbow fallback"
            ),
        }
        return {
            "best_k": int(selected_k),
            "coarse_k": int(dynamic_candidates[0]),
            "table": df,
            "models": models,
            "sample_n": int(sample_idx.size),
            "selection": selection,
        }

    if K_TRAJ is None:
        ek_auto = _eval_auto_k()
        K_final = int(ek_auto["best_k"])
        init_centers = ek_auto.get("models", {}).get(K_final, {}).get("centers")
        ek_auto.pop("models", None)
    else:
        ek_auto = None
        K_final = int(K_TRAJ)
        init_centers = None

    km_full = _weighted_kmeans(
        X_feat,
        row_rel,
        K_final,
        seed_local=_safe_seed(seed) + 90001,
        nstart=int(cfg.get("KMEANS_NSTART_FULL", 6)),
        itermax=int(cfg.get("KMEANS_ITER_FULL", 100)),
        init_centers=init_centers,
    )
    cluster_full, centers_full, min_d2_full = _spatial_refine_labels(
        X_feat,
        km_full["cluster"],
        row_rel,
        K_final,
        beta=float(cfg.get("FINAL_LABEL_BETA", 2.8)),
        max_iter=int(cfg.get("FINAL_LABEL_MAXITER", 10)),
    )
    cluster_full = np.asarray(cluster_full, dtype=np.int32)
    weight_size = np.array([np.sum(row_rel[cluster_full == k]) for k in range(1, K_final + 1)], dtype=float)
    size = np.array([np.sum(cluster_full == k) for k in range(1, K_final + 1)], dtype=int)
    final_wss = float(np.sum(row_rel.astype(float) * min_d2_full))
    final_comp = _component_metrics(cluster_full, K_final, row_rel)

    def _make_palette(labels, K):
        base_cols = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
        ]
        XY = np.asarray(grid_xy, dtype=float)
        XYc = XY - np.nanmean(XY, axis=0, keepdims=True)
        try:
            _, _, vt = np.linalg.svd(XYc[np.isfinite(XYc).all(axis=1)], full_matrices=False)
            axis = vt[0]
        except Exception:
            axis = np.array([1.0, 0.0])
        axis = axis / max(float(np.sqrt(np.sum(axis ** 2))), 1e-8)
        proj = XY @ axis
        ctr = []
        for k in range(1, K + 1):
            ii = labels == k
            ctr.append(float(np.nanmean(proj[ii])) if np.any(ii) else np.inf)
        order = np.argsort(ctr)
        palette = {}
        for rank, idx_k in enumerate(order.tolist()):
            palette[str(idx_k + 1)] = base_cols[rank % len(base_cols)]
        return palette

    palette = _make_palette(cluster_full, K_final)

    def _plot_cluster_map():
        bg = shared.get("combined")
        if isinstance(bg, pd.DataFrame) and "sample_id" in bg.columns:
            bg = bg.loc[bg["sample_id"].astype(str) == str(time_id_plot_map), :]
        else:
            bg = None
        fig, ax = plt.subplots(figsize=(6, 5))
        if isinstance(bg, pd.DataFrame) and bg.shape[0] > 0 and {"x", "y"}.issubset(bg.columns):
            ax.scatter(
                pd.to_numeric(bg["x"], errors="coerce"),
                pd.to_numeric(bg["y"], errors="coerce"),
                s=4,
                alpha=0.70,
                color="#CCCCCC",
                linewidths=0,
            )
        cols = [palette[str(int(k))] for k in cluster_full]
        ax.scatter(grid_xy[:, 0], grid_xy[:, 1], c=cols, s=6, alpha=0.95, linewidths=0)
        ax.set_aspect("equal")
        ax.set_title(f"[spatial_roi_traj] Trajectory clusters @ {time_id_plot_map} | K={K_final}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        return fig, ax

    def _plot_mean_trajectory():
        fig, ax = plt.subplots(figsize=(7, 4))
        xvals = [str(t) for t in time_ids_all]
        for k in range(1, K_final + 1):
            ii = np.where(cluster_full == k)[0]
            if ii.size == 0:
                continue
            w = row_rel[ii].astype(float)
            if np.sum(w) <= 0 or not np.all(np.isfinite(w)):
                y = np.nanmean(mu_sm[ii], axis=0)
            else:
                y = np.nansum(mu_sm[ii] * w[:, None], axis=0) / np.sum(w)
            ax.plot(xvals, y, marker="o", label=f"C{k}", color=palette.get(str(k)))
        ax.set_title(f"[spatial_roi_traj] Mean smoothed muA_adj trajectory per cluster | K={K_final}")
        ax.set_xlabel("time")
        ax.set_ylabel("mean smoothed muA_adj")
        ax.tick_params(axis="x", rotation=90)
        ax.legend(title="cluster")
        fig.tight_layout()
        return fig, ax

    result_key = f"K{K_final}"
    cluster_summary = pd.DataFrame({
        "cluster": np.arange(1, K_final + 1, dtype=int),
        "n": size,
        "weight_sum": weight_size,
        "weight_prop": weight_size / max(float(np.sum(weight_size)), 1e-8),
    })
    out = {
        "scheme_name": "spatial_roi_trajectory_global",
        "K_TRAJ": K_final,
        "time_ids_all": time_ids_all,
        "time_values": _resolved_time_numeric_for_ids(time_ids_all),
        "time_id_plot_map": time_id_plot_map,
        "ref_sample_id": ref_sample_id,
        "fusion_mode": "single_gene_spatial_roi_trajectory",
        "fit_names": [gene_name],
        "n_fit": 1,
        "row_reliability": row_rel,
        "X_fit": X_feat if keep_features else None,
        "X_plot": mu_sm if keep_features else None,
        "auto_k": None if ek_auto is None else {
            "best_k": int(ek_auto["best_k"]),
            "recommended_k": int(ek_auto["best_k"]),
            "coarse_k": int(ek_auto["coarse_k"]),
            "selection": dict(ek_auto.get("selection", {})),
            "table": ek_auto["table"],
            "sample_n": int(ek_auto["sample_n"]),
        },
        "diagnostics": {
            "feature_blocks": feature_blocks,
            "feature_shape": tuple(X_feat.shape),
            "cluster_summary": cluster_summary,
            "component_table": final_comp["component_table"],
            "final_roi_metrics": {k: v for k, v in final_comp.items() if k != "component_table"},
            "basis_shape": tuple(basis.shape),
            "spatial_weight": float(spatial_weight),
            "curvature_weight": float(curvature_weight),
        },
        "results_by_K": {
            result_key: {
                "K": K_final,
                "cluster_full": cluster_full,
                "palette": palette,
                "size": size,
                "weight_size": weight_size,
                "tot_withinss": final_wss,
                "component_metrics": final_comp,
            }
        },
    }

    if do_plot:
        p_map = _plot_cluster_map()
        p_traj = _plot_mean_trajectory()
        out["results_by_K"][result_key]["p_map"] = p_map
        out["results_by_K"][result_key]["p_traj"] = p_traj
        if ek_auto is not None and bool(cfg.get("SHOW_AUTO_K_DIAGNOSTIC", False)):
            tbl = ek_auto["table"]
            if {"mean_dynamic_gain", "se_dynamic_gain", "grid_fraction_in_small_components"}.issubset(tbl.columns):
                fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
                axes[0].errorbar(tbl["K"], tbl["mean_dynamic_gain"], yerr=tbl["se_dynamic_gain"], marker="o", color="#355C7D", capsize=3)
                axes[0].axhline(0.0, color="0.6", linewidth=1, linestyle=":")
                axes[0].axvline(K_final, linestyle="--", color="#C44536", linewidth=1.2)
                axes[0].set(xlabel="K", ylabel="Held-out dynamic gain", title="Complete-trajectory evidence", xticks=tbl["K"].tolist())
                axes[1].plot(tbl["K"], tbl["grid_fraction_in_small_components"], marker="o", color="#C44536")
                axes[1].axvline(K_final, linestyle="--", color="#C44536", linewidth=1.2, label=f"selected K={K_final}")
                axes[1].set(xlabel="K", ylabel="Grid fraction in sub-footprint components", title="R_map-resolution support", xticks=tbl["K"].tolist())
                axes[1].legend(frameon=False)
                fig.suptitle(f"[spatial_roi_traj] two-stage auto-K | final K={K_final}")
                out["auto_k"]["plot"] = fig
        plt.show()

    return out

def run_global_response_trajectory_clustering(*args, **kwargs):
    """Backward-compatible alias for the abandoned v1/v2 wrapper name."""
    for key in ["response_source", "level_weight", "response_weight", "response_shape_weight", "plot_base_mu"]:
        kwargs.pop(key, None)
    return run_global_spatial_roi_trajectory_clustering(*args, **kwargs)
