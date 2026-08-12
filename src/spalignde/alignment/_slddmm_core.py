"""General rasterization and shooting-LDDMM utilities for spAlignDE.

This module is intentionally independent of cross-sample rigid prealignment.
It operates on source/target point labels or multi-channel rasters and can be
reused by cross-sample and cross-modality workflows.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.fft
from torch.nn.functional import grid_sample
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree

def centers_to_edges(c):
    c = np.asarray(c, float)
    if c.size < 2:
        d = 1.0
    else:
        d = float(np.median(np.diff(c)))
    edges = np.empty(c.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (c[:-1] + c[1:])
    edges[0] = c[0] - d / 2.0
    edges[-1] = c[-1] + d / 2.0
    return edges


def build_shared_grid(x1, y1, x2, y2, dx=30.0, expand=1.05):
    """Build a square grid covering both point clouds.

    Parameters
    ----------
    dx : float
        Grid spacing in coordinate units. Larger values emphasize coarse shape
        and run faster; smaller values preserve local detail and cost more.
    expand : float
        Margin factor around the joint bounding box.
    """
    x_all = np.concatenate([np.asarray(x1, float), np.asarray(x2, float)])
    y_all = np.concatenate([np.asarray(y1, float), np.asarray(y2, float)])

    xmin, xmax = x_all.min(), x_all.max()
    ymin, ymax = y_all.min(), y_all.max()
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    half = 0.5 * max((xmax - xmin), (ymax - ymin)) * float(expand)

    xmin, xmax = cx - half, cx + half
    ymin, ymax = cy - half, cy + half

    X = np.arange(xmin, xmax + dx, dx, dtype=float)
    Y = np.arange(ymin, ymax + dx, dx, dtype=float)
    return X, Y


def rasterize_cluster_channels_on_grid(x, y, labels, cats, X, Y, blur_sigma=1.5):
    """Rasterize per-cluster point counts into channels on a fixed grid.

    `blur_sigma` controls Gaussian smoothing of sparse histograms.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    labels = np.asarray(labels).astype(str)
    cats = list(cats)

    x_edges = centers_to_edges(X)
    y_edges = centers_to_edges(Y)
    H, W = len(Y), len(X)
    K = len(cats)

    raw = np.zeros((K, H, W), dtype=np.float64)
    for k, c in enumerate(cats):
        sel = labels == c
        if not np.any(sel):
            continue
        h, _, _ = np.histogram2d(y[sel], x[sel], bins=[y_edges, x_edges])
        if blur_sigma and blur_sigma > 0:
            h = gaussian_filter(h, sigma=blur_sigma, mode='nearest')
        raw[k] = h
    return raw


def prep_lddmm_multichannel(raw_src, raw_tgt, cluster_w=0.6, density_w=1.0, eps=1e-8):
    """Build weighted multi-channel source/target images for LDDMM.

    Parameters
    ----------
    raw_src, raw_tgt : np.ndarray
        Per-cluster raster stacks with shape ``(K, H, W)``.
    cluster_w, density_w : float
        Channel weights for composition channels and density channel.
    eps : float
        Numerical stabilizer for composition normalization.

    Returns
    -------
    source_image, target_image : np.ndarray
        Weighted multi-channel arrays with shape ``(K+1, H, W)``.
    source_comp, target_comp : np.ndarray
        Normalized per-pixel cluster composition (K channels).
    source_density_ch, target_density_ch : np.ndarray
        Log-scaled density channel in ``[0, 1]``.
    weights : np.ndarray
        Per-channel weights applied to source/target images.
    """
    dens_src = raw_src.sum(axis=0)
    dens_tgt = raw_tgt.sum(axis=0)

    comp_src = raw_src / np.maximum(dens_src[None, :, :], eps)
    comp_tgt = raw_tgt / np.maximum(dens_tgt[None, :, :], eps)

    dens_src_log = np.log1p(dens_src)
    dens_tgt_log = np.log1p(dens_tgt)
    dens_scale = np.percentile(np.concatenate([dens_src_log.ravel(), dens_tgt_log.ravel()]), 99.0)
    dens_scale = max(float(dens_scale), 1e-6)

    dens_src_ch = np.clip(dens_src_log / dens_scale, 0.0, 1.0)
    dens_tgt_ch = np.clip(dens_tgt_log / dens_scale, 0.0, 1.0)

    source_image = np.concatenate([comp_src, dens_src_ch[None, :, :]], axis=0)
    target_image = np.concatenate([comp_tgt, dens_tgt_ch[None, :, :]], axis=0)

    K = raw_src.shape[0]
    weights = np.concatenate([np.full(K, cluster_w, dtype=np.float64), np.array([density_w], dtype=np.float64)])
    source_image = source_image * np.sqrt(weights)[:, None, None]
    target_image = target_image * np.sqrt(weights)[:, None, None]

    return source_image, target_image, comp_src, comp_tgt, dens_src_ch, dens_tgt_ch, weights


def composition_to_rgb(comp):
    K = comp.shape[0]
    if K == 0:
        return np.zeros((comp.shape[1], comp.shape[2], 3), dtype=float)

    hsv_h = np.linspace(0.0, 1.0, K, endpoint=False)
    palette = np.array([plt.cm.hsv(h)[:3] for h in hsv_h], dtype=float)
    rgb = np.tensordot(comp.transpose(1, 2, 0), palette, axes=([2], [0]))
    return np.clip(rgb, 0.0, 1.0)




def build_cluster_labels(df_src: pd.DataFrame, df_tgt: pd.DataFrame, cluster_col: str = "leiden_harmony_refined"):
    """Build unified cluster categories and aligned source/target label vectors."""
    cats = sorted(set(df_src[cluster_col].astype(str)) | set(df_tgt[cluster_col].astype(str)))
    labels1 = df_src[cluster_col].astype(str).values
    labels2 = df_tgt[cluster_col].astype(str).values
    return cats, labels1, labels2


def prepare_rasterization_and_multichannel(
    x_src_new, y_src_new, x_tgt, y_tgt, labels1, labels2, cats, dx=30, expand=1.05, blur_sigma=1, cluster_w=1, density_w=1
):
    """Rasterize source/target points and build weighted LDDMM channels.

    Returns a tuple to preserve backward compatibility with older notebook cells.
    Newer tutorials should read outputs as:
    ``source_grid_x, source_grid_y, target_grid_x, target_grid_y, ...``.

    Parameter notes
    ---------------
    dx : float
        Raster grid spacing.
    blur_sigma : float
        Smoothing on per-cluster histograms before composition normalization.
    cluster_w, density_w : float
        Relative contribution of cluster-composition channels vs. density channel
        in the LDDMM matching loss.
    """
    source_grid_x, source_grid_y = build_shared_grid(x_src_new, y_src_new, x_tgt, y_tgt, dx=dx, expand=expand)
    target_grid_x, target_grid_y = source_grid_x.copy(), source_grid_y.copy()

    source_raw_raster = rasterize_cluster_channels_on_grid(
        x_src_new, y_src_new, labels1, cats, source_grid_x, source_grid_y, blur_sigma=blur_sigma
    )
    target_raw_raster = rasterize_cluster_channels_on_grid(
        x_tgt, y_tgt, labels2, cats, target_grid_x, target_grid_y, blur_sigma=blur_sigma
    )

    source_image, target_image, source_comp, target_comp, source_density_ch, target_density_ch, channel_weights = prep_lddmm_multichannel(
        source_raw_raster,
        target_raw_raster,
        cluster_w=cluster_w,
        density_w=density_w,
    )

    source_preview = composition_to_rgb(source_comp)
    target_preview = composition_to_rgb(target_comp)
    return (
        source_grid_x,
        source_grid_y,
        target_grid_x,
        target_grid_y,
        source_raw_raster,
        target_raw_raster,
        source_image,
        target_image,
        source_comp,
        target_comp,
        source_density_ch,
        target_density_ch,
        channel_weights,
        source_preview,
        target_preview,
    )


def plot_rasterization_preview(
    preview_I,
    preview_J,
    dens_I_ch,
    dens_J_ch,
    *,
    figsize=(12, 10),
) -> None:
    """Visual QC for rasterized channels used by LDDMM."""
    fig, ax = plt.subplots(2, 2, figsize=figsize)

    ax[0, 0].imshow(preview_I, origin="lower")
    ax[0, 0].set_title("Source cluster composition (RGB preview)")

    ax[0, 1].imshow(preview_J, origin="lower")
    ax[0, 1].set_title("Target cluster composition (RGB preview)")

    ax[1, 0].imshow(dens_I_ch, origin="lower", cmap="magma")
    ax[1, 0].set_title("Source density channel")

    ax[1, 1].imshow(dens_J_ch, origin="lower", cmap="magma")
    ax[1, 1].set_title("Target density channel")

    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    plt.tight_layout()


# -----------------------------
# Utilities
# -----------------------------

def affine_from_components(L, T):
    """Build 2D homogeneous affine matrix from linear part and translation.

    Parameters
    ----------
    L : tensor, shape (2,2)
        Linear component.
    T : tensor, shape (2,)
        Translation component in (y, x) order.
    """
    A = torch.eye(3, device=L.device, dtype=L.dtype)
    A[:2, :2] = L
    A[:2, 2] = T
    return A


def clip(I):
    Ic = torch.clone(I)
    Ic[Ic < 0] = 0
    Ic[Ic > 1] = 1
    return Ic


def sample_image_on_coords(grid_coords, values, query_yx, padding_mode="zeros", mode="bilinear", align_corners=True):
    # grid_coords: [y_coords, x_coords]
    # values: (C,H,W)
    # query_yx: (2,*,*) in physical coords, channel order [y, x]
    values = torch.as_tensor(values)
    query_yx = torch.as_tensor(query_yx).clone()

    for dim in range(2):
        query_yx[dim] -= grid_coords[dim][0]
        query_yx[dim] /= (grid_coords[dim][-1] - grid_coords[dim][0])
    query_yx = query_yx * 2.0 - 1.0

    sampling_grid = query_yx.flip(0).permute((1, 2, 0))[None]
    out = grid_sample(
        values[None],
        sampling_grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    return out[0]


# -----------------------------
# Finite-difference ops
# -----------------------------

def central_diff(f, dx, dim):
    # f: (..., H, W)
    pad = [0, 0, 0, 0]
    if dim == -1:  # x/cols
        # pad W: left/right replicate
        left = f[..., :1]
        right = f[..., -1:]
        f_pad = torch.cat([left, f, right], dim=-1)
        return (f_pad[..., 2:] - f_pad[..., :-2])/(2.0*dx)
    else:          # y/rows
        top = f[..., :1, :]
        bottom = f[..., -1:, :]
        f_pad = torch.cat([top, f, bottom], dim=-2)
        return (f_pad[..., 2:, :] - f_pad[..., :-2, :])/(2.0*dx)


def jacobian_and_div(v, dv):
    # v: (H,W,2), dv: tensor([dy, dx])
    vy = v[..., 0]
    vx = v[..., 1]
    dy, dx = dv[0], dv[1]
    dvx_dx = central_diff(vx, dx, dim=-1)
    dvx_dy = central_diff(vx, dy, dim=-2)
    dvy_dx = central_diff(vy, dx, dim=-1)
    dvy_dy = central_diff(vy, dy, dim=-2)
    # shape alignment: central_diff keeps (H,W)
    div_v = dvx_dx + dvy_dy
    # (∇v)^T m term components will use these
    return dvx_dx, dvx_dy, dvy_dx, dvy_dy, div_v


# -----------------------------
# Convolution with K (Green's function): v = K * m
# -----------------------------

def build_kernel_K(xv, a=500.0, p=2.0):
    # xv: [y_coords, x_coords]
    y, x = xv
    dy = (y[1]-y[0]).to(y.device)
    dx = (x[1]-x[0]).to(x.device)
    H, W = len(y), len(x)

    # frequency grid in physical units matching earlier code
    fy = torch.arange(H, device=y.device, dtype=y.dtype)/(H*dy)
    fx = torch.arange(W, device=x.device, dtype=x.dtype)/(W*dx)
    FY, FX = torch.meshgrid(fy, fx, indexing='ij')
    # LL like in your code
    LL = (1.0 + 2.0*a**2 * ((1.0 - torch.cos(2.0*np.pi*FY*dy))/(dy**2) +
                             (1.0 - torch.cos(2.0*np.pi*FX*dx))/(dx**2)))**(p*2.0)
    K = 1.0/LL  # scalar kernel
    return K  # (H,W)


def apply_K(m, K):
    # m: (H,W,2), K: (H,W). Apply per-channel via FFT
    v = []
    for c in range(2):
        Mf = torch.fft.fftn(m[..., c], dim=(-2, -1))
        vf = Mf * K
        v_c = torch.fft.ifftn(vf, dim=(-2, -1)).real
        v.append(v_c)
    v = torch.stack(v, dim=-1)
    return v


# -----------------------------
# Semi-Lagrangian advection of vector/tensor fields
# -----------------------------

def advect_field(F, v, xv, dt):
    # F: (C,H,W)  (e.g., C=2 for vector m)
    # v: (H,W,2)
    H, W = v.shape[:2]
    Y, X = xv
    XV = torch.stack(torch.meshgrid(Y, X, indexing='ij'), -1)  # (H,W,2)
    back = (XV - v*dt).permute(2,0,1)                          # (2,H,W)
    return sample_image_on_coords(xv, F, back, padding_mode="border")


# -----------------------------
# Geodesic shooting integrator (Euler step)
# -----------------------------

def geodesic_shooting(m0, xv, nt, K, dv):
    dt = 1.0/nt
    m = m0
    v_list, m_list = [], []
    for t in range(nt):
        v = apply_K(m, K)            # v = K * m
        v_list.append(v)
        m_list.append(m)
        dvx_dx, dvx_dy, dvy_dx, dvy_dy, div_v = jacobian_and_div(v, dv)
        mx, my = m[..., 0], m[..., 1]
        Sx = dvx_dx*mx + dvy_dx*my
        Sy = dvx_dy*mx + dvy_dy*my
        S = torch.stack((Sx, Sy), dim=-1)
        S = S + m*div_v[..., None]
        m_tilde = (m - dt*S).permute(2, 0, 1)
        m_next = advect_field(m_tilde, v, xv, dt).permute(1, 2, 0)
        m = m_next
    return v_list, m_list  # lists of (H,W,2)


# -----------------------------
# Shooting-based LDDMM main
# -----------------------------

def LDDMM_shooting(
    x_src, I, x_tgt, J,
    affine_init=None, m0_init=None, velocity_grid=None,
    model_cfg=None, optim_cfg=None, em_cfg=None, intensity_cfg=None,
    device='cpu', dtype=torch.float64, muB=None, muA=None,
    verbose=True, print_every=100,
):
    """Shooting-based LDDMM with config-style parameter groups.

    Parameters
    ----------
    x_src : list[1D tensor/array]
        Source image grid coordinates in physical units: [y_coords, x_coords].
    I : tensor/array, shape (C, H_src, W_src)
        Source multi-channel image on ``x_src``.
    x_tgt : list[1D tensor/array]
        Target image grid coordinates in physical units: [y_coords, x_coords].
    J : tensor/array, shape (C, H_tgt, W_tgt)
        Target multi-channel image on ``x_tgt``.
    affine_init : tensor/array or None, shape (3,3) or (2,3)
        Initial affine transform mapping source -> target.
        If None, identity affine is used.
    m0_init : tensor/array or None, shape (H_v, W_v, 2)
        Initial momentum for geodesic shooting.
        If None, initialized to zeros.
    velocity_grid : list[1D tensor/array] or None
        Grid for velocity/momentum field [y_coords, x_coords].
        If None, built automatically from source extent and ``model_cfg``.
    model_cfg : dict or None
        Geometric/model settings:
        ``a`` (smoothness scale), ``p`` (operator power),
        ``expand`` (velocity-domain expansion factor), ``nt`` (time steps),
        ``grid_step`` (velocity grid spacing).
        Practical guide:
        - larger ``a`` -> smoother, more global deformation;
        - larger ``nt`` -> finer flow integration (more compute);
        - larger ``grid_step`` -> coarser velocity field (more regularized).
    optim_cfg : dict or None
        Optimization schedule settings:
        ``niter``, ``diffeo_start``, ``lrL``, ``lrT``, ``lrM``,
        ``affine_slowdown``, ``grad_clip_m0``, ``lrM_decay``, ``lrM_min``,
        ``rollback_on_rise``, ``rollback_factor``, ``rollback_patience``,
        ``minimum_affine_determinant`` and ``restore_best``.
    em_cfg : dict or None
        EM update schedule for mixture weights:
        ``update_every`` (iterations), ``start_iter`` (warmup end).
    intensity_cfg : dict or None
        Data-term and regularization scales:
        ``sigmaM`` (match), ``sigmaA``/``sigmaB`` (appearance classes),
        and ``sigmaR`` (deformation regularization).
        Practical guide:
        - smaller ``sigmaM`` increases matching pressure;
        - larger ``sigmaR`` weakens the deformation penalty and allows stronger
          warps;
    device : str
        Torch device (e.g., ``'cpu'`` or ``'cuda:0'``).
    dtype : torch dtype
        Numeric precision used internally.
    muA, muB : tensor/array or None
        Optional fixed appearance means for A/B mixture classes.
        If None, estimated during EM updates.
    draw_every : int
        Reserved display cadence parameter.
    verbose : bool
        Whether to print optimization progress.
    print_every : int
        Iteration interval for log printing.

    Returns
    -------
    out : dict
        Registration results including affine ``A``, velocity ``v``,
        momentum ``m0``, transformed coordinates, and energy history.
    """

    def _merge_cfg(defaults, user_cfg):
        cfg = dict(defaults)
        if user_cfg is not None:
            cfg.update(user_cfg)
        return cfg

    def _to_tensor(x, *, requires_grad=False):
        t = torch.as_tensor(x, device=device, dtype=dtype)
        if requires_grad:
            t = t.detach().clone().requires_grad_(True)
        return t

    model = _merge_cfg(
        {
            'a': 500.0,
            'p': 2.0,
            'expand': 2.0,
            'nt': 5,
            'grid_step': None,
        },
        model_cfg,
    )
    optim = _merge_cfg(
        {
            'niter': 500,
            'diffeo_start': 0,
            'lrL': 2e-8,
            'lrT': 2e-1,
            'lrM': 2e3,
            'affine_slowdown': 10.0,
            'grad_clip_m0': None,
            'lrM_decay': 1.0,
            'lrM_min': 2e3,
            'rollback_on_rise': False,
            'rollback_factor': 0.5,
            'rollback_patience': 6,
            'minimum_energy_improvement': 1e-5,
            'minimum_affine_determinant': 1e-6,
            'restore_best': False,
        },
        optim_cfg,
    )
    em = _merge_cfg(
        {
            'update_every': 5,
            'start_iter': 50,
        },
        em_cfg,
    )
    intensity = _merge_cfg(
        {
            'sigmaM': 1.0,
            'sigmaB': 2.0,
            'sigmaA': 5.0,
            'sigmaR': 5e5,
        },
        intensity_cfg,
    )

    a = model['a']
    p = model['p']
    expand = model['expand']
    nt = int(model['nt'])
    grid_step = model['grid_step']

    niter = int(optim['niter'])
    diffeo_start = int(optim['diffeo_start'])
    lrL = optim['lrL']
    lrT = optim['lrT']
    lrM = optim['lrM']
    affine_slowdown = optim['affine_slowdown']
    grad_clip_m0 = optim['grad_clip_m0']
    lrM_decay = optim['lrM_decay']
    lrM_min = optim['lrM_min']
    rollback_on_rise = bool(optim['rollback_on_rise'])
    rollback_factor = float(optim['rollback_factor'])
    rollback_patience = int(optim['rollback_patience'])
    minimum_energy_improvement = float(optim['minimum_energy_improvement'])
    minimum_affine_determinant = float(optim['minimum_affine_determinant'])
    restore_best = bool(optim['restore_best'])

    em_update_every = int(em['update_every'])
    em_start = int(em['start_iter'])

    sigmaM = intensity['sigmaM']
    sigmaB = intensity['sigmaB']
    sigmaA = intensity['sigmaA']
    sigmaR = intensity['sigmaR']

    # Tensorize image grids + images
    x_src = [_to_tensor(x) for x in x_src]
    x_tgt = [_to_tensor(x) for x in x_tgt]
    I = _to_tensor(I)
    J = _to_tensor(J)

    # Affine init (from affine_init matrix, or identity)
    if affine_init is None:
        L0 = torch.eye(2, device=device, dtype=dtype)
        T0 = torch.zeros(2, device=device, dtype=dtype)
    else:
        A0 = _to_tensor(affine_init)
        if A0.shape == (3, 3):
            L0, T0 = A0[:2, :2], A0[:2, -1]
        elif A0.shape == (2, 3):
            L0, T0 = A0[:2, :2], A0[:2, -1]
        else:
            raise ValueError(f'affine_init must be (3,3) or (2,3), got {tuple(A0.shape)}')

    L = L0.detach().clone().requires_grad_(True)
    T = T0.detach().clone().requires_grad_(True)

    # Velocity grid init
    if velocity_grid is None:
        lo = torch.stack([x[0] for x in x_src])
        hi = torch.stack([x[-1] for x in x_src])
        ctr = 0.5 * (lo + hi)
        rad = 0.5 * (hi - lo) * expand

        if grid_step is None:
            grid_step = float(a) * 0.5

        xv = []
        for c, r in zip(ctr, rad):
            g = torch.arange(c - r, c + r, grid_step, device=device, dtype=dtype)
            if g.numel() < 3:
                g = torch.linspace(c - r, c + r, steps=3, device=device, dtype=dtype)
            xv.append(g)
    else:
        xv = [_to_tensor(g) for g in velocity_grid]

    dv = torch.stack([g[1] - g[0] for g in xv])
    DV = torch.prod(dv)

    # Kernel for v = K * m
    K = build_kernel_K(xv, a=a, p=p)

    # Initial momentum
    H, W = len(xv[0]), len(xv[1])
    if m0_init is None:
        m0 = torch.zeros((H, W, 2), device=device, dtype=dtype, requires_grad=True)
    else:
        m0 = _to_tensor(m0_init, requires_grad=True)

    # Target mesh
    XJ = torch.stack(torch.meshgrid(*x_tgt, indexing='ij'), -1)

    # Mixture weights
    WM = torch.ones(J[0].shape, dtype=J.dtype, device=J.device)*0.5
    WB = torch.ones(J[0].shape, dtype=J.dtype, device=J.device)*0.4
    WA = torch.ones(J[0].shape, dtype=J.dtype, device=J.device)*0.1

    estimate_muA = muA is None
    estimate_muB = muB is None

    Esave = []
    t_start = time.time()
    best_energy = float('inf')
    best_iteration = -1
    best_state = None
    previous_energy = None
    previous_state = None
    bad_steps = 0
    step_scale = 1.0

    if verbose:
        print(f"[LDDMM] start: niter={niter}, nt={nt}, device={device}, I_shape={tuple(I.shape)}, J_shape={tuple(J.shape)}")

    # Optimization loop (GD for L, T, m0)
    for it in range(niter):
        # Compose affine inverse once per iter
        A = affine_from_components(L, T)
        determinant = torch.linalg.det(A[:2, :2])
        if (
            not torch.isfinite(A).all()
            or not torch.isfinite(determinant)
            or torch.abs(determinant) < minimum_affine_determinant
        ):
            if best_state is None:
                raise RuntimeError(
                    'The affine component became non-finite or singular before '
                    'a valid S-LDDMM checkpoint was available.'
                )
            with torch.no_grad():
                L.copy_(best_state['L'])
                T.copy_(best_state['T'])
                m0.copy_(best_state['m0'])
            if verbose:
                print('[LDDMM] stopped before a singular affine update')
            break
        Ai = torch.linalg.inv(A)

        # Shooting: build v(t), m(t) from m0
        v_list, m_list = geodesic_shooting(m0, xv, nt, K, dv)  # lists of (H,W,2)

        # Warp coordinates XJ back to source frame using v(t) and affine inverse
        Xs = (Ai[:2,:2] @ XJ[..., None])[..., 0] + Ai[:2, -1]
        for t in range(nt-1, -1, -1):
            v_t = torch.stack((v_list[t][...,0], v_list[t][...,1]), dim=0)  # (2,H,W)
            Xs = Xs + sample_image_on_coords(xv, -v_t, Xs.permute(2,0,1)).permute(1,2,0)/nt

        # Resample source image
        AI = sample_image_on_coords(x_src, I, Xs.permute(2,0,1), padding_mode="border")

        # Linear contrast estimation (same as your original)
        B = torch.ones(1 + AI.shape[0], AI.shape[1]*AI.shape[2], device=AI.device, dtype=AI.dtype)
        B[1:AI.shape[0]+1] = AI.reshape(AI.shape[0], -1)
        with torch.no_grad():
            BB = B @ (B*WM.ravel()).T
            BJ = B @ ((J*WM).reshape(J.shape[0], -1)).T
            small = 0.1
            coeffs = torch.linalg.solve(BB + small*torch.eye(BB.shape[0], device=BB.device, dtype=BB.dtype), BJ)
        fAI = ((B.T @ coeffs).T).reshape(J.shape)

        # Energies
        EM = torch.sum((fAI - J)**2 * WM) / (2.0 * (sigmaM**2))
        if EM.dim() != 0:
            EM = EM.sum()  # guard: ensure scalar
        # Regularizer: integrate over time using <m,v> = <Lv, v> (avoid FFT autograd)
        dt = 1.0/nt
        ER_terms = [torch.sum(m_list[t] * v_list[t]) * DV * dt for t in range(nt)]
        ER = (0.5/(sigmaR**2)) * torch.stack(ER_terms).sum()

        E = EM + ER
        E_scalar = E
        if E_scalar.dim() != 0:
            E_scalar = E_scalar.sum()
        tosave = [E_scalar.detach().cpu().item(), EM.detach().cpu().item(), ER.detach().cpu().item()]

        Esave.append(tosave)

        energy_value = float(tosave[0])
        if np.isfinite(energy_value) and energy_value < best_energy:
            best_energy = energy_value
            best_iteration = it
            best_state = {
                'L': L.detach().clone(),
                'T': T.detach().clone(),
                'm0': m0.detach().clone(),
            }

        if verbose and ((it % print_every == 0) or (it == niter - 1)):
            msg = f"[LDDMM] iter {it+1}/{niter} E={tosave[0]:.6e} EM={tosave[1]:.6e} ER={tosave[2]:.6e}"
            print(msg)

        if (
            rollback_on_rise
            and previous_energy is not None
            and energy_value > previous_energy + minimum_energy_improvement
        ):
            with torch.no_grad():
                L.copy_(previous_state['L'])
                T.copy_(previous_state['T'])
                m0.copy_(previous_state['m0'])
            step_scale = max(step_scale * rollback_factor, 1e-3)
            bad_steps += 1
            if verbose:
                print(
                    '[LDDMM] rollback: '
                    f'E={energy_value:.6e} > {previous_energy:.6e}; '
                    f'step_scale={step_scale:.3g}'
                )
            if bad_steps >= rollback_patience:
                if verbose:
                    print(f'[LDDMM] early stop after {bad_steps} rejected steps')
                break
            continue

        bad_steps = max(0, bad_steps - 1)
        previous_energy = energy_value
        previous_state = {
            'L': L.detach().clone(),
            'T': T.detach().clone(),
            'm0': m0.detach().clone(),
        }

        # Backprop
        E.backward()
        with torch.no_grad():
            # explicit schedules (no boolean-arithmetic expressions)
            diffeo_on = (it >= diffeo_start)
            if diffeo_on:
                affine_scale = 1.0 / affine_slowdown
                m_steps = it - diffeo_start
                lrM_t = max(lrM_min, lrM * (lrM_decay ** m_steps))
            else:
                affine_scale = 1.0
                lrM_t = 0.0

            # affine updates
            candidate_L = L - (lrL * affine_scale * step_scale) * L.grad
            candidate_T = T - (lrT * affine_scale * step_scale) * T.grad
            candidate_A = affine_from_components(candidate_L, candidate_T)
            candidate_det = torch.linalg.det(candidate_A[:2, :2])
            if (
                torch.isfinite(candidate_A).all()
                and torch.isfinite(candidate_det)
                and torch.abs(candidate_det) >= minimum_affine_determinant
            ):
                L.copy_(candidate_L)
                T.copy_(candidate_T)
            else:
                step_scale = max(step_scale * rollback_factor, 1e-3)
                if verbose:
                    print(
                        '[LDDMM] rejected non-finite/singular affine step; '
                        f'step_scale={step_scale:.3g}'
                    )
            L.grad.zero_(); T.grad.zero_()

            # initial momentum update with optional clipping
            if torch.isfinite(m0.grad).all() and lrM_t > 0.0:
                if grad_clip_m0 is not None:
                    gnorm = torch.linalg.vector_norm(m0.grad)
                    if torch.isfinite(gnorm) and gnorm > grad_clip_m0:
                        m0.grad.mul_(grad_clip_m0 / (gnorm + 1e-12))
                m0 -= (lrM_t * step_scale) * m0.grad
            m0.grad.zero_()

        # EM updates for mixture weights
        if not it % em_update_every:
            with torch.no_grad():
                if estimate_muA:
                    muA = torch.sum(WA*J, dim=(-1,-2))/torch.sum(WA)
                if estimate_muB:
                    muB = torch.sum(WB*J, dim=(-1,-2))/torch.sum(WB)

                if it >= em_start:
                    W = torch.stack((WM, WA, WB))
                    pi = torch.sum(W, dim=(1,2))
                    pi += torch.max(pi)*1e-6
                    pi /= torch.sum(pi)

                    WMn = pi[0]* torch.exp(-torch.sum((fAI - J)**2, 0)/2.0/(sigmaM**2)) / (np.sqrt(2.0*np.pi*(sigmaM**2))**J.shape[0])
                    WAn = pi[1]* torch.exp(-torch.sum((muA[...,None,None] - J)**2, 0)/2.0/(sigmaA**2)) / (np.sqrt(2.0*np.pi*(sigmaA**2))**J.shape[0])
                    WBn = pi[2]* torch.exp(-torch.sum((muB[...,None,None] - J)**2, 0)/2.0/(sigmaB**2)) / (np.sqrt(2.0*np.pi*(sigmaB**2))**J.shape[0])

                    WS = WMn + WAn + WBn
                    WS += torch.max(WS)*1e-6
                    WM, WA, WB = WMn/WS, WAn/WS, WBn/WS

    if restore_best and best_state is not None:
        with torch.no_grad():
            L.copy_(best_state['L'])
            T.copy_(best_state['T'])
            m0.copy_(best_state['m0'])

    # Final A and last v/m fields (from m0)
    A = affine_from_components(L, T).clone().detach()
    v_list, m_list = geodesic_shooting(m0.detach(), xv, nt, K, dv)
    v0 = torch.stack(v_list, dim=0).detach()  # (nt,H,W,2)
    m_all = torch.stack(m_list, dim=0).detach()

    elapsed_sec = time.time() - t_start
    final_E = Esave[-1][0] if len(Esave) > 0 else np.nan

    if verbose:
        print(f"[LDDMM] done in {elapsed_sec:.1f}s. finalE={final_E:.6e}")

    return {
        'A': A,
        'm0': m0.detach(),
        'v': v0,
        'm': m_all,
        'xv': xv,
        'WM': WM.clone().detach(),
        'WB': WB.clone().detach(),
        'WA': WA.clone().detach(),
        'Esave': Esave,
        'elapsed_sec': elapsed_sec,
        'best_energy': best_energy,
        'best_iteration': best_iteration,
        'step_scale_final': step_scale,
    }




def run_lddmm_pipeline_source_target(
    source_grid_y,
    source_grid_x,
    source_image,
    target_grid_y,
    target_grid_x,
    target_image,
    model_cfg=None,
    optim_cfg=None,
    device=None,
    verbose=True,
    print_every=100,
):
    """Run LDDMM with tutorial defaults using explicit source/target naming.

    Inputs are source and target grid axes plus multi-channel images on those
    grids. Returns the raw output dict from `LDDMM_shooting`.
    """
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if model_cfg is None:
        model_cfg = {
            "a": 300,
            "p": 2.0,
            "expand": 2.0,
            "nt": 3,
            "grid_step": 100,
        }

    out = LDDMM_shooting(
        [source_grid_y, source_grid_x],
        source_image,
        [target_grid_y, target_grid_x],
        target_image,
        model_cfg=model_cfg,
        optim_cfg=optim_cfg,
        device=device,
        verbose=verbose,
        print_every=print_every,
    )
    return out


def run_lddmm_pipeline(YI, XI, I, YJ, XJ, J, model_cfg=None, optim_cfg=None, device=None, verbose=True, print_every=100):
    """Backward-compatible wrapper with legacy ``I/J`` and ``XI/YI`` names."""
    return run_lddmm_pipeline_source_target(
        source_grid_y=YI,
        source_grid_x=XI,
        source_image=I,
        target_grid_y=YJ,
        target_grid_x=XJ,
        target_image=J,
        model_cfg=model_cfg,
        optim_cfg=optim_cfg,
        device=device,
        verbose=verbose,
        print_every=print_every,
    )


def _normalize_point_transform_inputs(velocity_grid, velocity_field, affine_matrix, points):
    """Harmonize dtype/device and validate shapes for point-transform utilities."""
    ref = None
    if torch.is_tensor(velocity_field):
        ref = velocity_field
    elif isinstance(velocity_grid, (list, tuple)) and len(velocity_grid) > 0 and torch.is_tensor(velocity_grid[0]):
        ref = velocity_grid[0]
    elif torch.is_tensor(affine_matrix):
        ref = affine_matrix

    device = ref.device if ref is not None else torch.device("cpu")
    dtype = ref.dtype if ref is not None else torch.float32

    def _to_tensor(x):
        return x.to(device=device, dtype=dtype) if torch.is_tensor(x) else torch.as_tensor(x, device=device, dtype=dtype)

    if not isinstance(velocity_grid, (list, tuple)) or len(velocity_grid) != 2:
        raise ValueError("velocity_grid must be [y_coords, x_coords]")

    vg = [_to_tensor(g) for g in velocity_grid]
    vf = _to_tensor(velocity_field)
    A = _to_tensor(affine_matrix)
    pts = _to_tensor(points)

    if vf.ndim != 4 or vf.shape[-1] != 2:
        raise ValueError(f"velocity_field must have shape (nt,H,W,2), got {tuple(vf.shape)}")
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must have shape (N,2), got {tuple(pts.shape)}")

    if A.shape == (2, 3):
        A_h = torch.eye(3, device=device, dtype=dtype)
        A_h[:2, :] = A
        A = A_h
    elif A.shape != (3, 3):
        raise ValueError(f"affine_matrix must be (3,3) or (2,3), got {tuple(A.shape)}")

    return vg, vf, A, pts


def map_points_source_to_target(velocity_grid, velocity_field, affine_matrix, source_points):
    """Forward map source points (y, x): diffeomorphic flow then affine.

    Typically called with `xv`, `v`, `A` from LDDMM output.
    """
    vg, vf, A, pts = _normalize_point_transform_inputs(velocity_grid, velocity_field, affine_matrix, source_points)

    nt = vf.shape[0]
    out_pts = pts.clone()
    for t in range(nt):
        v_t = vf[t].permute(2, 0, 1)  # (2,H,W)
        disp = sample_image_on_coords(vg, v_t, out_pts.T[..., None])[..., 0].T
        out_pts = out_pts + disp / nt

    out_pts = (A[:2, :2] @ out_pts.T + A[:2, 2:3]).T
    return out_pts


def map_points_target_to_source(velocity_grid, velocity_field, affine_matrix, target_points):
    """Inverse map target points (y, x): inverse affine then backward diffeomorphic flow."""
    vg, vf, A, pts = _normalize_point_transform_inputs(velocity_grid, velocity_field, affine_matrix, target_points)

    A_inv = torch.linalg.inv(A)
    out_pts = (A_inv[:2, :2] @ pts.T + A_inv[:2, 2:3]).T

    nt = vf.shape[0]
    for t in range(nt - 1, -1, -1):
        minus_v_t = (-vf[t]).permute(2, 0, 1)  # (2,H,W)
        disp = sample_image_on_coords(vg, minus_v_t, out_pts.T[..., None])[..., 0].T
        out_pts = out_pts + disp / nt

    return out_pts



def as_numpy_1d(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x).reshape(-1)



def compute_alignment_metrics(df_src, df_tgt, x_src_new, y_src_new, x_src_lddmm, y_src_lddmm, x_tgt, y_tgt, cluster_col="leiden_harmony_refined"):
    """Compute nearest-neighbor label/distance metrics before vs after LDDMM.

    Returns per-cluster and overall match-rate summaries, plus arrays used by
    plotting cells.
    """
    src_prealign_xy = np.stack([as_numpy_1d(x_src_new), as_numpy_1d(y_src_new)], axis=1)
    src_lddmm_xy = np.stack([as_numpy_1d(x_src_lddmm), as_numpy_1d(y_src_lddmm)], axis=1)
    tgt_xy = np.stack([as_numpy_1d(x_tgt), as_numpy_1d(y_tgt)], axis=1)

    src_labels = df_src[cluster_col].astype(str).to_numpy()
    tgt_labels = df_tgt[cluster_col].astype(str).to_numpy()

    tgt_tree = KDTree(tgt_xy)
    dist_pre_to_tgt, nn_tgt_idx_pre = tgt_tree.query(src_prealign_xy, k=1)
    dist_ldd_to_tgt, nn_tgt_idx_ldd = tgt_tree.query(src_lddmm_xy, k=1)

    pred_tgt_label_pre = tgt_labels[nn_tgt_idx_pre]
    pred_tgt_label_ldd = tgt_labels[nn_tgt_idx_ldd]

    match_pre = pred_tgt_label_pre == src_labels
    match_ldd = pred_tgt_label_ldd == src_labels

    cluster_perf = (
        pd.DataFrame(
            {
                "cluster": src_labels,
                "match_prealign": match_pre.astype(float),
                "match_lddmm": match_ldd.astype(float),
                "dist_prealign": dist_pre_to_tgt,
                "dist_lddmm": dist_ldd_to_tgt,
            }
        )
        .groupby("cluster", as_index=False)
        .agg(
            n_cells=("cluster", "size"),
            prealign_match_rate=("match_prealign", "mean"),
            lddmm_match_rate=("match_lddmm", "mean"),
            prealign_mean_nn_dist=("dist_prealign", "mean"),
            lddmm_mean_nn_dist=("dist_lddmm", "mean"),
        )
    )
    cluster_perf["match_rate_gain"] = cluster_perf["lddmm_match_rate"] - cluster_perf["prealign_match_rate"]

    overall_pre = float(match_pre.mean())
    overall_ldd = float(match_ldd.mean())

    pre_tree = KDTree(src_prealign_xy)
    ldd_tree = KDTree(src_lddmm_xy)
    dist_tgt_to_pre_src, nn_src_idx_pre = pre_tree.query(tgt_xy, k=1)
    dist_tgt_to_ldd_src, nn_src_idx_ldd = ldd_tree.query(tgt_xy, k=1)

    return {
        "src_prealign_xy": src_prealign_xy,
        "src_lddmm_xy": src_lddmm_xy,
        "tgt_xy": tgt_xy,
        "cluster_perf": cluster_perf,
        "overall_pre": overall_pre,
        "overall_ldd": overall_ldd,
        "dist_tgt_to_pre_src": dist_tgt_to_pre_src,
        "dist_tgt_to_ldd_src": dist_tgt_to_ldd_src,
        "nn_src_idx_pre": nn_src_idx_pre,
        "nn_src_idx_ldd": nn_src_idx_ldd,
    }


def plot_alignment_overlays(
    src_prealign_xy,
    src_lddmm_xy,
    tgt_xy,
    *,
    point_size=1.0,
    alpha=0.10,
    figsize=(12, 6),
):
    """Plot source-vs-target overlays before and after spAlignDE."""
    fig, ax = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    ax[0].scatter(src_prealign_xy[:, 0], src_prealign_xy[:, 1], s=point_size, alpha=alpha, label="source (prealign)")
    ax[0].scatter(tgt_xy[:, 0], tgt_xy[:, 1], s=point_size, alpha=alpha, label="target")
    ax[0].set_title("Prealign Overlay")
    ax[0].legend(markerscale=10, loc="lower left")

    ax[1].scatter(src_lddmm_xy[:, 0], src_lddmm_xy[:, 1], s=point_size, alpha=alpha, label="source (spAlignDE)")
    ax[1].scatter(tgt_xy[:, 0], tgt_xy[:, 1], s=point_size, alpha=alpha, label="target")
    ax[1].set_title("spAlignDE Overlay")
    ax[1].legend(markerscale=10, loc="lower left")

    for a in ax:
        a.set_aspect("equal", adjustable="box")

    plt.show()
    return fig, ax


def plot_cluster_overlay_before_after(
    df_src,
    df_tgt,
    x_src_new,
    y_src_new,
    x_src_lddmm,
    y_src_lddmm,
    x_tgt,
    y_tgt,
    cluster_col="leiden_harmony_refined",
    *,
    point_size=1.0,
    target_alpha=0.35,
    source_alpha=0.55,
    figsize=(16, 7),
):
    """Color-coded cluster overlay QC before and after spAlignDE."""
    if cluster_col not in df_src.columns or cluster_col not in df_tgt.columns:
        raise ValueError(f"{cluster_col} must exist in both df_src and df_tgt")

    src_cluster = df_src[cluster_col].astype(str).to_numpy()
    tgt_cluster = df_tgt[cluster_col].astype(str).to_numpy()
    all_clusters = np.array(sorted(set(src_cluster).union(set(tgt_cluster))))
    cluster_to_id = {c: i for i, c in enumerate(all_clusters)}

    src_cid = np.array([cluster_to_id[c] for c in src_cluster])
    tgt_cid = np.array([cluster_to_id[c] for c in tgt_cluster])

    src_pre_xy = np.stack([as_numpy_1d(x_src_new), as_numpy_1d(y_src_new)], axis=1)
    src_ldd_xy = np.stack([as_numpy_1d(x_src_lddmm), as_numpy_1d(y_src_lddmm)], axis=1)
    tgt_xy = np.stack([as_numpy_1d(x_tgt), as_numpy_1d(y_tgt)], axis=1)

    fig, ax = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    cmap = plt.cm.get_cmap("tab20", max(len(all_clusters), 1))

    ax[0].scatter(tgt_xy[:, 0], tgt_xy[:, 1], c=tgt_cid, cmap=cmap, s=point_size, alpha=target_alpha, marker="o", label="target")
    ax[0].scatter(src_pre_xy[:, 0], src_pre_xy[:, 1], c=src_cid, cmap=cmap, s=point_size, alpha=source_alpha, marker=".", label="source prealign")
    ax[0].set_title(f"Cluster Overlay Before spAlignDE ({cluster_col})")
    ax[0].set_aspect("equal", adjustable="box")
    ax[0].legend(markerscale=6, loc="lower left")

    ax[1].scatter(tgt_xy[:, 0], tgt_xy[:, 1], c=tgt_cid, cmap=cmap, s=point_size, alpha=target_alpha, marker="o", label="target")
    ax[1].scatter(src_ldd_xy[:, 0], src_ldd_xy[:, 1], c=src_cid, cmap=cmap, s=point_size, alpha=source_alpha, marker=".", label="source spAlignDE")
    ax[1].set_title(f"Cluster Overlay After spAlignDE ({cluster_col})")
    ax[1].set_aspect("equal", adjustable="box")
    ax[1].legend(markerscale=6, loc="lower left")

    plt.show()
    return fig, ax


def build_source_output_table(df_src, src_pre_xy, src_ldd_xy):
    """Append prealign/LDDMM transformed coordinates to source table."""
    df_src_out = df_src.copy()
    df_src_out["x_prealign_to_target"] = src_pre_xy[:, 0]
    df_src_out["y_prealign_to_target"] = src_pre_xy[:, 1]
    df_src_out["x_lddmm_to_target"] = src_ldd_xy[:, 0]
    df_src_out["y_lddmm_to_target"] = src_ldd_xy[:, 1]
    return df_src_out


def save_alignment_output(df_src_out, out_csv: str):
    """Save aligned source table to CSV and print destination path."""
    df_src_out.to_csv(out_csv, index=True)
    print("Saved:")
    print(f"  - {out_csv} ({df_src_out.shape[0]} rows)")


def print_output_preview(df_src_out):
    print("Target output preview (coordinates only):")
    print(
        df_src_out[
            [
                "x",
                "y",
                "x_prealign_to_target",
                "y_prealign_to_target",
                "x_lddmm_to_target",
                "y_lddmm_to_target",
            ]
        ].head()
    )


def print_cluster_performance(cluster_perf: pd.DataFrame):
    """Print cluster-level match/distance summary sorted by gain."""
    print("Cluster performance summary:")
    print(cluster_perf.sort_values("cluster").to_string(index=False))


# =============================
# Eigenmode-based alignment uncertainty (point displacement version)
# =============================


def _to_torch_grid(grid_like, device, dtype):
    y = torch.as_tensor(grid_like[0], device=device, dtype=dtype)
    x = torch.as_tensor(grid_like[1], device=device, dtype=dtype)
    return [y, x]


def _sample_vector_at_points(grid_coords, vector_map, points_yx,
                             padding_mode='border', align_corners=True):
    """
    Sample a 2D vector field on a regular grid at point coordinates.

    Parameters
    ----------
    grid_coords : [y, x]
        1D grid coordinates
    vector_map : torch.Tensor, shape (H, W, 2)
        Vector field defined on the grid
    points_yx : array-like, shape (N, 2)
        Points in [y, x] order

    Returns
    -------
    vals : torch.Tensor, shape (N, 2)
        Sampled vectors at points
    """
    from torch.nn.functional import grid_sample

    y, x = [torch.as_tensor(t, device=vector_map.device, dtype=vector_map.dtype)
            for t in grid_coords]

    H, W = len(y), len(x)
    pts = torch.as_tensor(points_yx, device=vector_map.device, dtype=vector_map.dtype)

    gy = (pts[:, 0] - y[0]) / (y[-1] - y[0]) * 2.0 - 1.0
    gx = (pts[:, 1] - x[0]) / (x[-1] - x[0]) * 2.0 - 1.0
    grid = torch.stack((gx, gy), dim=-1).view(1, -1, 1, 2)  # (1, N, 1, 2)

    # grid_sample wants (N, C, H, W)
    F = vector_map.permute(2, 0, 1).unsqueeze(0)  # (1, 2, H, W)

    vals = grid_sample(
        F, grid,
        mode='bilinear',
        padding_mode=padding_mode,
        align_corners=align_corners,
    )  # (1, 2, N, 1)

    vals = vals.squeeze(0).squeeze(-1).T  # (N, 2)
    return vals


def _build_velocity_from_m0(m0, res, nt=None):
    """
    Re-shoot from a perturbed m0 to get velocity field list.
    Assumes your codebase already has build_kernel_K and geodesic_shooting.
    """
    device = m0.device
    dtype = m0.dtype

    xv = [xx.to(device=device, dtype=dtype) for xx in res['xv']]
    if nt is None:
        nt = int(res['v'].shape[0])

    dv = torch.as_tensor(
        [xv[0][1] - xv[0][0], xv[1][1] - xv[1][0]],
        device=device, dtype=dtype
    )

    # Keep this consistent with your registration setup
    K = build_kernel_K(xv, a=500.0, p=2.0).to(device=device, dtype=dtype)

    v_list, _ = geodesic_shooting(m0, xv, nt, K, dv)
    v = torch.stack(v_list, dim=0)  # (nt, H, W, 2)
    return v


def _compose_forward_map_from_m0(m0, res, nt=None):
    """
    Build forward spatial transform map on the source/velocity grid.

    Returns
    -------
    phi_fwd : torch.Tensor, shape (H, W, 2)
        Forward map: source-grid location -> target coordinate
    """
    device = m0.device
    dtype = m0.dtype

    xv = [xx.to(device=device, dtype=dtype) for xx in res['xv']]
    A = res['A'].to(device=device, dtype=dtype)

    v = _build_velocity_from_m0(m0, res, nt=nt)

    phi_fwd = compose_spatial_transform(
        velocity_grid=xv,
        velocity_field=v,
        affine_matrix=A,
        flow_direction='forward',
        target_grid=xv,   # evaluate forward map on source/velocity grid
    )
    return phi_fwd


def _warp_points_with_m0(points_xy, m0, res, nt=None):
    """
    Warp source points using m0.

    Parameters
    ----------
    points_xy : array-like, shape (N, 2)
        ORIGINAL source points in [x, y] order, on the same coordinate system
        as res['xv'] / source grid.
    m0 : torch.Tensor, shape (H, W, 2)
    res : dict

    Returns
    -------
    warped_xy : torch.Tensor, shape (N, 2)
        Warped points in target coordinates, [x, y] order
    """
    device = m0.device
    dtype = m0.dtype

    xv = [xx.to(device=device, dtype=dtype) for xx in res['xv']]
    phi_fwd = _compose_forward_map_from_m0(m0, res, nt=nt)  # (H, W, 2)

    points_xy = np.asarray(points_xy, dtype=float)
    points_yx = np.column_stack([points_xy[:, 1], points_xy[:, 0]])

    warped_yx = _sample_vector_at_points(
        grid_coords=xv,
        vector_map=phi_fwd,
        points_yx=points_yx,
        padding_mode='border',
        align_corners=True,
    )

    warped_xy = torch.stack([warped_yx[:, 1], warped_yx[:, 0]], dim=1)
    return warped_xy


def _make_mode_perturbation(eigval, eigvec, lambda_prior=1e-3,
                            alpha=0.5, m0_ref=None, max_rel_norm=0.25):
    """
    Build perturbation delta_m along one eigenmode.

    Base scale:
        delta_m = alpha * (1 / sqrt(eigval + lambda_prior)) * eigvec

    Then optionally clip ||delta_m|| relative to ||m0_ref|| for stability.
    """
    eigval = torch.as_tensor(eigval, device=eigvec.device, dtype=eigvec.dtype)
    mode_std = 1.0 / torch.sqrt(eigval + torch.as_tensor(lambda_prior, device=eigvec.device, dtype=eigvec.dtype))
    delta_m = alpha * mode_std * eigvec

    if m0_ref is not None:
        n_delta = torch.sqrt(torch.sum(delta_m ** 2) + 1e-12)
        n_m0 = torch.sqrt(torch.sum(m0_ref ** 2) + 1e-12)
        max_allowed = max_rel_norm * n_m0
        if n_delta > max_allowed:
            delta_m = delta_m * (max_allowed / n_delta)

    return delta_m


def compute_alignment_uncertainty_from_eigenmodes(
    points_xy,
    res,
    eigvals,
    eigvecs,
    lambda_prior=1e-3,
    alpha=0.5,
    top_k=5,
    nt=None,
    max_rel_norm=0.25,
):
    """
    Compute top-1 and top-k alignment uncertainty by perturbing m0 along Hessian eigenmodes.

    Parameters
    ----------
    points_xy : array-like, shape (N, 2)
        ORIGINAL source points [x, y] before alignment
    res : dict
        Must contain 'm0', 'A', 'xv', 'v'
    eigvals : torch.Tensor, shape (k,)
    eigvecs : torch.Tensor, shape (k, H, W, 2)
    lambda_prior : float
    alpha : float
        Perturbation strength multiplier
    top_k : int
        Number of leading modes to combine
    nt : int or None
        Number of shooting time steps; default uses len(res['v'])
    max_rel_norm : float
        Clip perturbation norm to this fraction of ||m0|| for stability

    Returns
    -------
    out : dict
        {
          'x0_xy': original aligned coordinates, (N, 2)
          'u_top1': top-1 displacement uncertainty, (N,)
          'u_topk': top-k displacement uncertainty, (N,)
          'weights_topk': normalized mode weights, (k_used,)
          'per_mode_rms': per-mode RMS displacement, (k_used, N)
          'x_plus_list': list of warped coords for + perturbation
          'x_minus_list': list of warped coords for - perturbation
        }
    """
    device = res['m0'].device
    dtype = res['m0'].dtype

    m0 = res['m0'].to(device=device, dtype=dtype)
    eigvals = eigvals.to(device=device, dtype=dtype)
    eigvecs = eigvecs.to(device=device, dtype=dtype)

    k_used = min(int(top_k), int(eigvecs.shape[0]))
    if nt is None:
        nt = int(res['v'].shape[0])

    # Original aligned positions
    x0_xy = _warp_points_with_m0(points_xy, m0, res, nt=nt)  # (N, 2)

    # Mode weights from local posterior precision
    raw_w = 1.0 / (eigvals[:k_used] + lambda_prior)
    weights = raw_w / torch.sum(raw_w)

    per_mode_rms = []
    x_plus_list = []
    x_minus_list = []

    for i in range(k_used):
        delta_m = _make_mode_perturbation(
            eigval=eigvals[i],
            eigvec=eigvecs[i],
            lambda_prior=lambda_prior,
            alpha=alpha,
            m0_ref=m0,
            max_rel_norm=max_rel_norm,
        )

        m0_plus = m0 + delta_m
        m0_minus = m0 - delta_m

        x_plus = _warp_points_with_m0(points_xy, m0_plus, res, nt=nt)
        x_minus = _warp_points_with_m0(points_xy, m0_minus, res, nt=nt)

        d_plus = torch.sqrt(torch.sum((x_plus - x0_xy) ** 2, dim=1) + 1e-12)
        d_minus = torch.sqrt(torch.sum((x_minus - x0_xy) ** 2, dim=1) + 1e-12)

        # Symmetric displacement summary for this mode
        d_rms = torch.sqrt(0.5 * (d_plus ** 2 + d_minus ** 2))

        per_mode_rms.append(d_rms)
        x_plus_list.append(x_plus)
        x_minus_list.append(x_minus)

    per_mode_rms = torch.stack(per_mode_rms, dim=0)  # (k_used, N)

    # Top-1 uncertainty
    u_top1 = per_mode_rms[0]

    # Top-k weighted RMS uncertainty
    u_topk = torch.sqrt(torch.sum(weights[:, None] * (per_mode_rms ** 2), dim=0))

    return {
        'x0_xy': x0_xy,
        'u_top1': u_top1,
        'u_topk': u_topk,
        'weights_topk': weights,
        'per_mode_rms': per_mode_rms,
        'x_plus_list': x_plus_list,
        'x_minus_list': x_minus_list,
    }


def plot_point_uncertainty(points_xy, values, target_xy=None,
                           s_src=2, s_tgt=1,
                           alpha_src=0.9, alpha_tgt=0.05,
                           clip_percentile=99.0,
                           title="Alignment uncertainty (point displacement)",
                           cmap="magma"):
    """
    Scatter plot of source points colored by point displacement uncertainty.
    """
    pts = np.asarray(points_xy, dtype=float)
    vals = values.detach().cpu().numpy() if torch.is_tensor(values) else np.asarray(values, dtype=float)

    if clip_percentile is not None:
        vmax = np.percentile(vals, clip_percentile)
        vals_plot = np.clip(vals, None, vmax)
    else:
        vals_plot = vals

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)

    if target_xy is not None:
        tgt = np.asarray(target_xy, dtype=float)
        ax.scatter(tgt[:, 0], tgt[:, 1], s=s_tgt, color='gray', alpha=alpha_tgt, label='target')

    sc = ax.scatter(
        pts[:, 0], pts[:, 1],
        c=vals_plot,
        s=s_src,
        alpha=alpha_src,
        cmap=cmap,
        label='aligned source',
    )

    ax.set_title(title)
    ax.set_aspect('equal')
    ax.legend(markerscale=6, loc='lower left')
    cbar = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)
    cbar.set_label("point displacement uncertainty")
    plt.show()


def compute_uncertainty_from_current_alignment(x_src_lddmm, y_src_lddmm, res, eigvals, eigvecs, lambda_prior=7e-4, alpha=0.8, top_k=5, max_rel_norm=0.25):
    source_xy = np.column_stack([as_numpy_1d(x_src_lddmm), as_numpy_1d(y_src_lddmm)])
    unc_pts = compute_alignment_uncertainty_from_eigenmodes(
        points_xy=source_xy,
        res=res,
        eigvals=eigvals,
        eigvecs=eigvecs,
        lambda_prior=lambda_prior,
        alpha=alpha,
        top_k=top_k,
        nt=int(res["v"].shape[0]),
        max_rel_norm=max_rel_norm,
    )
    x0_xy = unc_pts["x0_xy"]
    u_top1 = unc_pts["u_top1"]
    u_topk = unc_pts["u_topk"]
    return unc_pts, x0_xy, u_top1, u_topk
