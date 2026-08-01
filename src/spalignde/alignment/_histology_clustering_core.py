#!/usr/bin/env python3
"""Reusable H&E feature-clustering functions for ST-to-histology alignment.

It contains the deterministic image-feature clustering, symmetry merging, and
spatial-cleanup implementation used by the public histology wrapper.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

try:
    import torch
except Exception:  # pragma: no cover - required only for KMeans execution.
    torch = None

try:
    from PIL import Image
except Exception:  # pragma: no cover - required only for image/RGB proxy loading.
    Image = None

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional until a plot function is called.
    plt = None

if Image is not None:
    Image.MAX_IMAGE_PIXELS = None

POST_SYMMETRY_MERGE = True
POST_SYMMETRY_AXIS = "ud"
POST_MAX_MERGES = 2
POST_MIN_SCORE_GAIN = 0.02
POST_MIN_REFLECTED_DICE = 0.15
POST_MAX_REFLECTED_CENTROID_DIST = 0.15
POST_MIN_FEATURE_COSINE = 0.30

def _embedding_to_hwc(arr, name="embedding"):
    """Accept histology feature arrays in (C,H,W), (H,W,C), or list-of-(H,W) format."""
    if isinstance(arr, (list, tuple)):
        # Tutorial extraction saves each channel as one (H,W) array in a list.
        # Stacking on the last axis already gives the desired (H,W,C) layout.
        arr = np.stack(arr, axis=-1)
        return np.asarray(arr, dtype=np.float32)
    else:
        arr = np.asarray(arr)

    if arr.ndim != 3:
        raise ValueError(f"{name} must be 3D, got shape={arr.shape}")

    # Merged feature arrays are saved as (C,H,W); image features are (H,W,C).
    if arr.shape[0] <= 2048 and arr.shape[1] > 32 and arr.shape[2] > 32:
        arr = arr.transpose(1, 2, 0)

    return np.asarray(arr, dtype=np.float32)


def _load_patch_rgb_proxy(prefix_dir, h, w):
    """Downsample he.jpg to the embedding grid so slide selection can use brightness."""
    img_path = Path(prefix_dir) / "he.jpg"
    if not img_path.exists():
        return np.zeros((h, w, 0), dtype=np.float32)
    if Image is None:
        raise ImportError("Pillow is required to load he.jpg as an RGB proxy.")

    img = Image.open(img_path).convert("RGB")
    img = img.resize((w, h), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def load_he_embeddings(pickle_path, border_trim_px=256):
    """
    Load H&E features for clustering.

    Supported formats:
    - HIPT dict from {'cls', 'sub', 'rgb'} -> cls + sub + rgb
    - Integrated feature dict: {'vit', 'uni'} -> vit + uni + patch RGB from he.jpg
    - tensor pickle: (C,H,W) or (H,W,C)
    """
    pickle_path = Path(pickle_path)
    with open(pickle_path, "rb") as f:
        embs = pickle.load(f)

    feature_format = "unknown"
    if isinstance(embs, dict) and all(k in embs for k in ("cls", "sub", "rgb")):
        cls = _embedding_to_hwc(embs["cls"], "cls")
        sub = _embedding_to_hwc(embs["sub"], "sub")
        rgb = _embedding_to_hwc(embs["rgb"], "rgb")
        feature_format = "hipt_vit"
    elif isinstance(embs, dict) and all(k in embs for k in ("vit", "uni")):
        cls = _embedding_to_hwc(embs["vit"], "vit")
        sub = _embedding_to_hwc(embs["uni"], "uni")
        if cls.shape[:2] != sub.shape[:2]:
            raise ValueError(f"vit and uni grids differ: vit={cls.shape}, uni={sub.shape}")
        rgb = _load_patch_rgb_proxy(pickle_path.parent, cls.shape[0], cls.shape[1])
        feature_format = "merged_vit_uni"
    else:
        features0 = _embedding_to_hwc(embs, "embedding")
        h0, w0, _ = features0.shape
        cls = features0
        sub = np.zeros((h0, w0, 0), dtype=np.float32)
        rgb = _load_patch_rgb_proxy(pickle_path.parent, h0, w0)
        if rgb.shape[-1] == 0:
            rgb = features0[..., : min(3, features0.shape[-1])]
        feature_format = "tensor"

    if rgb.shape[-1] > 0 and rgb.shape[:2] != cls.shape[:2]:
        raise ValueError(f"rgb grid differs from feature grid: rgb={rgb.shape}, cls={cls.shape}")
    if sub.shape[:2] != cls.shape[:2]:
        raise ValueError(f"sub/uni grid differs from feature grid: sub={sub.shape}, cls={cls.shape}")

    features = np.concatenate([cls, sub, rgb], axis=-1).astype(np.float32, copy=False)
    h, w, c = features.shape
    border_trim = max(1, int(border_trim_px) // 16)

    mask0 = np.isfinite(features).all(axis=-1)
    mask = mask0.copy()
    mask[:border_trim, :] = False
    mask[-border_trim:, :] = False
    mask[:, :border_trim] = False
    mask[:, -border_trim:] = False

    x_raw = features[mask].reshape(-1, c)

    return {
        "embs": embs,
        "feature_format": feature_format,
        "cls": cls,
        "sub": sub,
        "rgb": rgb,
        "features": features,
        "h": h,
        "w": w,
        "c": c,
        "mask": mask,
        "x_raw": x_raw,
        "border_trim": border_trim,
        "has_rgb": rgb.shape[-1] > 0,
        "block_dims": {"vit_or_cls": cls.shape[-1], "uni_or_sub": sub.shape[-1], "rgb": rgb.shape[-1]},
    }


def kmeans_torch(x_t, k, n_iters, init_samples, assign_chunk, rng):
    if torch is None:
        raise ImportError("PyTorch is required for H&E feature KMeans clustering.")
    ns = min(init_samples, x_t.shape[0])
    init_idx = rng.choice(x_t.shape[0], size=ns, replace=False)[:k]
    centers = x_t[init_idx].clone()

    for _ in range(n_iters):
        sums = torch.zeros((k, x_t.shape[1]), device=x_t.device)
        counts = torch.zeros(k, device=x_t.device)

        for start in range(0, x_t.shape[0], assign_chunk):
            end = min(start + assign_chunk, x_t.shape[0])
            chunk = x_t[start:end]
            dists = torch.cdist(chunk, centers)
            labels = torch.argmin(dists, dim=1)
            sums.index_add_(0, labels, chunk)
            counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float))

        counts = torch.clamp(counts, min=1.0)
        centers = sums / counts[:, None]

    labels_all = torch.empty((x_t.shape[0],), device=x_t.device, dtype=torch.int32)
    for start in range(0, x_t.shape[0], assign_chunk):
        end = min(start + assign_chunk, x_t.shape[0])
        chunk = x_t[start:end]
        dists = torch.cdist(chunk, centers)
        labels_all[start:end] = torch.argmin(dists, dim=1).to(torch.int32)

    return labels_all.cpu().numpy(), centers.cpu().numpy()


def _pair_lr_reflections(labels_img, bg_val=-1, min_area=200, min_iou=0.20, min_dice=0.30, verbose=True, axis="lr"):
    labels_img = np.asarray(labels_img)
    clusters = [int(c) for c in np.unique(labels_img) if c != bg_val]
    masks = {c: (labels_img == c) for c in clusters}
    masks = {c: m for c, m in masks.items() if m.sum() >= min_area}

    ids = list(masks.keys())
    axis = str(axis).lower()
    if axis in ("lr", "left-right", "left_right", "x", "horizontal"):
        flip_masks = {c: np.fliplr(m) for c, m in masks.items()}
        axis_label = "LR"
    elif axis in ("ud", "tb", "top-bottom", "top_bottom", "y", "vertical"):
        flip_masks = {c: np.flipud(m) for c, m in masks.items()}
        axis_label = "UD"
    else:
        raise ValueError(f"Unsupported reflection axis: {axis}. Use 'lr' or 'ud'.")

    pairs = []
    used = set()
    for c in ids:
        if c in used:
            continue
        best = None
        for d in ids:
            if c == d or d in used:
                continue
            inter = np.logical_and(flip_masks[c], masks[d]).sum()
            if inter == 0:
                continue
            union = np.logical_or(flip_masks[c], masks[d]).sum()
            iou = inter / (union + 1e-12)
            dice = 2 * inter / (flip_masks[c].sum() + masks[d].sum() + 1e-12)
            if (best is None) or (iou > best[0]):
                best = (iou, dice, d)
        if best and best[0] >= min_iou and best[1] >= min_dice:
            pairs.append((c, best[2], best[0], best[1]))
            used.add(c)
            used.add(best[2])

    mapping = {c: c for c in np.unique(labels_img) if c != bg_val}
    for a, b, _, _ in pairs:
        new_id = min(a, b)
        mapping[a] = new_id
        mapping[b] = new_id

    if verbose:
        print(f"{axis_label} reflection pairs (a,b,iou,dice):")
        for a, b, iou, dice in pairs:
            print(f"  {a} <-> {b} | IoU={iou:.3f} Dice={dice:.3f}")

    return pairs, mapping


def _apply_mapping(labels_img, mapping, bg_val=-1, compact=True):
    out = labels_img.copy()
    m = out != bg_val
    vals = out[m].astype(np.int64)
    mapped = np.vectorize(lambda x: mapping.get(int(x), int(x)))(vals)
    out[m] = mapped

    if compact:
        uniq = sorted([int(c) for c in np.unique(out) if c != bg_val])
        remap = {c: i for i, c in enumerate(uniq)}
        out[m] = np.vectorize(lambda x: remap[int(x)])(out[m])
        return out, remap

    return out, None


def merge_he_clusters(
    labels_full,
    slide_x_raw,
    labels_slide,
    k_slide,
    k_merge=20,
    do_reflection_merge=True,
    reflection_min_area=200,
    reflection_min_iou=0.20,
    reflection_min_dice=0.30,
    reflection_axis="lr",
):
    """Merge stage-2 clusters outside main pipeline so k_merge can be tuned quickly."""
    from sklearn.cluster import AgglomerativeClustering

    C = slide_x_raw.shape[1]
    centroids = np.zeros((k_slide, C), dtype=np.float64)
    counts = np.zeros(k_slide, dtype=np.int64)

    for j in range(k_slide):
        sel = labels_slide == j
        counts[j] = sel.sum()
        if counts[j] > 0:
            centroids[j] = slide_x_raw[sel].mean(axis=0)
        else:
            centroids[j] = np.nan

    valid = np.isfinite(centroids).all(axis=1)
    centroids_valid = centroids[valid]
    valid_ids = np.nonzero(valid)[0]

    agg = AgglomerativeClustering(n_clusters=k_merge, linkage="ward")
    meta_labels = agg.fit_predict(centroids_valid)

    map_slide_to_merge = np.full(k_slide, -1, dtype=np.int32)
    map_slide_to_merge[valid_ids] = meta_labels.astype(np.int32)

    labels_merged = labels_full.copy()
    m = labels_merged >= 0
    labels_merged[m] = map_slide_to_merge[labels_merged[m]]

    pairs_lr, map_lr, remap_lr = [], None, None
    if do_reflection_merge:
        pairs_lr, map_lr = _pair_lr_reflections(
            labels_merged,
            bg_val=-1,
            min_area=reflection_min_area,
            min_iou=reflection_min_iou,
            min_dice=reflection_min_dice,
            verbose=True,
            axis=reflection_axis,
        )
        labels_merged, remap_lr = _apply_mapping(labels_merged, map_lr, bg_val=-1, compact=True)

    return {
        "labels_merged": labels_merged,
        "map_slide_to_merge": map_slide_to_merge,
        "centroids": centroids,
        "counts": counts,
        "pairs_lr": pairs_lr,
        "map_lr": map_lr,
        "remap_lr": remap_lr,
    }


def run_he_clustering_pipeline(
    pickle_path,
    out_dir,
    k_bg=3,
    k_slide=21,
    k_merge=20,
    apply_merge=False,
    n_iters=10,
    init_samples=50000,
    assign_chunk=50000,
    border_trim_px=256,
    rgb_weight=0.25,
    xy_weight=0.3,
    smooth_sigma=0.8,
    random_state=0,
    max_fit=200000,
    do_reflection_merge=True,
    reflection_min_area=200,
    reflection_min_iou=0.20,
    reflection_min_dice=0.30,
    reflection_axis="lr",
    fast_mode=False,
    cpu_threads=None,
    pca_max_components=48,
    gmm_n_init=3,
    gmm_max_iter=200,
    gmm_covariance_type="tied",
    gmm_reg_covar=1e-4,
    proba_chunk=200000,
    keep_largest_tissue_cc=True,
    min_tissue_cc_area=0,
    min_slide_fraction=0.01,
    stage1_use_rgb_only=True,
    debug_mask=True,
):
    """
    Run full HE embedding clustering + smoothing pipeline.

    Required inputs:
        pickle_path: str
            Path to HIPT dict, merged spEnhance dict {'vit', 'uni'}, or tensor pickle.
        out_dir: str
            Folder to save quick outputs (e.g., slide_mask_k2.png).

    Main clustering controls:
        k_bg: int
            Number of clusters for stage-1 background/slide split.
        k_slide: int
            Number of KMeans clusters on slide pixels (set to 21 for embedding clustering).
        k_merge: int
            Number of clusters if apply_merge=True.
        apply_merge: bool
            If False, skip merge in pipeline; run merge_he_clusters(...) later.

    Smoothing and feature weighting:
        border_trim_px: int
            Border (in original pixels) removed before clustering.
        rgb_weight: float
            Weight for RGB block before KMeans.
        xy_weight: float
            Weight for spatial coordinates before KMeans.
        smooth_sigma: float
            Unused in KMeans mode (kept for compatibility).

    Fast mode:
        fast_mode: bool
            If True, uses a faster/cheaper configuration (lower sample/iter counts).
        cpu_threads: int or None
            Torch CPU threads. None = auto.

    Stage-1 mask robustness:
        min_slide_fraction: float
            If selected slide cluster is smaller than this fraction, auto-fallback to another cluster.
        stage1_use_rgb_only: bool
            If True and RGB is available, use RGB/brightness only for stage-1 slide/background selection.
            Stage-2 still clusters the combined VIT/UNI/RGB feature matrix.
        debug_mask: bool
            Print stage-1 cluster diagnostics (counts/brightness/final choice).

    Background cleanup:
        keep_largest_tissue_cc: bool
            If True, keeps only the largest connected tissue component in final labels.
        min_tissue_cc_area: int
            Minimum component area (pixels) to keep before selecting largest CC.

    Reflection merge (optional):
        do_reflection_merge: bool
            If True, merges left-right reflected clusters.
        reflection_min_area/min_iou/min_dice:
            Thresholds for reflection pairing.

    Returns:
        dict with key outputs:
            labels_full: (H, W) int32, stage-2 labels on slide, -1 background.
            labels_merged: (H, W) int32, final labels (or merged if apply_merge=True).
            slide_mask: (H, W) bool, detected slide region.
            labels_slide: (N_slide,) int32, slide-only labels.
            vit/uni-or-cls/sub/rgb/features/x_raw/... plus intermediate objects for debugging.
    """
    from scipy import ndimage

    if fast_mode:
        n_iters = min(n_iters, 6)
        init_samples = min(init_samples, 20000)
        assign_chunk = max(assign_chunk, 100000)

    os.makedirs(out_dir, exist_ok=True)
    pack = load_he_embeddings(pickle_path=pickle_path, border_trim_px=border_trim_px)

    embs = pack["embs"]
    cls = pack["cls"]
    sub = pack["sub"]
    rgb = pack["rgb"]
    features = pack["features"]
    feature_format = pack.get("feature_format", "unknown")
    block_dims = pack.get("block_dims", {})
    h, w, c = pack["h"], pack["w"], pack["c"]
    mask = pack["mask"]
    x_raw = pack["x_raw"]
    border_trim = pack["border_trim"]

    rgb_start = cls.shape[-1] + sub.shape[-1]
    rgb_end = rgb_start + rgb.shape[-1]
    has_rgb = bool(pack.get("has_rgb", rgb.shape[-1] > 0))

    # Stage-1 should identify tissue/background, not high-dimensional morphology outliers.
    # For merged features, RGB from he.jpg is the most stable brightness signal.
    if bool(stage1_use_rgb_only) and has_rgb and rgb_end > rgb_start:
        stage1_raw = x_raw[:, rgb_start:rgb_end]
        stage1_source = "rgb_proxy"
    else:
        stage1_raw = x_raw
        stage1_source = "all_features"

    mean = stage1_raw.mean(axis=0)
    std = stage1_raw.std(axis=0)
    std[std == 0] = 1.0
    x = (stage1_raw - mean) / std

    device = torch.device("cpu")
    if cpu_threads is None:
        cpu_threads = max(1, (os.cpu_count() or 4) // 2)
    torch.set_num_threads(int(cpu_threads))
    x_t = torch.from_numpy(x.astype(np.float32, copy=False)).to(device)
    rng = np.random.default_rng(random_state)

    labels_bg, centers_bg = kmeans_torch(
        x_t=x_t,
        k=k_bg,
        n_iters=n_iters,
        init_samples=init_samples,
        assign_chunk=assign_chunk,
        rng=rng,
    )

    counts_bg = np.zeros(k_bg, dtype=np.int64)
    mean_rgb_by_cluster = np.full(k_bg, np.inf, dtype=np.float64)
    for i in range(k_bg):
        sel = labels_bg == i
        counts_bg[i] = int(sel.sum())
        if counts_bg[i] > 0:
            if has_rgb and rgb_end > rgb_start:
                mean_rgb_by_cluster[i] = float(np.nanmean(x_raw[sel, rgb_start:rgb_end]))
            else:
                mean_rgb_by_cluster[i] = float(np.nanmean(x_raw[sel]))

    # Primary rule: tissue tends to be darker => lower mean RGB.
    # Ignore tiny dark outliers when a larger candidate exists.
    n_valid = max(1, int(labels_bg.size))
    frac = counts_bg / n_valid
    large_enough = np.flatnonzero(frac >= float(min_slide_fraction))
    if large_enough.size > 0:
        slide_cluster = int(large_enough[np.argmin(mean_rgb_by_cluster[large_enough])])
    else:
        slide_cluster = int(np.argmax(counts_bg))

    # Final hard guard: if still empty, choose largest non-empty cluster
    if counts_bg[slide_cluster] == 0:
        slide_cluster = int(np.argmax(counts_bg))

    if debug_mask:
        print("[Embedding] format:", feature_format, "shape:", features.shape, "blocks:", block_dims)
        print("[Stage1] source:", stage1_source)
        print('[Stage1-k2] counts:', counts_bg.tolist())
        print('[Stage1-k2] mean_rgb:', mean_rgb_by_cluster.tolist())
        print('[Stage1-k2] frac:', frac.tolist())
        print('[Stage1-k2] selected slide_cluster:', slide_cluster)

    slide_mask_flat = labels_bg == slide_cluster

    slide_mask = np.zeros(h * w, dtype=bool)
    slide_mask[mask.ravel()] = slide_mask_flat
    slide_mask = slide_mask.reshape(h, w)

    slide_mask[:border_trim, :] = False
    slide_mask[-border_trim:, :] = False
    slide_mask[:, :border_trim] = False
    slide_mask[:, -border_trim:] = False

    # Enforce slide-only pixels for stage-2 clustering (largest connected component)
    if keep_largest_tissue_cc:
        lab0, n0 = ndimage.label(slide_mask)
        if n0 > 0:
            counts0 = np.bincount(lab0.ravel())
            counts0[0] = 0
            best0 = int(np.argmax(counts0))
            slide_mask = lab0 == best0

    # Map cleaned slide mask back to flattened valid-pixel order used by x_raw
    flat_valid = np.flatnonzero(mask.ravel())
    slide_mask_flat = slide_mask.ravel()[flat_valid]

    slide_mask_path = os.path.join(out_dir, "slide_mask_k2.png")
    plt.figure(figsize=(6, 5))
    plt.imshow(slide_mask, cmap="gray")
    plt.title("Slide mask (k=2)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(slide_mask_path, dpi=150)
    plt.close()

    slide_x_raw = x_raw[slide_mask_flat].astype(np.float64)

    cls_dim = cls.shape[-1]
    sub_dim = sub.shape[-1]

    def z_block(xx):
        m = xx.mean(axis=0)
        s = xx.std(axis=0)
        s[s == 0] = 1.0
        return (xx - m) / s

    cls_z = z_block(slide_x_raw[:, :cls_dim])
    sub_z = z_block(slide_x_raw[:, cls_dim:cls_dim + sub_dim])
    rgb_z = z_block(slide_x_raw[:, cls_dim + sub_dim:])
    rgb_z *= float(rgb_weight)

    slide_x = np.concatenate([cls_z, sub_z, rgb_z], axis=1)

    flat_idx = np.flatnonzero(mask.ravel())
    rows = flat_idx // w
    cols = flat_idx % w
    slide_rows = rows[slide_mask_flat].astype(np.float32)
    slide_cols = cols[slide_mask_flat].astype(np.float32)

    xy = np.stack([slide_cols, slide_rows], axis=1)
    xy = (xy - xy.min(axis=0)) / (xy.max(axis=0) - xy.min(axis=0) + 1e-8)
    slide_x = np.concatenate([slide_x, float(xy_weight) * xy], axis=1)

    # Stage-2: pure KMeans on slide embedding features (k=21 by default).
    k_slide_eff = max(1, min(int(k_slide), int(slide_x.shape[0])))
    if k_slide_eff != int(k_slide) and debug_mask:
        print(f"[Stage2-kmeans] adjusted k_slide from {k_slide} to {k_slide_eff} (limited by N slide pixels)")
    k_slide = k_slide_eff

    slide_x_t = torch.from_numpy(slide_x.astype(np.float32)).to(device)
    labels_slide, centers_slide = kmeans_torch(
        x_t=slide_x_t,
        k=k_slide,
        n_iters=n_iters,
        init_samples=init_samples,
        assign_chunk=assign_chunk,
        rng=rng,
    )

    labels_full = np.full((h, w), -1, dtype=np.int32)
    labels_full[slide_mask] = labels_slide.astype(np.int32)
    labels_slide = labels_full[slide_mask].astype(np.int32)

    Xp = slide_x
    proba = None
    prob_maps = None

    if apply_merge:
        merged_pack = merge_he_clusters(
            labels_full=labels_full,
            slide_x_raw=slide_x_raw,
            labels_slide=labels_slide,
            k_slide=k_slide,
            k_merge=k_merge,
            do_reflection_merge=do_reflection_merge,
            reflection_min_area=reflection_min_area,
            reflection_min_iou=reflection_min_iou,
            reflection_min_dice=reflection_min_dice,
            reflection_axis=reflection_axis,
        )
        labels_merged = merged_pack["labels_merged"]
        map_slide_to_merge = merged_pack["map_slide_to_merge"]
        centroids = merged_pack["centroids"]
        counts = merged_pack["counts"]
        pairs_lr = merged_pack["pairs_lr"]
        map_lr = merged_pack["map_lr"]
        remap_lr = merged_pack["remap_lr"]
    else:
        labels_merged = labels_full.copy()
        map_slide_to_merge = None
        centroids = None
        counts = None
        pairs_lr, map_lr, remap_lr = [], None, None

    # Optional final background cleanup: keep only largest tissue CC
    if keep_largest_tissue_cc:
        tissue = labels_merged != -1
        lab, n = ndimage.label(tissue)
        if n > 0:
            counts = np.bincount(lab.ravel())
            counts[0] = 0
            if min_tissue_cc_area > 0:
                valid_ids = np.where(counts >= int(min_tissue_cc_area))[0]
                valid_ids = valid_ids[valid_ids != 0]
                if valid_ids.size > 0:
                    best_id = int(valid_ids[np.argmax(counts[valid_ids])])
                else:
                    best_id = int(np.argmax(counts))
            else:
                best_id = int(np.argmax(counts))

            keep = lab == best_id
            labels_merged[~keep] = -1
            labels_full[~keep] = -1
            slide_mask = keep
            labels_slide = labels_full[slide_mask].astype(np.int32)

    # Convenience views for plotting: background removed (NaN)
    labels_full_nobg = labels_full.astype(np.float32)
    labels_full_nobg[labels_full_nobg < 0] = np.nan

    labels_merged_nobg = labels_merged.astype(np.float32)
    labels_merged_nobg[labels_merged_nobg < 0] = np.nan

    return {
        "embs": embs,
        "feature_format": feature_format,
        "block_dims": block_dims,
        "cls": cls,
        "sub": sub,
        "rgb": rgb,
        "features": features,
        "h": h,
        "w": w,
        "c": c,
        "mask": mask,
        "x_raw": x_raw,
        "slide_mask": slide_mask,
        "slide_mask_flat": slide_mask_flat,
        "slide_x_raw": slide_x_raw,
        "labels_bg": labels_bg,
        "centers_bg": centers_bg,
        "labels_full": labels_full,
        "labels_slide": labels_slide,
        "labels_merged": labels_merged,
        "labels_full_nobg": labels_full_nobg,
        "labels_merged_nobg": labels_merged_nobg,
        "pairs_lr": pairs_lr,
        "map_lr": map_lr,
        "remap_lr": remap_lr,
        "map_slide_to_merge": map_slide_to_merge,
        "centroids": centroids,
        "counts": counts,
        "Xp": Xp,
        "proba": proba,
        "prob_maps": prob_maps,
        "k_bg": k_bg,
        "k_slide": k_slide,
        "k_merge": k_merge,
        "border_trim_px": border_trim_px,
        "border_trim": border_trim,
        "out_dir": out_dir,
        "pickle_path": pickle_path,
        "fast_mode": fast_mode,
        "cpu_threads": cpu_threads,
        "pca_max_components": pca_max_components,
        "gmm_n_init": gmm_n_init,
        "gmm_max_iter": gmm_max_iter,
        "gmm_covariance_type": gmm_covariance_type,
        "gmm_reg_covar": gmm_reg_covar,
        "proba_chunk": proba_chunk,
        "keep_largest_tissue_cc": keep_largest_tissue_cc,
        "min_tissue_cc_area": min_tissue_cc_area,
        "apply_merge": apply_merge,
        "min_slide_fraction": min_slide_fraction,
        "stage1_use_rgb_only": stage1_use_rgb_only,
        "debug_mask": debug_mask,
    }


def run_he_kmeans_crisp(
    pickle_path,
    out_dir,
    k_bg=2,
    k_slide=21,
    n_iters=10,
    init_samples=50000,
    assign_chunk=50000,
    border_trim_px=256,
    rgb_weight=0.25,
    xy_weight=0.05,
    random_state=0,
    max_fit=200000,
    fast_mode=False,
    cpu_threads=None,
    keep_largest_tissue_cc=True,
    min_tissue_cc_area=0,
    min_slide_fraction=0.01,
    stage1_use_rgb_only=True,
    debug_mask=True,
    reflection_axis="lr",
):
    """KMeans-only HE clustering with sharp boundaries (no smoothing/merge by default)."""
    return run_he_clustering_pipeline(
        pickle_path=pickle_path,
        out_dir=out_dir,
        k_bg=k_bg,
        k_slide=k_slide,
        k_merge=k_slide,
        apply_merge=False,
        n_iters=n_iters,
        init_samples=init_samples,
        assign_chunk=assign_chunk,
        border_trim_px=border_trim_px,
        rgb_weight=rgb_weight,
        xy_weight=xy_weight,
        smooth_sigma=0.0,
        random_state=random_state,
        max_fit=max_fit,
        do_reflection_merge=False,
        reflection_min_area=200,
        reflection_min_iou=0.20,
        reflection_min_dice=0.30,
        reflection_axis=reflection_axis,
        fast_mode=fast_mode,
        cpu_threads=cpu_threads,
        keep_largest_tissue_cc=keep_largest_tissue_cc,
        min_tissue_cc_area=min_tissue_cc_area,
        min_slide_fraction=min_slide_fraction,
        stage1_use_rgb_only=stage1_use_rgb_only,
        debug_mask=debug_mask,
    )


def fill_internal_label_holes(labels, bg=-1):
    labels = np.asarray(labels)
    filled = labels.copy()
    valid = filled != bg
    tissue = ndimage.binary_fill_holes(valid)
    holes = tissue & ~valid
    if holes.any() and valid.any():
        _, inds = ndimage.distance_transform_edt(~valid, return_indices=True)
        filled[holes] = filled[inds[0][holes], inds[1][holes]]
    return filled.astype(labels.dtype), holes


def compact_labels(labels, bg=-1):
    out = labels.copy()
    ids = sorted(int(x) for x in np.unique(out) if int(x) != bg)
    remap = {old: new for new, old in enumerate(ids)}
    for old, new in remap.items():
        out[labels == old] = new
    return out, remap


def symmetry_overlap_score(labels, axis="ud", bg=-1):
    if axis == "ud":
        ref = np.flipud(labels)
    elif axis == "lr":
        ref = np.fliplr(labels)
    else:
        raise ValueError("axis must be 'ud' or 'lr'")
    valid = (labels != bg) & (ref != bg)
    if not valid.any():
        return 0.0
    return float(np.mean(labels[valid] == ref[valid]))


def cluster_feature_centroids(labels_img, feature_matrix, bg=-1, feature_mask=None):
    if feature_mask is None:
        feature_mask = labels_img >= 0
    label_vec = labels_img[feature_mask].ravel().astype(np.int32)
    if len(label_vec) != feature_matrix.shape[0]:
        raise ValueError(f'feature rows ({feature_matrix.shape[0]}) do not match labeled pixels ({len(label_vec)})')
    centroids = {}
    for lab in sorted(int(x) for x in np.unique(label_vec) if int(x) != bg):
        centroids[lab] = feature_matrix[label_vec == lab].mean(axis=0).astype(np.float64)
    return centroids


def candidate_stats(labels, feature_centroids, a, b, axis="ud", bg=-1):
    H, W = labels.shape
    diag = float(np.hypot(H, W))
    ma = labels == a
    mb = labels == b
    if not ma.any() or not mb.any():
        return None

    ya, xa = np.where(ma)
    yb, xb = np.where(mb)
    ca = np.array([ya.mean(), xa.mean()], dtype=np.float64)
    cb = np.array([yb.mean(), xb.mean()], dtype=np.float64)

    if axis == "ud":
        ma_ref = np.flipud(ma)
        ca_ref = np.array([H - 1 - ca[0], ca[1]], dtype=np.float64)
    elif axis == "lr":
        ma_ref = np.fliplr(ma)
        ca_ref = np.array([ca[0], W - 1 - ca[1]], dtype=np.float64)
    else:
        raise ValueError("axis must be 'ud' or 'lr'")

    inter = int(np.logical_and(ma_ref, mb).sum())
    dice = 2.0 * inter / max(int(ma.sum() + mb.sum()), 1)
    dist = float(np.linalg.norm(ca_ref - cb) / diag)

    fa = feature_centroids[a]
    fb = feature_centroids[b]
    feature_cosine = float(np.dot(fa, fb) / ((np.linalg.norm(fa) + 1e-12) * (np.linalg.norm(fb) + 1e-12)))
    return {
        "a": int(a),
        "b": int(b),
        "reflected_dice": float(dice),
        "reflected_centroid_dist": float(dist),
        "feature_cosine": feature_cosine,
    }


def merge_pair(labels, a, b, bg=-1):
    out = labels.copy()
    keep = min(int(a), int(b))
    drop = max(int(a), int(b))
    out[out == drop] = keep
    out, remap = compact_labels(out, bg=bg)
    return out, remap


def post_merge_by_symmetry_score(labels, feature_matrix, axis="ud", bg=-1, feature_mask=None):
    current = labels.copy()
    history = []

    for step in range(POST_MAX_MERGES):
        base_score = symmetry_overlap_score(current, axis=axis, bg=bg)
        feature_centroids = cluster_feature_centroids(current, feature_matrix, bg=bg, feature_mask=feature_mask)
        ids = sorted(int(x) for x in np.unique(current) if int(x) != bg)
        best = None

        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                stats = candidate_stats(current, feature_centroids, a, b, axis=axis, bg=bg)
                if stats is None:
                    continue
                close_enough = (
                    stats["reflected_dice"] >= POST_MIN_REFLECTED_DICE
                    or stats["reflected_centroid_dist"] <= POST_MAX_REFLECTED_CENTROID_DIST
                )
                if not close_enough:
                    continue
                if stats["feature_cosine"] < POST_MIN_FEATURE_COSINE:
                    continue

                trial, _ = merge_pair(current, a, b, bg=bg)
                trial_score = symmetry_overlap_score(trial, axis=axis, bg=bg)
                gain = trial_score - base_score
                if gain < POST_MIN_SCORE_GAIN:
                    continue

                record = dict(stats)
                record["step"] = int(step + 1)
                record["score_before"] = float(base_score)
                record["score_after"] = float(trial_score)
                record["score_gain"] = float(gain)
                if best is None or record["score_gain"] > best["score_gain"]:
                    best = record

        if best is None:
            break

        current, remap = merge_pair(current, best["a"], best["b"], bg=bg)
        best["compact_remap"] = remap
        history.append(best)

    return current, history


def cleanup_label_islands(labels, bg=-1, min_size=250, max_iter=6):
    labels = np.asarray(labels)
    cleaned = labels.copy()
    valid = cleaned != bg
    removed = np.zeros(cleaned.shape, dtype=bool)

    for lab in [int(x) for x in np.unique(cleaned) if int(x) != bg]:
        mask = cleaned == lab
        cc, n = ndimage.label(mask)
        if n == 0:
            continue
        counts = np.bincount(cc.ravel())
        keep = np.zeros_like(mask, dtype=bool)
        for comp_id in range(1, len(counts)):
            if counts[comp_id] >= min_size:
                keep |= cc == comp_id
        small = mask & ~keep
        cleaned[small] = bg
        removed |= small

    # Iteratively fill removed pixels from neighboring labels.
    structure = np.ones((3, 3), dtype=bool)
    for _ in range(max_iter):
        holes = (cleaned == bg) & valid
        if not holes.any():
            break
        fill = cleaned.copy()
        ys, xs = np.where(holes)
        changed = 0
        for y, x in zip(ys, xs):
            y0, y1 = max(0, y - 1), min(cleaned.shape[0], y + 2)
            x0, x1 = max(0, x - 1), min(cleaned.shape[1], x + 2)
            neigh = cleaned[y0:y1, x0:x1]
            vals = neigh[neigh != bg]
            if vals.size:
                labs, cnts = np.unique(vals, return_counts=True)
                fill[y, x] = labs[np.argmax(cnts)]
                changed += 1
        cleaned = fill
        if changed == 0:
            break

    # Any remaining holes get nearest non-background label.
    holes = (cleaned == bg) & valid
    if holes.any():
        _, inds = ndimage.distance_transform_edt(cleaned == bg, return_indices=True)
        nearest = cleaned[tuple(inds)]
        cleaned[holes] = nearest[holes]

    return cleaned
