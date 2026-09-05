"""Dense shifted-view HIPT feature extraction used by spAlignDE.

The model architecture and checkpoints are loaded from an official HIPT clone.
This module owns the spAlignDE-specific dense stitching, shifted-view averaging,
smoothing, RGB downsampling, and output layout.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from time import time

import cv2 as cv
from einops import rearrange, reduce, repeat
import numpy as np
from PIL import Image
import torch


Image.MAX_IMAGE_PIXELS = None


def _load_hipt_classes(hipt_dir: Path):
    hipt_dir = hipt_dir.expanduser().resolve()
    sys.path.insert(0, str(hipt_dir))
    from hipt_4k import HIPT_4K  # type: ignore[import-not-found]
    from hipt_model_utils import eval_transforms  # type: ignore[import-not-found]

    return HIPT_4K, eval_transforms


def _patchify(image: np.ndarray, patch_size: int):
    original_hw = np.asarray(image.shape[:2], dtype=int)
    padded_hw = (original_hw + patch_size - 1) // patch_size * patch_size
    padded = np.pad(
        image,
        (
            (0, int(padded_hw[0] - original_hw[0])),
            (0, int(padded_hw[1] - original_hw[1])),
            (0, 0),
        ),
        mode="edge",
    )
    tile_grid = np.asarray(padded.shape[:2], dtype=int) // int(patch_size)
    tiles = []
    for row in range(int(tile_grid[0])):
        row_start = row * patch_size
        for column in range(int(tile_grid[1])):
            column_start = column * patch_size
            tiles.append(
                padded[
                    row_start : row_start + patch_size,
                    column_start : column_start + patch_size,
                ]
            )
    return tiles, {
        "original": original_hw,
        "padded": padded_hw,
        "tiles": tile_grid,
    }


def _clear_cuda(device: str) -> None:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.empty_cache()


def _forward_all256(model, image: torch.Tensor):
    """Return ViT-256 CLS and patch tokens using the official HIPT API."""
    patch_batch, patch_rows, patch_columns = model.prepare_img_tensor(image)
    patch_batch = patch_batch.unfold(2, 256, 256).unfold(3, 256, 256)
    patch_batch = rearrange(
        patch_batch,
        "b c p1 p2 h w -> (b p1 p2) c h w",
    )

    cls_batches = []
    subpatch_batches = []
    with torch.no_grad():
        for start in range(0, int(patch_batch.shape[0]), 256):
            minibatch = patch_batch[start : start + 256].to(
                model.device256,
                non_blocking=True,
            )
            tokens = model.model256.get_intermediate_layers(minibatch, n=1)[0]
            cls_batches.append(tokens[:, 0].detach().cpu())
            subpatch_batches.append(tokens[:, 1:].detach().cpu())

    cls_tokens = torch.vstack(cls_batches)
    subpatch_tokens = torch.vstack(subpatch_batches)
    cls_field = (
        cls_tokens.reshape(patch_rows, patch_columns, 384)
        .transpose(0, 1)
        .transpose(0, 2)
        .unsqueeze(0)
    )
    subpatch_field = (
        subpatch_tokens.reshape(
            patch_rows,
            patch_columns,
            16,
            16,
            384,
        )
        .permute(4, 0, 1, 2, 3)
        .unsqueeze(0)
    )
    return cls_field, subpatch_field


def _forward_all4k(model, cls_field: torch.Tensor):
    """Return ViT-4K CLS and spatial tokens using the official HIPT API."""
    _, _, patch_rows, patch_columns = cls_field.shape
    with torch.no_grad():
        tokens = model.model4k.get_intermediate_layers(
            cls_field.to(model.device4k, non_blocking=True),
            n=1,
        )[0]
    cls_token = tokens[:, 0].detach().cpu()
    context_field = (
        tokens[:, 1:]
        .detach()
        .cpu()
        .reshape(1, patch_rows, patch_columns, 192)
        .permute(0, 3, 1, 2)
    )
    return cls_token, context_field


def _extract_single_view(
    image: np.ndarray,
    *,
    hipt_dir: Path,
    model256_path: Path,
    model4k_path: Path,
    device: str,
):
    """Extract dense ViT-4K context and ViT-256 subpatch channels."""
    HIPT_4K, eval_transforms = _load_hipt_classes(hipt_dir)
    started = time()
    tile_size = 4096
    feature_stride = 16
    tiles, shapes = _patchify(image, patch_size=tile_size)
    model = HIPT_4K(
        model256_path=str(model256_path),
        model4k_path=str(model4k_path),
        device256=device,
        device4k=device,
    )
    model.eval()

    subpatch_tiles = []
    patch_tiles = []
    transform = eval_transforms()
    for index, tile in enumerate(tiles):
        if index % 10 == 0:
            print(f"tile {index} / {len(tiles)}", flush=True)
        tensor = transform(tile.astype(np.float32) / 255.0)
        patch_features, subpatch_features = _forward_all256(model, tensor[None])
        patch_tiles.append(
            patch_features.cpu().detach().numpy()[0].transpose(1, 2, 0)
        )
        subpatch_tiles.append(
            subpatch_features.cpu().detach().numpy()[0].transpose(1, 2, 3, 4, 0)
        )
    del tiles
    _clear_cuda(device)

    patch_grid = rearrange(
        patch_tiles,
        "(h1 w1) h2 w2 k -> (h1 h2) (w1 w2) k",
        h1=int(shapes["tiles"][0]),
        w1=int(shapes["tiles"][1]),
    )
    patch_tensor = torch.tensor(patch_grid.transpose(2, 0, 1))
    with torch.no_grad():
        _, context_tokens = _forward_all4k(model, patch_tensor[None])
    context_grid = (
        context_tokens.cpu().detach().numpy()[0].transpose(1, 2, 0)
    )
    del patch_grid, patch_tensor, patch_tiles, model
    _clear_cuda(device)

    target_hw = np.asarray(shapes["original"], dtype=int) // feature_stride
    sub_channels = []
    for channel in range(int(subpatch_tiles[0].shape[-1])):
        field = rearrange(
            np.asarray([tile[..., channel] for tile in subpatch_tiles]),
            "(h1 w1) h2 w2 h3 w3 -> (h1 h2 h3) (w1 w2 w3)",
            h1=int(shapes["tiles"][0]),
            w1=int(shapes["tiles"][1]),
        )
        sub_channels.append(field[: target_hw[0], : target_hw[1]])
    del subpatch_tiles

    context_channels = []
    for channel in range(int(context_grid.shape[-1])):
        field = repeat(
            context_grid[..., channel],
            "h w -> (h h3) (w w3)",
            h3=16,
            w3=16,
        )
        context_channels.append(field[: target_hw[0], : target_hw[1]])
    print(f"single view: {int(time() - started)} sec", flush=True)
    return context_channels, sub_channels


def _extract_shifted_views(
    image: np.ndarray,
    *,
    hipt_dir: Path,
    model256_path: Path,
    model4k_path: Path,
    device: str,
    margin: int = 256,
    stride: int = 64,
):
    factor = 16
    feature_hw = np.asarray(image.shape[:2], dtype=int) // factor
    context_sum = [np.zeros(feature_hw, dtype=np.float32) for _ in range(192)]
    subpatch_sum = [np.zeros(feature_hw, dtype=np.float32) for _ in range(384)]
    offsets = list(range(0, margin, stride))
    views = 0
    for row_start in offsets:
        for column_start in offsets:
            print(
                f"shift {row_start}/{margin}, {column_start}/{margin}",
                flush=True,
            )
            row_stop = -margin + row_start
            column_stop = -margin + column_start
            shifted = image[row_start:row_stop, column_start:column_stop]
            context, subpatch = _extract_single_view(
                shifted,
                hipt_dir=hipt_dir,
                model256_path=model256_path,
                model4k_path=model4k_path,
                device=device,
            )
            feature_row_start = row_start // factor
            feature_column_start = column_start // factor
            feature_row_stop = row_stop // factor
            feature_column_stop = column_stop // factor
            for index, channel in enumerate(context):
                context_sum[index][
                    feature_row_start:feature_row_stop,
                    feature_column_start:feature_column_stop,
                ] += channel
            for index, channel in enumerate(subpatch):
                subpatch_sum[index][
                    feature_row_start:feature_row_stop,
                    feature_column_start:feature_column_stop,
                ] += channel
            views += 1

    border = margin // factor
    for channel in (*context_sum, *subpatch_sum):
        channel /= views
        channel[-border:] = 0.0
        channel[:, -border:] = 0.0
    return context_sum, subpatch_sum


def _smooth_channels(channels: list[np.ndarray], size: int):
    kernel = np.ones((size, size), dtype=np.float32) / float(size**2)
    return [
        cv.filter2D(
            channel,
            ddepth=-1,
            kernel=kernel,
            borderType=cv.BORDER_REFLECT,
        )
        for channel in channels
    ]


def _downsample_rgb(image: np.ndarray):
    return np.stack(
        [
            reduce(
                image[..., channel].astype(np.float16) / 255.0,
                "(h1 h) (w1 w) -> h1 w1",
                "mean",
                h=16,
                w=16,
            ).astype(np.float32)
            for channel in range(3)
        ]
    )


def extract_features(
    image_path: Path,
    output_path: Path,
    *,
    hipt_dir: Path,
    model256_path: Path,
    model4k_path: Path,
    device: str,
    shifted_tiles: bool,
) -> None:
    np.random.seed(0)
    torch.manual_seed(0)
    with Image.open(image_path) as image:
        image_rgb = np.asarray(image.convert("RGB"))
    print(f"Image loaded: {image_path.name}", flush=True)
    if shifted_tiles:
        context, subpatch = _extract_shifted_views(
            image_rgb,
            hipt_dir=hipt_dir,
            model256_path=model256_path,
            model4k_path=model4k_path,
            device=device,
        )
    else:
        context, subpatch = _extract_single_view(
            image_rgb,
            hipt_dir=hipt_dir,
            model256_path=model256_path,
            model4k_path=model4k_path,
            device=device,
        )
    print("Smoothing context channels...", flush=True)
    context = _smooth_channels(context, size=16)
    print("Smoothing subpatch channels...", flush=True)
    subpatch = _smooth_channels(subpatch, size=4)
    embeddings = {
        "cls": context,
        "sub": subpatch,
        "rgb": _downsample_rgb(image_rgb),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(embeddings, handle)
    print(
        "Saved HIPT feature groups: "
        f"cls={len(context)}, sub={len(subpatch)}, rgb={embeddings['rgb'].shape}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hipt-dir", type=Path, required=True)
    parser.add_argument("--model256", type=Path, required=True)
    parser.add_argument("--model4k", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-shift", action="store_true")
    args = parser.parse_args()
    extract_features(
        args.image,
        args.output,
        hipt_dir=args.hipt_dir,
        model256_path=args.model256,
        model4k_path=args.model4k,
        device=args.device,
        shifted_tiles=not args.no_shift,
    )


if __name__ == "__main__":
    main()
