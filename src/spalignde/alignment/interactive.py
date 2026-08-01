"""Notebook controls for reproducible manual cross-sample pre-alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np

from ..io import (
    spatial_coordinates,
    validate_cross_sample_anndata,
    validate_sample_selection,
)
from .cross_sample import (
    ManualPrealignmentConfig,
    PrealignmentResult,
    apply_similarity_transform,
    prealign_cross_sample_manual,
)


def _sample_coordinates(
    adata: ad.AnnData,
    *,
    query_sample: str,
    reference_sample: str,
    sample_key: str,
    cluster_key: str,
    spatial_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    validate_cross_sample_anndata(
        adata,
        sample_key=sample_key,
        cluster_key=cluster_key,
        spatial_key=spatial_key,
    )
    validate_sample_selection(
        adata,
        query_sample=query_sample,
        reference_sample=reference_sample,
        sample_key=sample_key,
    )
    sample = adata.obs[sample_key].astype(str)
    spatial = spatial_coordinates(adata, spatial_key=spatial_key)
    query = spatial[(sample == str(query_sample)).to_numpy()]
    reference = spatial[(sample == str(reference_sample)).to_numpy()]
    return query, reference


def _subsample(
    points: np.ndarray,
    *,
    max_points: int,
    random_state: int,
) -> np.ndarray:
    if max_points < 1:
        raise ValueError("max_points must be at least 1")
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(points), size=max_points, replace=False)
    return points[indices]


def _set_equal_limits(axis: Any, *arrays: np.ndarray) -> None:
    points = np.vstack(arrays)
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    center = 0.5 * (lower + upper)
    half = 0.525 * float(np.max(upper - lower))
    if half <= 0:
        half = 0.5
    axis.set_xlim(center[0] - half, center[0] + half)
    axis.set_ylim(center[1] - half, center[1] + half)


def plot_manual_prealignment_preview(
    adata: ad.AnnData,
    *,
    query_sample: str,
    reference_sample: str,
    config: ManualPrealignmentConfig,
    sample_key: str = "sample_id",
    cluster_key: str = "cluster",
    spatial_key: str = "spatial",
    max_points: int = 50_000,
    random_state: int = 1_000,
    point_size: float = 0.25,
    query_alpha: float = 0.16,
    reference_alpha: float = 0.14,
    figsize: tuple[float, float] = (11, 5.5),
) -> tuple[Any, np.ndarray]:
    """Preview a manual transform without modifying the input AnnData."""
    query, reference = _sample_coordinates(
        adata,
        query_sample=query_sample,
        reference_sample=reference_sample,
        sample_key=sample_key,
        cluster_key=cluster_key,
        spatial_key=spatial_key,
    )
    transformed = apply_similarity_transform(query, config)
    query_plot = _subsample(
        query,
        max_points=max_points,
        random_state=random_state + 1,
    )
    reference_plot = _subsample(
        reference,
        max_points=max_points,
        random_state=random_state + 2,
    )
    transformed_plot = _subsample(
        transformed,
        max_points=max_points,
        random_state=random_state + 3,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        constrained_layout=True,
    )
    panels = (
        (axes[0], query_plot, f"{query_sample} raw", "Before"),
        (
            axes[1],
            transformed_plot,
            f"{query_sample} manual pre-align",
            f"After: scale={config.scale:.4f}, theta={config.theta_deg:.2f}",
        ),
    )
    for axis, query_points, query_label, title in panels:
        axis.scatter(
            reference_plot[:, 0],
            reference_plot[:, 1],
            s=point_size,
            alpha=reference_alpha,
            color="#6F6F6F",
            label=f"{reference_sample} reference",
            rasterized=True,
        )
        axis.scatter(
            query_points[:, 0],
            query_points[:, 1],
            s=point_size,
            alpha=query_alpha,
            color="#2F7F73",
            label=query_label,
            rasterized=True,
        )
        _set_equal_limits(axis, query_points, reference_plot)
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_aspect("equal")
        axis.legend(markerscale=8, loc="lower left", frameon=False)
    return fig, axes


@dataclass
class ManualPrealignmentUI:
    """Interactive controller returned by manual pre-alignment notebooks."""

    adata: ad.AnnData
    query_sample: str
    reference_sample: str
    controls: dict[str, Any]
    widget: Any
    sample_key: str = "sample_id"
    cluster_key: str = "cluster"
    spatial_key: str = "spatial"
    max_points: int = 50_000
    random_state: int = 1_000

    @property
    def selected_config(self) -> ManualPrealignmentConfig:
        """Return the similarity transform currently selected by the sliders."""
        return ManualPrealignmentConfig(
            scale=float(self.controls["scale"].value),
            theta_deg=float(self.controls["theta_deg"].value),
            translation_x=float(self.controls["translation_x"].value),
            translation_y=float(self.controls["translation_y"].value),
        )

    def preview(self, **kwargs: Any) -> tuple[Any, np.ndarray]:
        """Create a static preview using the current slider values."""
        return plot_manual_prealignment_preview(
            self.adata,
            query_sample=self.query_sample,
            reference_sample=self.reference_sample,
            config=self.selected_config,
            sample_key=self.sample_key,
            cluster_key=self.cluster_key,
            spatial_key=self.spatial_key,
            max_points=self.max_points,
            random_state=self.random_state,
            **kwargs,
        )

    def display(self) -> "ManualPrealignmentUI":
        """Display the controls and live preview in a Jupyter notebook."""
        try:
            from IPython.display import display
        except ImportError as exc:  # pragma: no cover - notebook environment
            raise ImportError(
                "Interactive display requires IPython. Install spAlignDE[tutorial]."
            ) from exc
        display(self.widget)
        return self

    def apply(
        self,
        *,
        copy: bool = True,
        verbose: bool = True,
    ) -> PrealignmentResult:
        """Apply the currently selected transform and return package output."""
        return prealign_cross_sample_manual(
            self.adata,
            query_sample=self.query_sample,
            reference_sample=self.reference_sample,
            config=self.selected_config,
            sample_key=self.sample_key,
            cluster_key=self.cluster_key,
            spatial_key=self.spatial_key,
            copy=copy,
            verbose=verbose,
        )


def interactive_manual_prealignment(
    adata: ad.AnnData,
    *,
    query_sample: str,
    reference_sample: str,
    initial_config: ManualPrealignmentConfig | None = None,
    sample_key: str = "sample_id",
    cluster_key: str = "cluster",
    spatial_key: str = "spatial",
    max_points: int = 50_000,
    random_state: int = 1_000,
    scale_range_fraction: float = 0.35,
    rotation_range_deg: float = 35.0,
    translation_radius: float | None = None,
    display_ui: bool = True,
) -> ManualPrealignmentUI:
    """Build a live Jupyter UI for manual query-to-reference pre-alignment.

    The returned controller exposes ``selected_config`` for inspection and
    ``apply()`` for generating the standard :class:`PrealignmentResult`.
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Interactive manual pre-alignment requires ipywidgets and IPython. "
            "Install spAlignDE[tutorial]."
        ) from exc

    initial_config = initial_config or ManualPrealignmentConfig()
    query, reference = _sample_coordinates(
        adata,
        query_sample=query_sample,
        reference_sample=reference_sample,
        sample_key=sample_key,
        cluster_key=cluster_key,
        spatial_key=spatial_key,
    )
    if scale_range_fraction <= 0:
        raise ValueError("scale_range_fraction must be positive")
    if rotation_range_deg <= 0:
        raise ValueError("rotation_range_deg must be positive")
    if translation_radius is None:
        coordinate_span = float(
            np.max(np.ptp(np.vstack([query, reference]), axis=0))
        )
        translation_radius = max(100.0, 0.5 * coordinate_span)
    if translation_radius <= 0:
        raise ValueError("translation_radius must be positive")

    scale_min = max(0.001, initial_config.scale * (1 - scale_range_fraction))
    scale_max = initial_config.scale * (1 + scale_range_fraction)
    translation_step = max(0.01, float(translation_radius) / 100.0)
    controls = {
        "scale": widgets.FloatSlider(
            value=initial_config.scale,
            min=scale_min,
            max=scale_max,
            step=max(0.0001, initial_config.scale / 1_000),
            readout_format=".4f",
            description="scale",
            continuous_update=False,
        ),
        "theta_deg": widgets.FloatSlider(
            value=initial_config.theta_deg,
            min=initial_config.theta_deg - rotation_range_deg,
            max=initial_config.theta_deg + rotation_range_deg,
            step=0.1,
            readout_format=".1f",
            description="theta",
            continuous_update=False,
        ),
        "translation_x": widgets.FloatSlider(
            value=initial_config.translation_x,
            min=initial_config.translation_x - translation_radius,
            max=initial_config.translation_x + translation_radius,
            step=translation_step,
            readout_format=".2f",
            description="tx",
            continuous_update=False,
        ),
        "translation_y": widgets.FloatSlider(
            value=initial_config.translation_y,
            min=initial_config.translation_y - translation_radius,
            max=initial_config.translation_y + translation_radius,
            step=translation_step,
            readout_format=".2f",
            description="ty",
            continuous_update=False,
        ),
    }

    def update_preview(
        scale: float,
        theta_deg: float,
        translation_x: float,
        translation_y: float,
    ) -> None:
        config = ManualPrealignmentConfig(
            scale=scale,
            theta_deg=theta_deg,
            translation_x=translation_x,
            translation_y=translation_y,
        )
        fig, _ = plot_manual_prealignment_preview(
            adata,
            query_sample=query_sample,
            reference_sample=reference_sample,
            config=config,
            sample_key=sample_key,
            cluster_key=cluster_key,
            spatial_key=spatial_key,
            max_points=max_points,
            random_state=random_state,
        )
        display(fig)
        plt.close(fig)

    output = widgets.interactive_output(update_preview, controls)
    widget = widgets.VBox([widgets.VBox(list(controls.values())), output])
    controller = ManualPrealignmentUI(
        adata=adata,
        query_sample=str(query_sample),
        reference_sample=str(reference_sample),
        controls=controls,
        widget=widget,
        sample_key=sample_key,
        cluster_key=cluster_key,
        spatial_key=spatial_key,
        max_points=max_points,
        random_state=random_state,
    )
    if display_ui:
        controller.display()
    return controller


__all__ = [
    "ManualPrealignmentUI",
    "interactive_manual_prealignment",
    "plot_manual_prealignment_preview",
]
