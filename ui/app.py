from io import BytesIO
import base64
from datetime import datetime
import json
import os
from pathlib import Path
import re
from typing import Callable, Optional
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nrrd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, rotate as ndimage_rotate
from scipy.spatial import cKDTree


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
UPLOADED_DATASETS_DIR = Path(
    os.environ.get("SPALIGNDE_UI_UPLOAD_DIR", APP_DIR / "uploaded_datasets")
).expanduser().resolve()
UPLOADED_MANIFEST_PATH = UPLOADED_DATASETS_DIR / "manifest.json"
PLOTLY_MODE_COMPONENT_DIR = Path(__file__).parent / "plotly_mode_component"
plotly_mode_component = components.declare_component(
    "spalign_plotly_mode_experimental",
    path=str(PLOTLY_MODE_COMPONENT_DIR),
)
CLICK_IMAGE_WIDTH = 540
DEFAULT_ATLAS_Z_SLICE = 675
ATLAS_DIM_ALPHA = 0.55
ATLAS_OUTLINE_WIDTH = 2
ATLAS_GLOW_WIDTH = 4
ATLAS_BACKGROUND_STYLE_VERSION = 2
VIEWER_HEIGHT = 540
HISTOLOGY_VIEWER_HEIGHT = VIEWER_HEIGHT
HISTOLOGY_DIM_ALPHA = 0.30
FLEXIBLE_REUSE_MODE = "Flexible reuse mode"
EXCLUSIVE_REUSE_MODE = "Exclusive assignment mode"
CUSTOM_REGION_REUSE_MODES = [FLEXIBLE_REUSE_MODE, EXCLUSIVE_REUSE_MODE]
CUSTOM_REGION_ASSIGNED_ALPHA = 0.42
CLUSTER_CSV_PATHS = [
    APP_DIR / "S2R2.csv",
    DATA_DIR / "S2R2.csv",
]
EXPERIMENTAL_DATASETS = {
    "Allen CCF Atlas": {
        "kind": "atlas",
        "source": "built-in",
        "dataset_id": "allen_ccf_atlas",
        "display_name": "Allen CCF Atlas",
        "path": APP_DIR / "annotation_10.nrrd",
    },
}
CUSTOM_DATASET_TYPES = {
    "Point/cluster CSV": "st",
    "2D label image NPY": "histology",
}
ALLEN_CCF_DIR = Path(
    os.environ.get("SPALIGNDE_ALLEN_CCF_DIR", APP_DIR / "data" / "allen_ccf_2022")
).expanduser()
ANNOTATION_PATHS = [
    Path(os.environ["SPALIGNDE_ALLEN_ANNOTATION"]).expanduser()
    if os.environ.get("SPALIGNDE_ALLEN_ANNOTATION")
    else None,
    ALLEN_CCF_DIR / "annotation_10.nrrd",
    APP_DIR / "annotation_10.nrrd",
]
METADATA_PATHS = [
    Path(os.environ["SPALIGNDE_ALLEN_STRUCTURE_TABLE"]).expanduser()
    if os.environ.get("SPALIGNDE_ALLEN_STRUCTURE_TABLE")
    else None,
    ALLEN_CCF_DIR / "voxel_count_and_differences.csv",
    DATA_DIR / "voxel_count_and_differences.csv",
]
METADATA_COLUMNS = [
    "id",
    "acronym",
    "name",
    "color_hex_triplet",
    "parent_structure_id",
    "structure_id_path",
    "total_voxel_count",
]


def initialize_state() -> None:
    if "pairs" not in st.session_state:
        st.session_state.pairs = []
    if "selected_atlas_regions" not in st.session_state:
        st.session_state.selected_atlas_regions = []
    if "selected_clusters" not in st.session_state:
        st.session_state.selected_clusters = []
    if "next_group_id" not in st.session_state:
        st.session_state.next_group_id = 1
    if "cluster_label" not in st.session_state:
        st.session_state.cluster_label = ""
    if "st_rotation_degrees" not in st.session_state:
        st.session_state.st_rotation_degrees = 0.0
    if "st_x_offset" not in st.session_state:
        st.session_state.st_x_offset = 0.0
    if "st_y_offset" not in st.session_state:
        st.session_state.st_y_offset = 0.0
    if "st_toolbar_mode" not in st.session_state:
        st.session_state.st_toolbar_mode = "pan"
    if "atlas_toolbar_mode" not in st.session_state:
        st.session_state.atlas_toolbar_mode = "pan"
    if "st_selection_revision" not in st.session_state:
        st.session_state.st_selection_revision = 0
    if "atlas_selection_revision" not in st.session_state:
        st.session_state.atlas_selection_revision = 0
    if "atlas_orientation_version" not in st.session_state:
        st.session_state.flip_atlas_vertical = False
        st.session_state.flip_atlas_horizontal = False
        st.session_state.atlas_orientation_version = 2
    if "flip_atlas_vertical" not in st.session_state:
        st.session_state.flip_atlas_vertical = False
    if "flip_atlas_horizontal" not in st.session_state:
        st.session_state.flip_atlas_horizontal = False
    for key in (
        "cluster_x",
        "cluster_y",
        "cluster_id",
        "cluster_point_count",
        "cluster_csv_upload_id",
        "annotation_upload_id",
        "active_z_slice",
        "last_atlas_click",
        "last_cluster_click",
        "processed_cluster_click",
        "processed_atlas_click",
    ):
        if key not in st.session_state:
            st.session_state[key] = None


def point_to_text(point: Optional[dict]) -> str:
    if not point:
        return "No click yet"
    return f"x={point['x']:.2f}, y={point['y']:.2f}"


def stored_cluster_point() -> Optional[dict]:
    if st.session_state.cluster_x is None or st.session_state.cluster_y is None:
        return None
    return {"x": st.session_state.cluster_x, "y": st.session_state.cluster_y}


def upload_id(uploaded_file) -> Optional[tuple]:
    if uploaded_file is None:
        return None
    return (uploaded_file.name, uploaded_file.size)


def reset_cluster_if_upload_changed(current_upload_id: Optional[tuple]) -> None:
    if st.session_state.cluster_csv_upload_id == current_upload_id:
        return
    st.session_state.cluster_csv_upload_id = current_upload_id
    clear_current_cluster_selection()


def find_cluster_csv_path() -> Optional[Path]:
    for path in CLUSTER_CSV_PATHS:
        if path.exists():
            return path
    return None


@st.cache_data(show_spinner="Loading cluster CSV...")
def load_cluster_csv_from_path(path_text: str, modified_time: float) -> pd.DataFrame:
    del modified_time
    return pd.read_csv(path_text)


@st.cache_data(show_spinner="Loading uploaded cluster CSV...")
def load_cluster_csv_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(file_bytes))


def load_cluster_csv(uploaded_file) -> tuple[Optional[pd.DataFrame], str, Optional[str]]:
    if uploaded_file is not None:
        uploaded_file.seek(0)
        cluster_df = load_cluster_csv_from_bytes(uploaded_file.read())
        source = uploaded_file.name
    else:
        csv_path = find_cluster_csv_path()
        if csv_path is None:
            checked_paths = ", ".join(str(path) for path in CLUSTER_CSV_PATHS)
            return None, "No cluster CSV loaded", f"Upload S2R2.csv or place it at one of: {checked_paths}."
        cluster_df = load_cluster_csv_from_path(str(csv_path), csv_path.stat().st_mtime)
        source = str(csv_path)

    required_columns = ["x", "y", "cluster"]
    missing_columns = [column for column in required_columns if column not in cluster_df.columns]
    if missing_columns:
        return None, source, f"Cluster CSV is missing required columns: {', '.join(missing_columns)}."

    cluster_df = cluster_df[required_columns].dropna().copy()
    cluster_df["cluster"] = cluster_df["cluster"].astype(int)
    return cluster_df, source, None


def cluster_bounds(cluster_df: pd.DataFrame) -> tuple[float, float, float, float]:
    xmin = float(cluster_df["x"].min())
    xmax = float(cluster_df["x"].max())
    ymin = float(cluster_df["y"].min())
    ymax = float(cluster_df["y"].max())
    x_margin = max((xmax - xmin) * 0.05, 1.0)
    y_margin = max((ymax - ymin) * 0.05, 1.0)
    return xmin - x_margin, xmax + x_margin, ymin - y_margin, ymax + y_margin


def normalize_rotation_angle(angle: float) -> float:
    return float(angle % 360.0)


def reset_rotation() -> None:
    st.session_state.st_rotation_degrees = 0.0


def transform_st_coordinates(
    cluster_df: pd.DataFrame,
    rotation_degrees: float,
    x_offset: float,
    y_offset: float,
) -> pd.DataFrame:
    transformed = cluster_df.copy()
    center_x = float(transformed["x"].mean())
    center_y = float(transformed["y"].mean())

    theta = np.deg2rad(rotation_degrees)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    x_centered = transformed["x"] - center_x
    y_centered = transformed["y"] - center_y

    transformed["x_final"] = (
        x_centered * cos_theta
        - y_centered * sin_theta
        + center_x
        + x_offset
    )
    transformed["y_final"] = (
        x_centered * sin_theta
        + y_centered * cos_theta
        + center_y
        + y_offset
    )

    render_df = transformed[["x_final", "y_final", "cluster"]].rename(
        columns={"x_final": "x", "y_final": "y"}
    )
    return render_df


POINT_SIZE = 2.0
POINT_ALPHA = 0.98
ST_DIM_ALPHA = 0.55
SELECTED_POINT_SIZE_MULTIPLIER = 4.0
SELECTED_GLOW_SIZE_MULTIPLIER = 8.0
FIG_DPI = 360

CLUSTER_COLORS = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
    (255, 187, 120),
    (152, 223, 138),
    (255, 152, 150),
    (197, 176, 213),
    (196, 156, 148),
    (247, 182, 210),
    (199, 199, 199),
    (219, 219, 141),
    (158, 218, 229),
    (174, 199, 232),
]


def cluster_color(cluster_id: int) -> tuple[int, int, int]:
    if cluster_id == 0:
        return (0, 0, 0)
    return CLUSTER_COLORS[int(cluster_id) % len(CLUSTER_COLORS)]


def zoomed_bounds(
    bounds: tuple[float, float, float, float],
    zoom: float,
) -> tuple[float, float, float, float]:
    xmin, xmax, ymin, ymax = bounds
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    x_half = (xmax - xmin) / (2 * zoom)
    y_half = (ymax - ymin) / (2 * zoom)
    return (
        x_center - x_half,
        x_center + x_half,
        y_center - y_half,
        y_center + y_half,
    )


def scatter_bounds(
    cluster_df: pd.DataFrame,
    zoom: float,
) -> tuple[float, float, float, float]:
    xmin = float(cluster_df["x"].min())
    xmax = float(cluster_df["x"].max())
    ymin = float(cluster_df["y"].min())
    ymax = float(cluster_df["y"].max())
    x_span = xmax - xmin
    y_span = ymax - ymin
    span = max(x_span, y_span, 1.0)
    padding = span * 0.015
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    half_span = (span / 2) + padding
    full_bounds = (
        x_center - half_span,
        x_center + half_span,
        y_center - half_span,
        y_center + half_span,
    )
    return zoomed_bounds(full_bounds, zoom)


def scatter_cluster_image(
    cluster_df: pd.DataFrame,
    selected_clusters: set[int],
    zoom: float,
) -> tuple[Image.Image, tuple[float, float, float, float], dict[int, int]]:
    bounds = scatter_bounds(cluster_df, zoom)
    xmin, xmax, ymin, ymax = bounds
    point_counts = cluster_df.groupby("cluster").size().astype(int).to_dict()

    fig, ax = plt.subplots(figsize=(6, 6), dpi=FIG_DPI)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    selected_mask = cluster_df["cluster"].isin(selected_clusters)
    nonselected = cluster_df[~selected_mask]
    unselected_alpha = ST_DIM_ALPHA if selected_clusters else POINT_ALPHA

    for cluster_id, group in nonselected.groupby("cluster", sort=True):
        ax.scatter(
            group["x"],
            group["y"],
            s=POINT_SIZE,
            c=[np.array(cluster_color(int(cluster_id))) / 255],
            marker=".",
            linewidths=0,
            alpha=unselected_alpha,
        )

    if selected_clusters and selected_mask.any():
        selected_points = cluster_df[selected_mask]
        ax.scatter(
            selected_points["x"],
            selected_points["y"],
            s=POINT_SIZE * SELECTED_GLOW_SIZE_MULTIPLIER,
            c="white",
            marker=".",
            linewidths=0,
            alpha=0.85,
        )
        ax.scatter(
            selected_points["x"],
            selected_points["y"],
            s=POINT_SIZE * (SELECTED_GLOW_SIZE_MULTIPLIER * 0.65),
            c="black",
            marker=".",
            linewidths=0,
            alpha=0.75,
        )
        for cluster_id, group in selected_points.groupby("cluster", sort=True):
            ax.scatter(
                group["x"],
                group["y"],
                s=POINT_SIZE * SELECTED_POINT_SIZE_MULTIPLIER,
                c=[np.array(cluster_color(int(cluster_id))) / 255],
                marker=".",
                linewidths=0,
                alpha=1.0,
            )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.set_position([0, 0, 1, 1])
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    image_buffer = BytesIO()
    fig.savefig(image_buffer, format="png", dpi=FIG_DPI, facecolor="white", pad_inches=0)
    plt.close(fig)
    image_buffer.seek(0)
    return Image.open(image_buffer).convert("RGB"), bounds, point_counts


class PlotlyEventFigure:
    def __init__(self, figure: go.Figure):
        self.figure = figure

    def to_json(self) -> str:
        figure_json = json.loads(self.figure.to_json())
        figure_json["config"] = {
            "displayModeBar": True,
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": True,
            "modeBarButtonsToRemove": ["autoScale2d", "lasso2d", "select2d"],
        }
        return json.dumps(figure_json)


def plotly_mode_events(
    figure: go.Figure,
    key: str,
    current_mode: str,
    height: int = VIEWER_HEIGHT,
) -> dict:
    result = plotly_mode_component(
        plot_obj=PlotlyEventFigure(figure).to_json(),
        override_height=height,
        override_width="100%",
        current_mode=current_mode,
        key=key,
        default={"type": "mode", "mode": "pan"},
    )
    if isinstance(result, str):
        return json.loads(result)
    return result


def viewer_axis(range_values: list[float]) -> dict:
    return {
        "range": range_values,
        "showgrid": False,
        "showline": False,
        "showticklabels": False,
        "ticks": "",
        "zeroline": False,
        "fixedrange": False,
        "constrain": "domain",
    }


def cluster_plotly_figure(
    cluster_df: pd.DataFrame,
    selected_clusters: set[int],
    bounds: tuple[float, float, float, float],
    assigned_clusters: Optional[set[int]] = None,
    assigned_cluster_colors: Optional[dict[int, tuple[int, int, int]]] = None,
    selected_cluster_colors: Optional[dict[int, tuple[int, int, int]]] = None,
    color_function: Callable[[int], tuple[int, int, int]] = cluster_color,
    point_size: float = POINT_SIZE,
    include_picker_trace: bool = True,
    uirevision: str = "st-cluster-view",
    point_hoverinfo: str = "skip",
    viewer_height: int = VIEWER_HEIGHT,
    dim_alpha: float = ST_DIM_ALPHA,
    selected_point_multiplier: float = SELECTED_POINT_SIZE_MULTIPLIER,
    selected_glow_multiplier: float = SELECTED_GLOW_SIZE_MULTIPLIER,
    selected_inner_glow_ratio: float = 0.65,
    background_color: str = "white",
) -> go.Figure:
    xmin, xmax, ymin, ymax = bounds
    figure = go.Figure()
    assigned_clusters = assigned_clusters or set()
    assigned_cluster_colors = assigned_cluster_colors or {}
    selected_cluster_colors = selected_cluster_colors or {}
    selected_mask = cluster_df["cluster"].isin(selected_clusters)
    unselected_alpha = dim_alpha if selected_clusters else POINT_ALPHA

    for cluster_id, group in cluster_df[~selected_mask].groupby("cluster", sort=True):
        red, green, blue = color_function(int(cluster_id))
        figure.add_trace(
            go.Scattergl(
                x=group["x"],
                y=group["y"],
                mode="markers",
                marker={
                    "size": point_size,
                    "color": f"rgba({red},{green},{blue},{unselected_alpha})",
                },
                customdata=np.full(len(group), int(cluster_id)),
                hoverinfo=point_hoverinfo,
                showlegend=False,
            )
        )

    assigned_ids = set(assigned_cluster_colors) or assigned_clusters
    assigned_mask = cluster_df["cluster"].isin(assigned_ids) & ~selected_mask
    if assigned_mask.any():
        assigned_points = cluster_df[assigned_mask]
        figure.add_trace(
            go.Scattergl(
                x=assigned_points["x"],
                y=assigned_points["y"],
                mode="markers",
                marker={
                    "size": point_size * 2.4,
                    "color": "rgba(255,255,255,0.55)",
                    "symbol": "circle-open",
                },
                customdata=assigned_points["cluster"].astype(int),
                hoverinfo=point_hoverinfo,
                showlegend=False,
            )
        )
        for cluster_id, group in assigned_points.groupby("cluster", sort=True):
            red, green, blue = assigned_cluster_colors.get(
                int(cluster_id),
                (120, 120, 120),
            )
            figure.add_trace(
                go.Scattergl(
                    x=group["x"],
                    y=group["y"],
                    mode="markers",
                    marker={
                        "size": point_size * 2.2,
                        "color": (
                            f"rgba({red},{green},{blue},"
                            f"{CUSTOM_REGION_ASSIGNED_ALPHA})"
                        ),
                    },
                    customdata=np.full(len(group), int(cluster_id)),
                    hoverinfo=point_hoverinfo,
                    showlegend=False,
                )
            )

    if selected_clusters and selected_mask.any():
        selected_points = cluster_df[selected_mask]
        figure.add_trace(
            go.Scattergl(
                x=selected_points["x"],
                y=selected_points["y"],
                mode="markers",
                marker={
                    "size": point_size * selected_glow_multiplier,
                    "color": "rgba(255,255,255,0.85)",
                },
                customdata=selected_points["cluster"].astype(int),
                hoverinfo=point_hoverinfo,
                showlegend=False,
            )
        )
        figure.add_trace(
            go.Scattergl(
                x=selected_points["x"],
                y=selected_points["y"],
                mode="markers",
                marker={
                    "size": point_size
                    * selected_glow_multiplier
                    * selected_inner_glow_ratio,
                    "color": "rgba(0,0,0,0.75)",
                },
                customdata=selected_points["cluster"].astype(int),
                hoverinfo=point_hoverinfo,
                showlegend=False,
            )
        )
        for cluster_id, group in selected_points.groupby("cluster", sort=True):
            red, green, blue = selected_cluster_colors.get(
                int(cluster_id),
                color_function(int(cluster_id)),
            )
            figure.add_trace(
                go.Scattergl(
                    x=group["x"],
                    y=group["y"],
                    mode="markers",
                    marker={
                        "size": point_size * selected_point_multiplier,
                        "color": f"rgb({red},{green},{blue})",
                    },
                    customdata=np.full(len(group), int(cluster_id)),
                    hoverinfo=point_hoverinfo,
                    showlegend=False,
                )
            )

    if include_picker_trace:
        figure.add_trace(
            go.Scattergl(
                x=cluster_df["x"],
                y=cluster_df["y"],
                mode="markers",
                marker={
                    "size": max(point_size * 4, 6),
                    "color": "rgba(0,0,0,0.01)",
                },
                customdata=cluster_df["cluster"].astype(int),
                hoverinfo="none",
                showlegend=False,
            )
        )

    figure.update_layout(
        height=viewer_height,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor=background_color,
        plot_bgcolor=background_color,
        dragmode="pan",
        clickmode="event",
        hovermode="closest",
        uirevision=uirevision,
        xaxis=viewer_axis([xmin, xmax]),
        yaxis={
            **viewer_axis([ymax, ymin]),
            "scaleanchor": "x",
            "scaleratio": 1,
        },
    )
    return figure


def atlas_plotly_figure(
    image: Image.Image,
    slice_labels: np.ndarray,
    crop_bounds: tuple[int, int, int, int],
    uirevision: str = "atlas-view",
) -> go.Figure:
    crop_y0, crop_y1, crop_x0, crop_x1 = crop_bounds
    image_buffer = BytesIO()
    image.save(image_buffer, format="PNG", optimize=True)
    image_source = (
        "data:image/png;base64,"
        + base64.b64encode(image_buffer.getvalue()).decode("ascii")
    )

    click_height = min(slice_labels.shape[0], 180)
    click_width = min(slice_labels.shape[1], 256)
    click_x = np.linspace(crop_x0, crop_x1 - 1, click_width)
    click_y = np.linspace(crop_y0, crop_y1 - 1, click_height)

    figure = go.Figure(
        go.Image(
            source=image_source,
            x0=crop_x0,
            y0=crop_y0,
            dx=(crop_x1 - crop_x0) / max(image.width, 1),
            dy=(crop_y1 - crop_y0) / max(image.height, 1),
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Heatmap(
            z=np.zeros((click_height, click_width), dtype=np.uint8),
            x=click_x,
            y=click_y,
            colorscale=[
                [0, "rgba(0,0,0,0)"],
                [1, "rgba(0,0,0,0)"],
            ],
            showscale=False,
            hoverinfo="none",
            opacity=0.01,
        )
    )
    figure.update_layout(
        height=VIEWER_HEIGHT,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="white",
        plot_bgcolor="white",
        dragmode="pan",
        clickmode="event",
        hovermode="closest",
        uirevision=uirevision,
        xaxis=viewer_axis([crop_x0, crop_x1]),
        yaxis={
            **viewer_axis([crop_y1, crop_y0]),
            "scaleanchor": "x",
            "scaleratio": 1,
        },
    )
    return figure


def first_plot_click(event_payload: dict) -> Optional[dict]:
    if (
        not event_payload
        or event_payload.get("type") != "click"
        or event_payload.get("mode") != "select"
    ):
        return None
    points = event_payload.get("points", [])
    if not points:
        return None
    event = points[0]
    if "x" not in event or "y" not in event:
        return None
    return event


def clicked_trace_customdata(figure: go.Figure, event: dict) -> Optional[int]:
    curve_number = int(event.get("curveNumber", -1))
    point_number = int(event.get("pointNumber", -1))
    if curve_number < 0 or curve_number >= len(figure.data):
        return None
    customdata = figure.data[curve_number].customdata
    if customdata is None or point_number < 0 or point_number >= len(customdata):
        return None
    value = customdata[point_number]
    if isinstance(value, (list, tuple, np.ndarray)):
        value = value[0]
    return int(value)


def display_to_data_coordinate(
    click: dict,
    display_image: Image.Image,
    bounds: tuple[float, float, float, float],
) -> tuple[float, float]:
    xmin, xmax, ymin, ymax = bounds
    x_fraction = click["x"] / max(display_image.width - 1, 1)
    y_fraction = click["y"] / max(display_image.height - 1, 1)
    data_x = xmin + x_fraction * (xmax - xmin)
    data_y = ymin + y_fraction * (ymax - ymin)
    return float(data_x), float(data_y)


def nearest_cluster_at_point(cluster_df: pd.DataFrame, x: float, y: float) -> int:
    coords = cluster_df[["x", "y"]].to_numpy()
    if coords.size == 0:
        return 0
    tree = cKDTree(coords)
    _distance, index = tree.query([[x, y]], k=1)
    return int(cluster_df.iloc[int(index[0])]["cluster"])


def selected_cluster_ids() -> set[int]:
    return {int(cluster["cluster_id"]) for cluster in st.session_state.selected_clusters}


def selected_clusters_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        st.session_state.selected_clusters,
        columns=["cluster_id", "cluster_point_count", "cluster_x", "cluster_y"],
    )


def toggle_cluster_selection(cluster_id: int, x: float, y: float, point_counts: dict[int, int]) -> None:
    st.session_state.last_cluster_click = {"x": x, "y": y, "cluster_id": cluster_id}

    if cluster_id in selected_cluster_ids():
        st.session_state.selected_clusters = [
            cluster
            for cluster in st.session_state.selected_clusters
            if int(cluster["cluster_id"]) != int(cluster_id)
        ]
        if not st.session_state.selected_clusters:
            st.session_state.cluster_x = None
            st.session_state.cluster_y = None
            st.session_state.cluster_id = None
            st.session_state.cluster_point_count = None
            st.session_state.cluster_label = ""
        return

    selected_cluster = {
        "cluster_id": int(cluster_id),
        "cluster_point_count": int(point_counts.get(cluster_id, 0)),
        "cluster_x": float(x),
        "cluster_y": float(y),
    }
    st.session_state.selected_clusters.append(selected_cluster)
    st.session_state.cluster_id = int(cluster_id)
    st.session_state.cluster_label = str(cluster_id)
    st.session_state.cluster_point_count = selected_cluster["cluster_point_count"]
    st.session_state.cluster_x = x
    st.session_state.cluster_y = y


def reset_atlas_if_annotation_changed(current_upload_id: Optional[tuple]) -> None:
    if st.session_state.annotation_upload_id == current_upload_id:
        return
    st.session_state.annotation_upload_id = current_upload_id
    clear_selected_atlas_points()
    st.session_state.active_z_slice = None
    st.session_state.processed_atlas_click = None


def reset_atlas_orientation() -> None:
    st.session_state.flip_atlas_vertical = False
    st.session_state.flip_atlas_horizontal = False
    clear_selected_atlas_points()


def image_coordinates_key(prefix: str, current_upload_id: tuple) -> str:
    filename, size = current_upload_id
    return f"{prefix}_image_coordinates_{filename}_{size}"


def resized_for_click(image: Image.Image, width: int = CLICK_IMAGE_WIDTH) -> Image.Image:
    if image.width == width:
        return image
    height = max(1, round(image.height * (width / image.width)))
    return image.resize((width, height), Image.Resampling.NEAREST)


def display_to_array_index(
    display_value: int,
    display_size: int,
    array_size: int,
) -> int:
    if display_size <= 1:
        return 0
    scaled = round(display_value * (array_size - 1) / (display_size - 1))
    return int(np.clip(scaled, 0, array_size - 1))


def centered_crop_bounds(
    height: int,
    width: int,
    zoom: float,
) -> tuple[int, int, int, int]:
    crop_height = max(1, round(height / zoom))
    crop_width = max(1, round(width / zoom))
    y_start = max(0, (height - crop_height) // 2)
    x_start = max(0, (width - crop_width) // 2)
    return y_start, y_start + crop_height, x_start, x_start + crop_width


def load_uploaded_image(uploaded_file) -> Optional[Image.Image]:
    if uploaded_file is None:
        return None
    uploaded_file.seek(0)
    return Image.open(uploaded_file).convert("RGB")


@st.cache_resource(show_spinner="Loading annotation NRRD...")
def load_annotation_from_path(path_text: str, modified_time: float) -> tuple[np.ndarray, dict]:
    del modified_time
    annotation, header = nrrd.read(path_text)
    return annotation, dict(header)


@st.cache_resource(show_spinner="Loading uploaded annotation NRRD...")
def load_annotation_from_bytes(file_bytes: bytes) -> tuple[np.ndarray, dict]:
    annotation, header = nrrd.read(BytesIO(file_bytes))
    return annotation, dict(header)


def load_annotation(uploaded_file) -> tuple[Optional[np.ndarray], Optional[dict], str]:
    if uploaded_file is not None:
        uploaded_file.seek(0)
        annotation, header = load_annotation_from_bytes(uploaded_file.read())
        return annotation, header, uploaded_file.name

    annotation_path = next(
        (path for path in ANNOTATION_PATHS if path is not None and path.is_file()),
        None,
    )
    if annotation_path is not None:
        annotation, header = load_annotation_from_path(
            str(annotation_path),
            annotation_path.stat().st_mtime,
        )
        return annotation, header, str(annotation_path)

    return None, None, "No annotation loaded"


def atlas_voxel_spacing(header: Optional[dict]) -> tuple[float, float]:
    dx = dy = 10.0
    if header is None:
        return dx, dy

    space_directions = header.get("space directions", None)
    if space_directions is not None:
        dy = float(np.linalg.norm(space_directions[1]))
        dx = float(np.linalg.norm(space_directions[2]))
    return dx, dy


def atlas_slice_info(
    annotation: np.ndarray,
    header: Optional[dict],
    z_slice: int,
    flip_vertical: bool,
    flip_horizontal: bool,
) -> dict:
    sl = annotation[z_slice].astype(int)
    if flip_vertical:
        sl = np.flipud(sl)
    if flip_horizontal:
        sl = np.fliplr(sl)
    dx, dy = atlas_voxel_spacing(header)
    height, width = sl.shape
    x_axis = np.linspace(0, (width - 1) * dx, width)
    y_axis = np.linspace(0, (height - 1) * dy, height)
    return {
        "sl": sl,
        "xJ": x_axis,
        "yJ": y_axis,
        "dx": dx,
        "dy": dy,
        "H": height,
        "W": width,
        "z": z_slice,
    }


def find_metadata_path() -> Optional[Path]:
    for path in METADATA_PATHS:
        if path is not None and path.is_file():
            return path
    return None


def normalize_hex(value) -> str:
    text = str(value).strip().lstrip("#")
    if len(text) == 6:
        return text.upper()
    return "808080"


def hex_to_rgb(value) -> tuple[int, int, int]:
    text = normalize_hex(value)
    return tuple(int(text[index : index + 2], 16) for index in (0, 2, 4))


def rgb_to_hex(color: tuple[int, int, int]) -> str:
    return "".join(f"{int(channel):02X}" for channel in color)


def rgb_distance(color_a: tuple[int, int, int], color_b: tuple[int, int, int]) -> float:
    return float(np.linalg.norm(np.array(color_a) - np.array(color_b)))


def distinct_label_color(label: int) -> tuple[int, int, int]:
    hue = ((label * 0.618033988749895) % 1.0)
    saturation = 0.72 + 0.2 * ((label % 5) / 4)
    value = 0.82 + 0.14 * ((label % 7) / 6)
    sector = int(hue * 6)
    fraction = hue * 6 - sector
    p = value * (1 - saturation)
    q = value * (1 - fraction * saturation)
    t = value * (1 - (1 - fraction) * saturation)
    sector = sector % 6
    if sector == 0:
        rgb = (value, t, p)
    elif sector == 1:
        rgb = (q, value, p)
    elif sector == 2:
        rgb = (p, value, t)
    elif sector == 3:
        rgb = (p, q, value)
    elif sector == 4:
        rgb = (t, p, value)
    else:
        rgb = (value, p, q)
    return tuple(int(channel * 255) for channel in rgb)


def fallback_color(label: int) -> tuple[int, int, int]:
    if label == 0:
        return (0, 0, 0)
    rng = np.random.default_rng(label)
    return tuple(int(channel) for channel in rng.integers(70, 230, size=3))


@st.cache_data(show_spinner="Loading atlas metadata...")
def load_region_metadata() -> tuple[pd.DataFrame, Optional[Path], Optional[str]]:
    metadata_path = find_metadata_path()
    if metadata_path is None:
        return (
            pd.DataFrame(columns=METADATA_COLUMNS),
            None,
            "Could not find voxel_count_and_differences.csv. Set "
            "SPALIGNDE_ALLEN_CCF_DIR or SPALIGNDE_ALLEN_STRUCTURE_TABLE.",
        )

    metadata = pd.read_csv(metadata_path)
    missing_columns = [column for column in METADATA_COLUMNS if column not in metadata.columns]
    if missing_columns:
        return (
            pd.DataFrame(columns=METADATA_COLUMNS),
            metadata_path,
            f"Metadata CSV is missing required columns: {', '.join(missing_columns)}",
        )

    metadata = metadata[METADATA_COLUMNS].copy()
    metadata["id"] = metadata["id"].astype(int)
    metadata["acronym"] = metadata["acronym"].astype(str)
    metadata["color_hex_triplet"] = metadata["color_hex_triplet"].apply(normalize_hex)
    metadata = metadata.sort_values("acronym")
    return metadata, metadata_path, None


def metadata_by_id(metadata: pd.DataFrame) -> dict[int, dict]:
    if metadata.empty:
        return {}
    return metadata.set_index("id").to_dict("index")


def label_details(label: int, metadata_lookup: dict[int, dict]) -> dict:
    details = metadata_lookup.get(label, {})
    return {
        "label": int(label),
        "atlas_region": details.get("acronym", f"label_{label}"),
        "name": details.get("name", "Unknown region"),
        "color_hex_triplet": details.get("color_hex_triplet", "808080"),
        "parent_structure_id": details.get("parent_structure_id", ""),
        "structure_id_path": details.get("structure_id_path", ""),
        "total_voxel_count": details.get("total_voxel_count", ""),
    }


def selected_label_set() -> set[int]:
    return {int(region["label"]) for region in st.session_state.selected_atlas_regions}


def toggle_atlas_label(label: int, x: int, y: int, metadata_lookup: dict[int, dict]) -> None:
    if label == 0:
        st.session_state.last_atlas_click = {
            "x": x,
            "y": y,
            "label": label,
            "atlas_region": "background",
            "name": "Background",
        }
        return

    current_labels = selected_label_set()
    if label in current_labels:
        st.session_state.selected_atlas_regions = [
            region
            for region in st.session_state.selected_atlas_regions
            if int(region["label"]) != label
        ]
    else:
        region = label_details(label, metadata_lookup)
        region["atlas_x"] = x
        region["atlas_y"] = y
        st.session_state.selected_atlas_regions.append(region)

    clicked = label_details(label, metadata_lookup)
    clicked["x"] = x
    clicked["y"] = y
    st.session_state.last_atlas_click = clicked


def boundary_mask(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    neighbors_same = (
        padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return center & ~neighbors_same


def atlas_slice_image(
    slice_labels: np.ndarray,
    metadata_lookup: dict[int, dict],
    selected_labels: set[int],
    color_mode: str,
    assigned_labels: Optional[set[int]] = None,
    assigned_label_colors: Optional[dict[int, tuple[int, int, int]]] = None,
    selected_label_colors: Optional[dict[int, tuple[int, int, int]]] = None,
) -> Image.Image:
    labels = np.unique(slice_labels)
    image_array = np.zeros((*slice_labels.shape, 3), dtype=np.uint8)
    used_colors: list[tuple[int, int, int]] = []
    assigned_labels = assigned_labels or set()
    assigned_label_colors = assigned_label_colors or {
        int(label): (120, 120, 120)
        for label in assigned_labels
    }
    selected_label_colors = selected_label_colors or {}

    for label in labels:
        label_int = int(label)
        if label_int == 0:
            color = (255, 255, 255)
            image_array[slice_labels == label_int] = color
            continue

        details = metadata_lookup.get(label_int)
        metadata_color = hex_to_rgb(details["color_hex_triplet"]) if details else None

        if color_mode == "Subsection colors":
            if metadata_color is None or any(rgb_distance(metadata_color, prior) < 45 for prior in used_colors):
                color = distinct_label_color(label_int)
            else:
                color = metadata_color
            used_colors.append(color)
        else:
            color = metadata_color if metadata_color else fallback_color(label_int)

        image_array[slice_labels == label_int] = color

    if selected_labels:
        selected_mask = np.isin(slice_labels, list(selected_labels))
        background_mask = slice_labels == 0
        unselected_mask = ~selected_mask & ~background_mask

        image_array[unselected_mask] = (
            image_array[unselected_mask].astype(np.float32)
            * ATLAS_DIM_ALPHA
        ).astype(np.uint8)

        glow = binary_dilation(selected_mask, iterations=ATLAS_GLOW_WIDTH) & ~selected_mask
        image_array[glow] = (0, 0, 0)

        outer_edge = binary_dilation(selected_mask, iterations=ATLAS_OUTLINE_WIDTH)
        inner_edge = binary_erosion(selected_mask, iterations=ATLAS_OUTLINE_WIDTH)
        outline = outer_edge & ~inner_edge
        image_array[outline] = (255, 255, 255)

    for label, color in assigned_label_colors.items():
        label_mask = slice_labels == int(label)
        if not label_mask.any():
            continue
        color_array = np.array(color, dtype=np.float32)
        image_array[label_mask] = (
            image_array[label_mask].astype(np.float32)
            * (1.0 - CUSTOM_REGION_ASSIGNED_ALPHA)
            + color_array * CUSTOM_REGION_ASSIGNED_ALPHA
        ).astype(np.uint8)
        image_array[boundary_mask(label_mask)] = color

    for label, color in selected_label_colors.items():
        label_mask = slice_labels == int(label)
        if not label_mask.any():
            continue
        image_array[label_mask] = color
        image_array[boundary_mask(label_mask)] = (255, 255, 255)

    return Image.fromarray(image_array)


@st.cache_data(show_spinner="Loading histology labels...")
def load_histology_labels(path_text: str, modified_time: float) -> np.ndarray:
    del modified_time
    labels = np.load(path_text)
    if labels.ndim != 2:
        raise ValueError("Histology labels must be a 2D label image.")
    return labels.astype(float)


@st.cache_data(show_spinner="Preparing histology image...")
def rotated_histology_labels(
    path_text: str,
    modified_time: float,
    rotation_degrees: float,
) -> tuple[np.ndarray, tuple[int, int, int, int], dict[int, int]]:
    labels = load_histology_labels(path_text, modified_time)
    rotated = ndimage_rotate(
        labels,
        angle=float(rotation_degrees),
        reshape=True,
        order=0,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )
    valid = ~np.isnan(rotated)
    if not valid.any():
        raise ValueError("The rotated histology image contains no visible labels.")

    rotated_h, rotated_w = rotated.shape
    image_bounds = (0, rotated_h, 0, rotated_w)

    source_valid = labels[~np.isnan(labels)].astype(int)
    unique_labels, counts = np.unique(source_valid, return_counts=True)
    pixel_counts = {
        int(label): int(count)
        for label, count in zip(unique_labels, counts)
    }
    return rotated, image_bounds, pixel_counts


def histology_label_color(label: int) -> tuple[int, int, int]:
    tab20 = plt.get_cmap("tab20")
    if 0 <= label < 20:
        color = tab20(label)
        return tuple(int(channel * 255) for channel in color[:3])
    return distinct_label_color(label)


def histology_label_image(
    labels: np.ndarray,
    selected_labels: set[int],
    assigned_labels: Optional[set[int]] = None,
    assigned_label_colors: Optional[dict[int, tuple[int, int, int]]] = None,
    selected_label_colors: Optional[dict[int, tuple[int, int, int]]] = None,
) -> Image.Image:
    image_array = np.full((*labels.shape, 3), 255, dtype=np.uint8)
    valid = ~np.isnan(labels)
    assigned_labels = assigned_labels or set()
    assigned_label_colors = assigned_label_colors or {
        int(label): (120, 120, 120)
        for label in assigned_labels
    }
    selected_label_colors = selected_label_colors or {}

    for label in np.unique(labels[valid]).astype(int):
        image_array[valid & (labels == label)] = histology_label_color(int(label))

    if selected_labels:
        selected_mask = valid & np.isin(labels, list(selected_labels))
        unselected_mask = valid & ~selected_mask
        faded = image_array[unselected_mask].astype(np.float32)
        image_array[unselected_mask] = (
            faded * HISTOLOGY_DIM_ALPHA
            + 255.0 * (1.0 - HISTOLOGY_DIM_ALPHA)
        ).astype(np.uint8)

        glow = binary_dilation(selected_mask, iterations=2) & ~selected_mask
        image_array[glow] = (0, 0, 0)
        outline = boundary_mask(selected_mask)
        image_array[outline] = (255, 255, 255)

    for label, color in assigned_label_colors.items():
        label_mask = valid & (labels == int(label))
        if not label_mask.any():
            continue
        color_array = np.array(color, dtype=np.float32)
        image_array[label_mask] = (
            np.full((int(label_mask.sum()), 3), 255.0, dtype=np.float32)
            * (1.0 - CUSTOM_REGION_ASSIGNED_ALPHA)
            + color_array * CUSTOM_REGION_ASSIGNED_ALPHA
        ).astype(np.uint8)
        image_array[boundary_mask(label_mask)] = color

    for label, color in selected_label_colors.items():
        label_mask = valid & (labels == int(label))
        if not label_mask.any():
            continue
        image_array[label_mask] = color
        image_array[boundary_mask(label_mask)] = (255, 255, 255)

    return Image.fromarray(image_array)


def selected_regions_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        st.session_state.selected_atlas_regions,
        columns=[
            "label",
            "atlas_region",
            "name",
            "color_hex_triplet",
            "parent_structure_id",
            "total_voxel_count",
            "atlas_x",
            "atlas_y",
        ],
    )


def clear_selected_atlas_points() -> None:
    st.session_state.selected_atlas_regions = []
    st.session_state.last_atlas_click = None
    st.session_state.atlas_selection_revision += 1


def clear_current_cluster_selection() -> None:
    st.session_state.cluster_x = None
    st.session_state.cluster_y = None
    st.session_state.cluster_id = None
    st.session_state.cluster_point_count = None
    st.session_state.cluster_label = ""
    st.session_state.last_cluster_click = None
    st.session_state.selected_clusters = []
    st.session_state.st_selection_revision += 1


def clear_all_unsaved_selections() -> None:
    clear_current_cluster_selection()
    clear_selected_atlas_points()


def clear_saved_pairs() -> None:
    st.session_state.pairs = []
    st.session_state.next_group_id = 1


def experimental_key(side: str, dataset_name: str, field: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", dataset_name.lower()).strip("_")
    return f"experimental_{side}_{slug}_{field}"


def safe_filename(filename: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name).strip("._")
    return clean or "uploaded_dataset"


def dataset_display_name(dataset_name: str, definition: Optional[dict] = None) -> str:
    if definition and definition.get("display_name"):
        return str(definition["display_name"])
    return str(dataset_name).replace("Custom: ", "", 1)


def custom_dataset_name(display_name: str) -> str:
    clean_name = str(display_name).strip() or "User Dataset"
    if clean_name.lower().startswith("custom:"):
        return clean_name
    return f"Custom: {clean_name}"


def uploaded_manifest_entries() -> list[dict]:
    if not UPLOADED_MANIFEST_PATH.exists():
        return []
    try:
        with UPLOADED_MANIFEST_PATH.open("r", encoding="utf-8") as manifest_file:
            data = json.load(manifest_file)
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(data, dict):
        entries = data.get("datasets", [])
    else:
        entries = data
    return entries if isinstance(entries, list) else []


def write_uploaded_manifest(entries: list[dict]) -> None:
    UPLOADED_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    with UPLOADED_MANIFEST_PATH.open("w", encoding="utf-8") as manifest_file:
        json.dump({"datasets": entries}, manifest_file, indent=2)


def manifest_entry_to_dataset_name(entry: dict) -> str:
    return custom_dataset_name(str(entry.get("display_name", "User Dataset")))


def manifest_entry_to_definition(entry: dict) -> Optional[dict]:
    saved_file_path = Path(str(entry.get("saved_file_path", "")))
    if not saved_file_path.exists():
        return None
    dataset_type = entry.get("dataset_type")
    if dataset_type not in {"st", "histology"}:
        return None
    definition = {
        "kind": dataset_type,
        "source": "uploaded",
        "dataset_id": str(
            entry.get("dataset_id")
            or f"custom_{safe_filename(saved_file_path.stem)}"
        ),
        "display_name": str(entry.get("display_name", saved_file_path.stem)),
        "file_name": str(entry.get("original_filename", saved_file_path.name)),
        "original_filename": str(entry.get("original_filename", saved_file_path.name)),
        "path": saved_file_path,
        "saved_file_path": str(saved_file_path),
        "upload_timestamp": str(entry.get("upload_timestamp", "")),
    }
    if dataset_type == "st":
        if not entry.get("x_column") or not entry.get("y_column") or not entry.get("label_column"):
            return None
        definition.update(
            {
                "x_column": entry.get("x_column"),
                "y_column": entry.get("y_column"),
                "label_column": entry.get("label_column"),
            }
        )
    return definition


def load_saved_uploaded_datasets() -> dict:
    datasets = {}
    changed = False
    valid_entries = []
    for entry in uploaded_manifest_entries():
        definition = manifest_entry_to_definition(entry)
        if definition is None:
            changed = True
            continue
        dataset_name = manifest_entry_to_dataset_name(entry)
        if dataset_name in datasets:
            dataset_name = unique_dataset_name(dataset_name, set(EXPERIMENTAL_DATASETS) | set(datasets))
        datasets[dataset_name] = definition
        valid_entries.append(entry)
    if changed:
        write_uploaded_manifest(valid_entries)
    return datasets


def ensure_uploaded_datasets_loaded() -> None:
    UPLOADED_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    if st.session_state.get("experimental_uploaded_manifest_loaded"):
        return
    st.session_state.experimental_custom_datasets = load_saved_uploaded_datasets()
    st.session_state.experimental_uploaded_manifest_loaded = True


def experimental_datasets() -> dict:
    ensure_uploaded_datasets_loaded()
    datasets = dict(EXPERIMENTAL_DATASETS)
    datasets.update(st.session_state.get("experimental_custom_datasets", {}))
    return datasets


def dataset_definition(dataset_name: str) -> dict:
    return experimental_datasets()[dataset_name]


def unique_dataset_name(base_name: str, existing: set[str]) -> str:
    clean_name = str(base_name).strip() or "Custom: User Dataset"
    if clean_name not in existing:
        return clean_name
    index = 2
    while f"{clean_name} {index}" in existing:
        index += 1
    return f"{clean_name} {index}"


def unique_custom_dataset_name(base_name: str) -> str:
    return unique_dataset_name(custom_dataset_name(base_name), set(experimental_datasets()))


def initialize_experimental_state() -> None:
    ensure_uploaded_datasets_loaded()
    dataset_defaults = {
        "left_dataset": "Allen CCF Atlas",
        "right_dataset": "Allen CCF Atlas",
    }
    all_dataset_names = set(experimental_datasets())
    for key, default in dataset_defaults.items():
        legacy_key = f"experimental_{key}"
        if key not in st.session_state:
            previous_value = st.session_state.get(legacy_key, default)
            st.session_state[key] = (
                previous_value
                if previous_value in all_dataset_names
                else default
            )
        elif st.session_state[key] not in all_dataset_names:
            st.session_state[key] = default
        st.session_state.pop(legacy_key, None)

    defaults = {
        "experimental_pairs": [],
        "experimental_next_group_id": 1,
        "experimental_custom_regions": {},
        "experimental_next_custom_region_id": 1,
        "experimental_atlas_state_version": 1,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    for dataset_name in experimental_datasets():
        st.session_state.experimental_custom_regions.setdefault(dataset_name, [])
        reuse_key = experimental_key("dataset", dataset_name, "custom_region_reuse_mode")
        if st.session_state.get(reuse_key) not in CUSTOM_REGION_REUSE_MODES:
            st.session_state[reuse_key] = FLEXIBLE_REUSE_MODE
    ensure_custom_region_display_colors()

    for side in ("left", "right"):
        mode_key = f"experimental_{side}_toolbar_mode"
        if mode_key not in st.session_state:
            st.session_state[mode_key] = "pan"
        active_dataset_key = f"experimental_{side}_active_dataset"
        if (
            active_dataset_key not in st.session_state
            or st.session_state[active_dataset_key] not in all_dataset_names
        ):
            st.session_state[active_dataset_key] = st.session_state[
                f"{side}_dataset"
            ]
        for dataset_name, definition in experimental_datasets().items():
            values = {
                "selected": [],
                "selected_custom_regions": [],
                "last_click": None,
                "selection_message": "",
                "processed_click": None,
                "suppress_next_click": False,
            }
            if definition["kind"] == "st":
                values["rotation"] = 0.0
            elif definition["kind"] == "histology":
                values["rotation"] = 0.0
            else:
                values.update(
                    {
                        "z_slice": DEFAULT_ATLAS_Z_SLICE,
                        "flip_vertical": False,
                        "flip_horizontal": False,
                        "color_mode": "Subsection colors",
                    }
                )
            for field, value in values.items():
                key = experimental_key(side, dataset_name, field)
                if key not in st.session_state:
                    st.session_state[key] = value

    if st.session_state.experimental_atlas_state_version < 2:
        for side in ("left", "right"):
            st.session_state[
                experimental_key(side, "Allen CCF Atlas", "z_slice")
            ] = DEFAULT_ATLAS_Z_SLICE
        st.session_state.experimental_atlas_state_version = 2


@st.cache_data(show_spinner="Loading ST dataset...")
def load_experimental_cluster_csv(path_text: str, modified_time: float) -> pd.DataFrame:
    del modified_time
    frame = pd.read_csv(path_text)
    required = ["x", "y", "cluster"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    frame = frame[required].dropna().copy()
    frame["cluster"] = frame["cluster"].astype(int)
    return frame


@st.cache_data(show_spinner="Loading uploaded point dataset...")
def load_custom_cluster_csv(
    file_bytes: bytes,
    x_column: str,
    y_column: str,
    label_column: str,
) -> pd.DataFrame:
    frame = pd.read_csv(BytesIO(file_bytes))
    selected = frame[[x_column, y_column, label_column]].dropna().copy()
    selected = selected.rename(
        columns={
            x_column: "x",
            y_column: "y",
            label_column: "cluster",
        }
    )
    selected["x"] = pd.to_numeric(selected["x"], errors="coerce")
    selected["y"] = pd.to_numeric(selected["y"], errors="coerce")
    selected["cluster"] = pd.to_numeric(selected["cluster"], errors="coerce")
    selected = selected.dropna().copy()
    selected["cluster"] = selected["cluster"].astype(int)
    return selected


@st.cache_data(show_spinner="Loading saved point dataset...")
def load_custom_cluster_csv_from_path(
    path_text: str,
    modified_time: float,
    x_column: str,
    y_column: str,
    label_column: str,
) -> pd.DataFrame:
    del modified_time
    frame = pd.read_csv(path_text)
    selected = frame[[x_column, y_column, label_column]].dropna().copy()
    selected = selected.rename(
        columns={
            x_column: "x",
            y_column: "y",
            label_column: "cluster",
        }
    )
    selected["x"] = pd.to_numeric(selected["x"], errors="coerce")
    selected["y"] = pd.to_numeric(selected["y"], errors="coerce")
    selected["cluster"] = pd.to_numeric(selected["cluster"], errors="coerce")
    selected = selected.dropna().copy()
    selected["cluster"] = selected["cluster"].astype(int)
    return selected


@st.cache_data(show_spinner="Loading uploaded label image...")
def load_custom_histology_labels(file_bytes: bytes) -> np.ndarray:
    labels = np.load(BytesIO(file_bytes)).astype(float)
    if labels.ndim != 2:
        raise ValueError("Custom NPY label images must be 2D.")
    labels[labels == -1] = np.nan
    return labels


@st.cache_data(show_spinner="Loading saved label image...")
def load_custom_histology_labels_from_path(
    path_text: str,
    modified_time: float,
) -> np.ndarray:
    del modified_time
    labels = np.load(path_text).astype(float)
    if labels.ndim != 2:
        raise ValueError("Custom NPY label images must be 2D.")
    labels[labels == -1] = np.nan
    return labels


@st.cache_data(show_spinner="Preparing uploaded label image...")
def rotated_custom_histology_labels(
    file_bytes: bytes,
    rotation_degrees: float,
) -> tuple[np.ndarray, tuple[int, int, int, int], dict[int, int]]:
    labels = load_custom_histology_labels(file_bytes)
    rotated = ndimage_rotate(
        labels,
        angle=float(rotation_degrees),
        reshape=True,
        order=0,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )
    valid = ~np.isnan(rotated)
    if not valid.any():
        raise ValueError("The rotated custom label image contains no visible labels.")

    rotated_h, rotated_w = rotated.shape
    image_bounds = (0, rotated_h, 0, rotated_w)
    source_valid = labels[~np.isnan(labels)].astype(int)
    unique_labels, counts = np.unique(source_valid, return_counts=True)
    pixel_counts = {
        int(label): int(count)
        for label, count in zip(unique_labels, counts)
    }
    return rotated, image_bounds, pixel_counts


@st.cache_data(show_spinner="Preparing saved label image...")
def rotated_custom_histology_labels_from_path(
    path_text: str,
    modified_time: float,
    rotation_degrees: float,
) -> tuple[np.ndarray, tuple[int, int, int, int], dict[int, int]]:
    labels = load_custom_histology_labels_from_path(path_text, modified_time)
    rotated = ndimage_rotate(
        labels,
        angle=float(rotation_degrees),
        reshape=True,
        order=0,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )
    valid = ~np.isnan(rotated)
    if not valid.any():
        raise ValueError("The rotated custom label image contains no visible labels.")

    rotated_h, rotated_w = rotated.shape
    image_bounds = (0, rotated_h, 0, rotated_w)
    source_valid = labels[~np.isnan(labels)].astype(int)
    unique_labels, counts = np.unique(source_valid, return_counts=True)
    pixel_counts = {
        int(label): int(count)
        for label, count in zip(unique_labels, counts)
    }
    return rotated, image_bounds, pixel_counts


def likely_column(columns: list[str], candidates: list[str]) -> Optional[str]:
    lower_lookup = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lower_lookup:
            return lower_lookup[candidate.lower()]
    for column in columns:
        lower = column.lower()
        if any(candidate.lower() in lower for candidate in candidates):
            return column
    return None


def custom_dataset_defaults(frame: pd.DataFrame) -> tuple[Optional[str], Optional[str], Optional[str]]:
    columns = list(frame.columns)
    x_column = likely_column(columns, ["x", "xcoord", "x_coord", "pixel_x", "col"])
    y_column = likely_column(columns, ["y", "ycoord", "y_coord", "pixel_y", "row"])
    label_column = likely_column(
        columns,
        ["cluster", "label", "region", "region_id", "cluster_id", "id"],
    )
    return x_column, y_column, label_column


def add_custom_dataset(name: str, definition: dict) -> str:
    file_bytes = definition.get("data_bytes")
    if not file_bytes:
        raise ValueError("Uploaded dataset bytes are missing.")

    dataset_name = unique_custom_dataset_name(name)
    display_name = dataset_display_name(dataset_name)
    existing_ids = {
        str(item.get("dataset_id"))
        for item in uploaded_manifest_entries()
        if item.get("dataset_id")
    }
    base_dataset_id = f"custom_{uuid4().hex[:12]}"
    dataset_id = base_dataset_id
    suffix = 2
    while dataset_id in existing_ids:
        dataset_id = f"{base_dataset_id}_{suffix}"
        suffix += 1

    UPLOADED_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    original_filename = definition.get("file_name", "uploaded_dataset")
    saved_filename = f"{dataset_id}_{safe_filename(original_filename)}"
    saved_file_path = UPLOADED_DATASETS_DIR / saved_filename
    with saved_file_path.open("wb") as dataset_file:
        dataset_file.write(file_bytes)

    upload_timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    saved_definition = {
        key: value
        for key, value in definition.items()
        if key != "data_bytes"
    }
    saved_definition.update(
        {
            "source": "uploaded",
            "dataset_id": dataset_id,
            "display_name": display_name,
            "file_name": original_filename,
            "original_filename": original_filename,
            "path": saved_file_path,
            "saved_file_path": str(saved_file_path),
            "upload_timestamp": upload_timestamp,
        }
    )
    manifest_entry = {
        "dataset_id": dataset_id,
        "display_name": display_name,
        "dataset_type": saved_definition["kind"],
        "original_filename": original_filename,
        "saved_file_path": str(saved_file_path),
        "upload_timestamp": upload_timestamp,
    }
    if saved_definition["kind"] == "st":
        manifest_entry.update(
            {
                "x_column": saved_definition["x_column"],
                "y_column": saved_definition["y_column"],
                "label_column": saved_definition["label_column"],
            }
        )

    entries = uploaded_manifest_entries()
    entries.append(manifest_entry)
    write_uploaded_manifest(entries)

    st.session_state.experimental_custom_datasets[dataset_name] = saved_definition
    st.session_state.experimental_custom_regions.setdefault(dataset_name, [])
    reuse_key = custom_region_reuse_key(dataset_name)
    if st.session_state.get(reuse_key) not in CUSTOM_REGION_REUSE_MODES:
        st.session_state[reuse_key] = FLEXIBLE_REUSE_MODE
    for side in ("left", "right"):
        values = {
            "selected": [],
            "selected_custom_regions": [],
            "last_click": None,
            "selection_message": "",
            "processed_click": None,
            "suppress_next_click": False,
            "rotation": 0.0,
        }
        for field, value in values.items():
            key = experimental_key(side, dataset_name, field)
            if key not in st.session_state:
                st.session_state[key] = value
    return dataset_name


def delete_uploaded_dataset(dataset_name: str) -> None:
    custom_datasets = st.session_state.get("experimental_custom_datasets", {})
    definition = custom_datasets.get(dataset_name)
    if not definition:
        return

    dataset_id = str(definition.get("dataset_id", ""))
    entries = [
        entry
        for entry in uploaded_manifest_entries()
        if str(entry.get("dataset_id", "")) != dataset_id
    ]
    write_uploaded_manifest(entries)

    saved_file_path = Path(str(definition.get("saved_file_path", definition.get("path", ""))))
    try:
        upload_dir = UPLOADED_DATASETS_DIR.resolve()
        resolved_path = saved_file_path.resolve()
        if resolved_path.exists() and resolved_path.parent == upload_dir:
            resolved_path.unlink()
    except OSError:
        pass

    custom_datasets.pop(dataset_name, None)
    st.session_state.experimental_custom_regions.pop(dataset_name, None)

    for side in ("left", "right"):
        if st.session_state.get(f"{side}_dataset") == dataset_name:
            st.session_state[f"{side}_dataset"] = "Allen CCF Atlas"
        active_key = f"experimental_{side}_active_dataset"
        if st.session_state.get(active_key) == dataset_name:
            st.session_state[active_key] = st.session_state.get(f"{side}_dataset", "Allen CCF Atlas")
        prefix = experimental_key(side, dataset_name, "")
        for key in list(st.session_state.keys()):
            if str(key).startswith(prefix):
                st.session_state.pop(key, None)
    dataset_prefix = experimental_key("dataset", dataset_name, "")
    for key in list(st.session_state.keys()):
        if str(key).startswith(dataset_prefix):
            st.session_state.pop(key, None)

    st.session_state.experimental_custom_datasets = custom_datasets
    request_experimental_rerun()


def render_custom_dataset_upload() -> None:
    with st.expander("Upload custom dataset", expanded=False):
        st.caption(
            "Upload a point/cluster CSV or a 2D NPY label image. "
            "Custom NRRD upload is not enabled yet; the built-in Allen CCF atlas still uses annotation_10.nrrd."
        )
        upload_name = st.text_input(
            "Custom dataset name",
            key="custom_dataset_upload_name",
            placeholder="Experiment A",
        )
        dataset_type_label = st.radio(
            "Dataset type",
            list(CUSTOM_DATASET_TYPES),
            key="custom_dataset_upload_type",
            horizontal=True,
        )
        uploaded_file = st.file_uploader(
            "Upload custom dataset file",
            type=["csv", "npy"],
            key="custom_dataset_upload_file",
        )
        if uploaded_file is not None:
            uploaded_file.seek(0)
            file_bytes = uploaded_file.read()
            dataset_kind = CUSTOM_DATASET_TYPES[dataset_type_label]
            definition = {
                "kind": dataset_kind,
                "source": "upload",
                "file_name": uploaded_file.name,
                "data_bytes": file_bytes,
            }

            histology_upload_valid = False
            if dataset_kind == "st":
                try:
                    preview = pd.read_csv(BytesIO(file_bytes), nrows=100)
                except Exception as error:
                    st.error(f"Could not read the uploaded CSV: {error}")
                    preview = None
                if preview is not None:
                    columns = list(preview.columns)
                    if len(columns) < 3:
                        st.error("Custom CSV datasets need at least x, y, and cluster/label columns.")
                    else:
                        detected_x, detected_y, detected_label = custom_dataset_defaults(preview)
                        x_index = columns.index(detected_x) if detected_x in columns else 0
                        y_index = columns.index(detected_y) if detected_y in columns else min(1, len(columns) - 1)
                        label_index = (
                            columns.index(detected_label)
                            if detected_label in columns
                            else min(2, len(columns) - 1)
                        )
                        col_1, col_2, col_3 = st.columns(3)
                        with col_1:
                            x_column = st.selectbox(
                                "x column",
                                columns,
                                index=x_index,
                                key="custom_dataset_x_column",
                            )
                        with col_2:
                            y_column = st.selectbox(
                                "y column",
                                columns,
                                index=y_index,
                                key="custom_dataset_y_column",
                            )
                        with col_3:
                            label_column = st.selectbox(
                                "cluster/label column",
                                columns,
                                index=label_index,
                                key="custom_dataset_label_column",
                            )
                        definition.update(
                            {
                                "x_column": x_column,
                                "y_column": y_column,
                                "label_column": label_column,
                            }
                        )
            else:
                try:
                    labels = load_custom_histology_labels(file_bytes)
                    st.caption(f"Loaded label image shape: {labels.shape[0]} x {labels.shape[1]}")
                    histology_upload_valid = True
                except Exception as error:
                    st.error(f"Could not read the uploaded NPY file: {error}")

            ready_to_add = (
                (dataset_kind == "histology" and histology_upload_valid)
                or {"x_column", "y_column", "label_column"}.issubset(definition)
            )
            if st.button(
                "Add custom dataset",
                key="add_custom_dataset",
                disabled=not ready_to_add,
            ):
                try:
                    dataset_name = add_custom_dataset(upload_name or uploaded_file.name, definition)
                except Exception as error:
                    st.error(f"Could not save the uploaded dataset: {error}")
                else:
                    st.success(f"Added {dataset_name}")
                    request_experimental_rerun()

        uploaded_dataset_names = sorted(st.session_state.get("experimental_custom_datasets", {}))
        if uploaded_dataset_names:
            st.divider()
            st.caption("Manage saved uploaded datasets.")
            delete_name = st.selectbox(
                "Uploaded datasets",
                uploaded_dataset_names,
                key="custom_dataset_delete_name",
            )
            st.button(
                "Delete uploaded dataset",
                key="delete_custom_dataset",
                on_click=delete_uploaded_dataset,
                args=(delete_name,),
            )


def panel_selected(side: str, dataset_name: str) -> list[dict]:
    return st.session_state[experimental_key(side, dataset_name, "selected")]


def panel_selected_ids(side: str, dataset_name: str) -> set[int]:
    definition = dataset_definition(dataset_name)
    id_field = "cluster_id" if definition["kind"] == "st" else "label"
    return {int(item[id_field]) for item in panel_selected(side, dataset_name)}


def source_region_type(dataset_name: str) -> str:
    kind = dataset_definition(dataset_name)["kind"]
    if kind == "st":
        return "st_cluster"
    if kind == "atlas":
        return "atlas_label"
    return "histology_label"


def custom_regions_for_dataset(dataset_name: str) -> list[dict]:
    return st.session_state.experimental_custom_regions.setdefault(dataset_name, [])


def custom_region_color(custom_region_id: str) -> str:
    digits = "".join(character for character in str(custom_region_id) if character.isdigit())
    color_seed = int(digits) if digits else abs(hash(str(custom_region_id))) % 100000
    return rgb_to_hex(distinct_label_color(color_seed + 1000))


def ensure_custom_region_display_colors() -> None:
    for regions in st.session_state.experimental_custom_regions.values():
        for region in regions:
            if "display_color" not in region:
                region["display_color"] = custom_region_color(region["custom_region_id"])


def custom_region_reuse_key(dataset_name: str) -> str:
    return experimental_key("dataset", dataset_name, "custom_region_reuse_mode")


def custom_region_reuse_mode(dataset_name: str) -> str:
    key = custom_region_reuse_key(dataset_name)
    mode = st.session_state.get(key, FLEXIBLE_REUSE_MODE)
    if mode not in CUSTOM_REGION_REUSE_MODES:
        mode = FLEXIBLE_REUSE_MODE
        st.session_state[key] = mode
    return mode


def assigned_region_ids(dataset_name: str) -> set[int]:
    assigned: set[int] = set()
    for region in custom_regions_for_dataset(dataset_name):
        assigned.update(int(value) for value in region.get("included_region_ids", []))
    return assigned


def assigned_region_color_map(dataset_name: str) -> dict[int, tuple[int, int, int]]:
    color_map: dict[int, tuple[int, int, int]] = {}
    for region in custom_regions_for_dataset(dataset_name):
        color = hex_to_rgb(region.get("display_color", custom_region_color(region["custom_region_id"])))
        for value in region.get("included_region_ids", []):
            region_id = int(value)
            color_map.setdefault(region_id, color)
    return color_map


def assigned_region_message(dataset_name: str, region_id: int) -> str:
    for region in custom_regions_for_dataset(dataset_name):
        if int(region_id) in {int(value) for value in region.get("included_region_ids", [])}:
            return (
                "This region is already assigned to a custom region: "
                f"{region['custom_region_name']}."
            )
    return "This region is already assigned to a custom region."


def custom_region_lookup(dataset_name: str) -> dict[str, dict]:
    return {
        str(region["custom_region_id"]): region
        for region in custom_regions_for_dataset(dataset_name)
    }


def panel_selected_custom_region_ids(side: str, dataset_name: str) -> list[str]:
    key = experimental_key(side, dataset_name, "selected_custom_regions")
    lookup = custom_region_lookup(dataset_name)
    selected = [
        str(region_id)
        for region_id in st.session_state.get(key, [])
        if str(region_id) in lookup
    ]
    if selected != st.session_state.get(key, []):
        st.session_state[key] = selected
    return selected


def panel_selected_custom_regions(side: str, dataset_name: str) -> list[dict]:
    lookup = custom_region_lookup(dataset_name)
    return [
        lookup[region_id]
        for region_id in panel_selected_custom_region_ids(side, dataset_name)
        if region_id in lookup
    ]


def panel_effective_selected_ids(side: str, dataset_name: str) -> set[int]:
    selected_ids = set(panel_selected_ids(side, dataset_name))
    for region in panel_selected_custom_regions(side, dataset_name):
        selected_ids.update(int(value) for value in region["included_region_ids"])
    return selected_ids


def visible_assigned_ids(side: str, dataset_name: str) -> set[int]:
    if custom_region_reuse_mode(dataset_name) != EXCLUSIVE_REUSE_MODE:
        return set()
    return assigned_region_ids(dataset_name) - panel_effective_selected_ids(side, dataset_name)


def visible_assigned_color_map(side: str, dataset_name: str) -> dict[int, tuple[int, int, int]]:
    if custom_region_reuse_mode(dataset_name) != EXCLUSIVE_REUSE_MODE:
        return {}
    active_ids = panel_effective_selected_ids(side, dataset_name)
    return {
        region_id: color
        for region_id, color in assigned_region_color_map(dataset_name).items()
        if region_id not in active_ids
    }


def selected_custom_region_color_map(side: str, dataset_name: str) -> dict[int, tuple[int, int, int]]:
    color_map: dict[int, tuple[int, int, int]] = {}
    for region in panel_selected_custom_regions(side, dataset_name):
        color = hex_to_rgb(region.get("display_color", custom_region_color(region["custom_region_id"])))
        for value in region.get("included_region_ids", []):
            color_map[int(value)] = color
    return color_map


def selection_message_key(side: str, dataset_name: str) -> str:
    return experimental_key(side, dataset_name, "selection_message")


def set_panel_selection_message(side: str, dataset_name: str, message: str) -> None:
    st.session_state[selection_message_key(side, dataset_name)] = message


def clear_panel_selection_message(side: str, dataset_name: str) -> None:
    set_panel_selection_message(side, dataset_name, "")


def can_select_region_id(side: str, dataset_name: str, region_id: int) -> bool:
    raw_selected_ids = panel_selected_ids(side, dataset_name)
    if int(region_id) in raw_selected_ids:
        return True
    if (
        custom_region_reuse_mode(dataset_name) == EXCLUSIVE_REUSE_MODE
        and int(region_id) in assigned_region_ids(dataset_name)
    ):
        set_panel_selection_message(
            side,
            dataset_name,
            assigned_region_message(dataset_name, int(region_id)),
        )
        return False
    clear_panel_selection_message(side, dataset_name)
    return True


def selection_ids_for_items(dataset_name: str, selected: list[dict]) -> list[int]:
    id_field = "cluster_id" if dataset_definition(dataset_name)["kind"] == "st" else "label"
    return sorted({int(item[id_field]) for item in selected})


def custom_region_counts(dataset_name: str, selected: list[dict]) -> dict:
    kind = dataset_definition(dataset_name)["kind"]
    if kind == "st":
        point_count = int(sum(int(item.get("cluster_point_count", 0)) for item in selected))
        return {
            "point_count": point_count,
            "pixel_count": "",
            "total_voxel_count": "",
            "point_count_or_pixel_count": point_count,
        }
    if kind == "histology":
        pixel_count = int(sum(int(item.get("pixel_count", 0)) for item in selected))
        return {
            "point_count": "",
            "pixel_count": pixel_count,
            "total_voxel_count": "",
            "point_count_or_pixel_count": pixel_count,
        }

    voxel_counts = pd.to_numeric(
        pd.Series([item.get("total_voxel_count", np.nan) for item in selected]),
        errors="coerce",
    ).dropna()
    total_voxel_count = int(voxel_counts.sum()) if not voxel_counts.empty else ""
    return {
        "point_count": "",
        "pixel_count": "",
        "total_voxel_count": total_voxel_count,
        "point_count_or_pixel_count": total_voxel_count,
    }


def create_custom_region_from_selection(side: str, dataset_name: str) -> None:
    selected = list(panel_selected(side, dataset_name))
    included_ids = selection_ids_for_items(dataset_name, selected)
    if not included_ids:
        return
    reuse_mode = custom_region_reuse_mode(dataset_name)
    if reuse_mode == EXCLUSIVE_REUSE_MODE:
        conflicts = sorted(set(included_ids) & assigned_region_ids(dataset_name))
        if conflicts:
            set_panel_selection_message(
                side,
                dataset_name,
                (
                    "This region is already assigned to a custom region: "
                    + ", ".join(map(str, conflicts))
                ),
            )
            return

    name_key = experimental_key(side, dataset_name, "custom_region_name")
    fallback_name = f"Custom Region {st.session_state.experimental_next_custom_region_id}"
    region_name = str(st.session_state.get(name_key, fallback_name)).strip() or fallback_name
    region_id = f"custom_region_{st.session_state.experimental_next_custom_region_id}"
    st.session_state.experimental_next_custom_region_id += 1

    custom_region = {
        "custom_region_id": region_id,
        "custom_region_name": region_name,
        "display_color": custom_region_color(region_id),
        "source_dataset": dataset_name,
        "source_panel": side,
        "source_region_type": source_region_type(dataset_name),
        "included_region_ids": included_ids,
        "assigned_region_ids": included_ids,
        "reuse_mode": reuse_mode,
        "created_from": "merged_existing_regions",
        **custom_region_counts(dataset_name, selected),
    }
    custom_regions_for_dataset(dataset_name).append(custom_region)

    selected_custom_key = experimental_key(side, dataset_name, "selected_custom_regions")
    selected_custom = panel_selected_custom_region_ids(side, dataset_name)
    if region_id not in selected_custom:
        selected_custom.append(region_id)
    st.session_state[selected_custom_key] = selected_custom

    clear_panel_raw_selection(side, dataset_name)
    clear_panel_selection_message(side, dataset_name)
    st.session_state[name_key] = f"Custom Region {st.session_state.experimental_next_custom_region_id}"


def rename_custom_region(dataset_name: str, region_id: str, name_key: str) -> None:
    next_name = str(st.session_state.get(name_key, "")).strip()
    if not next_name:
        return
    for region in custom_regions_for_dataset(dataset_name):
        if str(region["custom_region_id"]) == str(region_id):
            region["custom_region_name"] = next_name
            return


def delete_custom_region(dataset_name: str, region_id: str) -> None:
    st.session_state.experimental_custom_regions[dataset_name] = [
        region
        for region in custom_regions_for_dataset(dataset_name)
        if str(region["custom_region_id"]) != str(region_id)
    ]
    for side in ("left", "right"):
        selected_key = experimental_key(side, dataset_name, "selected_custom_regions")
        st.session_state[selected_key] = [
            selected_id
            for selected_id in st.session_state.get(selected_key, [])
            if str(selected_id) != str(region_id)
        ]


def clear_panel_raw_selection(side: str, dataset_name: str) -> None:
    st.session_state[experimental_key(side, dataset_name, "selected")] = []
    st.session_state[experimental_key(side, dataset_name, "last_click")] = None
    clear_panel_selection_message(side, dataset_name)
    st.session_state[experimental_key(side, dataset_name, "suppress_next_click")] = True

    if dataset_definition(dataset_name)["kind"] == "atlas":
        st.session_state.pop(
            experimental_key(side, dataset_name, "image_cache_signature"),
            None,
        )


def clear_panel_selection(side: str, dataset_name: str) -> None:
    clear_panel_raw_selection(side, dataset_name)
    st.session_state[experimental_key(side, dataset_name, "selected_custom_regions")] = []


def change_panel_dataset(side: str) -> None:
    dataset_key = f"{side}_dataset"
    active_dataset_key = f"experimental_{side}_active_dataset"
    next_dataset = st.session_state[dataset_key]
    previous_dataset = st.session_state.get(active_dataset_key, next_dataset)
    if previous_dataset == next_dataset:
        return
    available_datasets = experimental_datasets()
    if previous_dataset in available_datasets:
        clear_panel_selection(side, previous_dataset)
    if next_dataset in available_datasets:
        clear_panel_selection(side, next_dataset)
    st.session_state[active_dataset_key] = next_dataset


def request_experimental_rerun() -> None:
    st.session_state.experimental_deferred_rerun = True


def reset_panel_rotation(side: str, dataset_name: str) -> None:
    st.session_state[experimental_key(side, dataset_name, "rotation")] = 0.0


def reset_panel_atlas_orientation(side: str, dataset_name: str) -> None:
    st.session_state[experimental_key(side, dataset_name, "flip_vertical")] = False
    st.session_state[experimental_key(side, dataset_name, "flip_horizontal")] = False


def sync_panel_atlas_z(
    widget_key: str,
    state_key: str,
    max_z: int,
) -> None:
    value = st.session_state.get(widget_key, DEFAULT_ATLAS_Z_SLICE)
    st.session_state[state_key] = int(
        np.clip(value, 0, max_z)
    )


def toggle_panel_cluster(
    side: str,
    dataset_name: str,
    cluster_id: int,
    x: float,
    y: float,
    point_counts: dict[int, int],
) -> None:
    selected = list(panel_selected(side, dataset_name))
    existing_ids = {int(item["cluster_id"]) for item in selected}
    if cluster_id in existing_ids:
        selected = [
            item for item in selected if int(item["cluster_id"]) != int(cluster_id)
        ]
        clear_panel_selection_message(side, dataset_name)
    elif not can_select_region_id(side, dataset_name, int(cluster_id)):
        st.session_state[experimental_key(side, dataset_name, "last_click")] = {
            "cluster_id": int(cluster_id),
            "x": float(x),
            "y": float(y),
        }
        return
    else:
        selected.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_point_count": int(point_counts.get(cluster_id, 0)),
                "x": float(x),
                "y": float(y),
            }
        )
    st.session_state[experimental_key(side, dataset_name, "selected")] = selected
    st.session_state[experimental_key(side, dataset_name, "last_click")] = {
        "cluster_id": int(cluster_id),
        "x": float(x),
        "y": float(y),
    }


def toggle_panel_atlas(
    side: str,
    dataset_name: str,
    label: int,
    x: int,
    y: int,
    metadata_lookup: dict[int, dict],
) -> None:
    if label == 0:
        st.session_state[experimental_key(side, dataset_name, "last_click")] = {
            "label": 0,
            "atlas_region": "background",
            "name": "Background",
            "x": x,
            "y": y,
        }
        return

    selected = list(panel_selected(side, dataset_name))
    existing_ids = {int(item["label"]) for item in selected}
    if label in existing_ids:
        selected = [item for item in selected if int(item["label"]) != int(label)]
        clear_panel_selection_message(side, dataset_name)
    elif not can_select_region_id(side, dataset_name, int(label)):
        st.session_state[experimental_key(side, dataset_name, "last_click")] = {
            **label_details(label, metadata_lookup),
            "x": x,
            "y": y,
        }
        return
    else:
        item = label_details(label, metadata_lookup)
        item["x"] = x
        item["y"] = y
        selected.append(item)
    st.session_state[experimental_key(side, dataset_name, "selected")] = selected
    clicked = label_details(label, metadata_lookup)
    clicked["x"] = x
    clicked["y"] = y
    st.session_state[experimental_key(side, dataset_name, "last_click")] = clicked


def toggle_panel_histology(
    side: str,
    dataset_name: str,
    label: int,
    x: float,
    y: float,
    pixel_counts: dict[int, int],
) -> None:
    selected = list(panel_selected(side, dataset_name))
    existing_ids = {int(item["label"]) for item in selected}
    if label in existing_ids:
        selected = [item for item in selected if int(item["label"]) != int(label)]
        clear_panel_selection_message(side, dataset_name)
    elif not can_select_region_id(side, dataset_name, int(label)):
        st.session_state[experimental_key(side, dataset_name, "last_click")] = {
            "label": int(label),
            "histology_region": f"label_{int(label)}",
            "x": float(x),
            "y": float(y),
        }
        return
    else:
        selected.append(
            {
                "label": int(label),
                "histology_region": f"label_{int(label)}",
                "pixel_count": int(pixel_counts.get(int(label), 0)),
                "x": float(x),
                "y": float(y),
            }
        )
    st.session_state[experimental_key(side, dataset_name, "selected")] = selected
    st.session_state[experimental_key(side, dataset_name, "last_click")] = {
        "label": int(label),
        "histology_region": f"label_{int(label)}",
        "x": float(x),
        "y": float(y),
    }


def process_panel_mode_event(side: str, event: dict) -> bool:
    if event.get("type") != "mode" or not event.get("eventId"):
        return False
    mode_key = f"experimental_{side}_toolbar_mode"
    next_mode = event.get("mode", "pan")
    if next_mode == st.session_state[mode_key]:
        return False
    st.session_state[mode_key] = next_mode
    return True


def unprocessed_panel_click(
    side: str,
    dataset_name: str,
    event: dict,
) -> Optional[dict]:
    click = first_plot_click(event)
    if click is None:
        return None
    event_mode = event.get("mode")
    if event_mode in ("pan", "select", "zoom"):
        st.session_state[f"experimental_{side}_toolbar_mode"] = event_mode
    processed_key = experimental_key(side, dataset_name, "processed_click")
    suppress_key = experimental_key(side, dataset_name, "suppress_next_click")
    event_id = event.get("eventId")
    if event_id == st.session_state[processed_key]:
        if st.session_state[suppress_key]:
            st.session_state[suppress_key] = False
        return None
    if st.session_state[suppress_key]:
        st.session_state[suppress_key] = False
    st.session_state[processed_key] = event_id
    return click


def sync_custom_region_reuse_mode(widget_key: str, dataset_name: str) -> None:
    mode = st.session_state.get(widget_key, FLEXIBLE_REUSE_MODE)
    if mode not in CUSTOM_REGION_REUSE_MODES:
        mode = FLEXIBLE_REUSE_MODE
    st.session_state[custom_region_reuse_key(dataset_name)] = mode


def display_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.astype(str)


def mode_badge(side: str) -> None:
    mode = st.session_state[f"experimental_{side}_toolbar_mode"].title()
    st.caption(f"Current mode: {mode}")


def render_panel_custom_region_controls(side: str, dataset_name: str) -> None:
    raw_selected_ids = sorted(panel_selected_ids(side, dataset_name))
    name_key = experimental_key(side, dataset_name, "custom_region_name")
    if name_key not in st.session_state:
        st.session_state[name_key] = (
            f"Custom Region {st.session_state.experimental_next_custom_region_id}"
        )

    st.caption(
        "Flexible reuse mode allows regions in multiple custom regions. "
        "Exclusive assignment mode allows each region to belong to only one custom region."
    )
    mode_widget_key = experimental_key(side, dataset_name, "custom_region_reuse_mode_widget")
    st.session_state[mode_widget_key] = custom_region_reuse_mode(dataset_name)
    st.radio(
        "Custom region reuse mode",
        CUSTOM_REGION_REUSE_MODES,
        key=mode_widget_key,
        horizontal=True,
        on_change=sync_custom_region_reuse_mode,
        args=(mode_widget_key, dataset_name),
    )
    assigned_ids = sorted(assigned_region_ids(dataset_name))
    if custom_region_reuse_mode(dataset_name) == EXCLUSIVE_REUSE_MODE and assigned_ids:
        st.caption(
            "Assigned region IDs: "
            + ", ".join(map(str, assigned_ids))
        )

    selection_message = st.session_state.get(selection_message_key(side, dataset_name), "")
    if selection_message:
        st.warning(selection_message)

    if raw_selected_ids:
        st.text_input(
            "Custom region name",
            key=name_key,
            placeholder="Hippocampus candidate",
        )
        st.button(
            "Create custom region",
            key=f"experimental_{side}_{dataset_name}_create_custom_region",
            use_container_width=True,
            on_click=create_custom_region_from_selection,
            args=(side, dataset_name),
        )
    else:
        st.caption("Select one or more regions in the viewer to create a custom region.")

    regions = custom_regions_for_dataset(dataset_name)
    if not regions:
        return

    options = [str(region["custom_region_id"]) for region in regions]
    selected_key = experimental_key(side, dataset_name, "selected_custom_regions")
    st.session_state[selected_key] = [
        region_id
        for region_id in st.session_state.get(selected_key, [])
        if region_id in options
    ]
    label_lookup = {
        str(region["custom_region_id"]): (
            f"{region['custom_region_name']} "
            f"({', '.join(map(str, region['included_region_ids']))})"
        )
        for region in regions
    }
    st.multiselect(
        "Use custom regions",
        options=options,
        key=selected_key,
        format_func=lambda region_id: label_lookup.get(str(region_id), str(region_id)),
    )


def render_st_panel(side: str, dataset_name: str) -> None:
    definition = dataset_definition(dataset_name)
    data_path = definition.get("path")
    if data_path is not None and not data_path.exists():
        st.error(f"Missing ST dataset: {data_path}")
        return

    rotation_key = experimental_key(side, dataset_name, "rotation")
    with st.expander("Visualization controls", expanded=True):
        st.slider(
            "Rotation angle",
            min_value=0.0,
            max_value=360.0,
            step=1.0,
            key=rotation_key,
        )
        st.button(
            "Reset rotation",
            key=f"experimental_{side}_{dataset_name}_reset_rotation",
            on_click=reset_panel_rotation,
            args=(side, dataset_name),
        )

    mode = st.session_state[f"experimental_{side}_toolbar_mode"]
    mode_badge(side)

    try:
        if definition.get("source") in {"upload", "uploaded"}:
            if "data_bytes" in definition:
                cluster_df = load_custom_cluster_csv(
                    definition["data_bytes"],
                    definition["x_column"],
                    definition["y_column"],
                    definition["label_column"],
                )
            else:
                cluster_df = load_custom_cluster_csv_from_path(
                    str(data_path),
                    data_path.stat().st_mtime,
                    definition["x_column"],
                    definition["y_column"],
                    definition["label_column"],
                )
        else:
            cluster_df = load_experimental_cluster_csv(
                str(data_path),
                data_path.stat().st_mtime,
            )
    except (OSError, ValueError) as error:
        st.error(f"Could not load {dataset_name}: {error}")
        return

    transformed = transform_st_coordinates(
        cluster_df,
        st.session_state[rotation_key],
        0.0,
        0.0,
    )
    point_counts = transformed.groupby("cluster").size().astype(int).to_dict()
    figure = cluster_plotly_figure(
        transformed,
        panel_effective_selected_ids(side, dataset_name),
        scatter_bounds(transformed, 1.0),
        assigned_clusters=visible_assigned_ids(side, dataset_name),
        assigned_cluster_colors=visible_assigned_color_map(side, dataset_name),
        selected_cluster_colors=selected_custom_region_color_map(side, dataset_name),
    )
    figure.update_layout(uirevision=f"experimental-{side}-{dataset_name}")
    event = plotly_mode_events(
        figure,
        key=f"experimental_{side}_{dataset_name}_viewer",
        current_mode=mode,
    )
    if process_panel_mode_event(side, event):
        request_experimental_rerun()

    click = unprocessed_panel_click(side, dataset_name, event)
    if click is not None:
        x = float(click["x"])
        y = float(click["y"])
        cluster_id = click.get("customdata")
        if isinstance(cluster_id, (list, tuple)):
            cluster_id = cluster_id[0]
        if cluster_id is None:
            cluster_id = clicked_trace_customdata(figure, click)
        if cluster_id is None:
            cluster_id = nearest_cluster_at_point(transformed, x, y)
        toggle_panel_cluster(
            side,
            dataset_name,
            int(cluster_id),
            x,
            y,
            point_counts,
        )
        request_experimental_rerun()

    with st.expander("Selection controls", expanded=True):
        st.markdown("##### Selected Regions")
        selected = panel_selected(side, dataset_name)
        selected_ids = sorted(panel_selected_ids(side, dataset_name))
        st.caption(
            f"Selected clusters: {', '.join(map(str, selected_ids))}"
            if selected_ids
            else "No clusters selected."
        )
        st.button(
            "Clear selected regions",
            key=f"experimental_{side}_{dataset_name}_clear_clusters",
            use_container_width=True,
            disabled=panel_pair_count(side, dataset_name) == 0,
            on_click=clear_panel_selection,
            args=(side, dataset_name),
        )
        st.dataframe(
            display_dataframe(
                pd.DataFrame(
                    selected,
                    columns=["cluster_id", "cluster_point_count", "x", "y"],
                )
            ),
            width="stretch",
            height=220,
            hide_index=True,
        )
    with st.expander("Custom regions", expanded=False):
        render_panel_custom_region_controls(side, dataset_name)


def render_histology_panel(side: str, dataset_name: str) -> None:
    definition = dataset_definition(dataset_name)
    data_path = definition.get("path")
    if data_path is not None and not data_path.exists():
        st.error(f"Missing histology label image: {data_path}")
        return

    rotation_key = experimental_key(side, dataset_name, "rotation")
    with st.expander("Visualization controls", expanded=True):
        st.slider(
            "Rotation angle",
            min_value=0.0,
            max_value=360.0,
            step=1.0,
            key=rotation_key,
        )
        st.button(
            "Reset rotation",
            key=f"experimental_{side}_{dataset_name}_reset_rotation",
            on_click=reset_panel_rotation,
            args=(side, dataset_name),
        )

    mode = st.session_state[f"experimental_{side}_toolbar_mode"]
    mode_badge(side)

    try:
        if definition.get("source") in {"upload", "uploaded"}:
            if "data_bytes" in definition:
                rotated_labels, image_bounds, pixel_counts = rotated_custom_histology_labels(
                    definition["data_bytes"],
                    st.session_state[rotation_key],
                )
            else:
                rotated_labels, image_bounds, pixel_counts = rotated_custom_histology_labels_from_path(
                    str(data_path),
                    data_path.stat().st_mtime,
                    st.session_state[rotation_key],
                )
        else:
            rotated_labels, image_bounds, pixel_counts = rotated_histology_labels(
                str(data_path),
                data_path.stat().st_mtime,
                st.session_state[rotation_key],
            )
    except (OSError, ValueError) as error:
        st.error(f"Could not load {dataset_name}: {error}")
        return

    rotation_token = f"{float(st.session_state[rotation_key]) % 360.0:.1f}"
    image = histology_label_image(
        rotated_labels,
        panel_effective_selected_ids(side, dataset_name),
        visible_assigned_ids(side, dataset_name),
        visible_assigned_color_map(side, dataset_name),
        selected_custom_region_color_map(side, dataset_name),
    )
    figure = atlas_plotly_figure(
        image,
        rotated_labels,
        image_bounds,
        uirevision=f"experimental-{side}-{dataset_name}-{rotation_token}",
    )
    event = plotly_mode_events(
        figure,
        key=f"experimental_{side}_{dataset_name}_viewer_{rotation_token}",
        current_mode=mode,
        height=HISTOLOGY_VIEWER_HEIGHT,
    )
    if process_panel_mode_event(side, event):
        request_experimental_rerun()

    click = unprocessed_panel_click(side, dataset_name, event)
    if click is not None:
        rotated_h, rotated_w = rotated_labels.shape
        xi = int(np.clip(round(float(click["x"])), 0, rotated_w - 1))
        yi = int(np.clip(round(float(click["y"])), 0, rotated_h - 1))
        label_value = rotated_labels[yi, xi]
        if not np.isnan(label_value):
            toggle_panel_histology(
                side,
                dataset_name,
                int(label_value),
                float(xi),
                float(yi),
                pixel_counts,
            )
            request_experimental_rerun()

    with st.expander("Selection controls", expanded=True):
        st.markdown("##### Selected Regions")
        selected = panel_selected(side, dataset_name)
        selected_ids = sorted(panel_selected_ids(side, dataset_name))
        st.caption(
            f"Selected labels: {', '.join(map(str, selected_ids))}"
            if selected_ids
            else "No histology labels selected."
        )
        st.button(
            "Clear selected regions",
            key=f"experimental_{side}_{dataset_name}_clear_histology",
            use_container_width=True,
            disabled=panel_pair_count(side, dataset_name) == 0,
            on_click=clear_panel_selection,
            args=(side, dataset_name),
        )
        st.dataframe(
            display_dataframe(
                pd.DataFrame(
                    selected,
                    columns=["label", "histology_region", "pixel_count", "x", "y"],
                )
            ),
            width="stretch",
            height=220,
            hide_index=True,
        )
    with st.expander("Custom regions", expanded=False):
        render_panel_custom_region_controls(side, dataset_name)


def render_atlas_panel(
    side: str,
    dataset_name: str,
    annotation: Optional[np.ndarray],
    annotation_header: Optional[dict],
    annotation_source: str,
    metadata_lookup: dict[int, dict],
    metadata_error: Optional[str],
) -> None:
    if annotation is None:
        st.error(
            "annotation_10.nrrd was not found. Set SPALIGNDE_ALLEN_CCF_DIR "
            "or SPALIGNDE_ALLEN_ANNOTATION before starting the app."
        )
        return

    z_key = experimental_key(side, dataset_name, "z_slice")
    z_widget_key = experimental_key(side, dataset_name, "z_slice_widget_v2")
    flip_v_key = experimental_key(side, dataset_name, "flip_vertical")
    flip_h_key = experimental_key(side, dataset_name, "flip_horizontal")
    color_key = experimental_key(side, dataset_name, "color_mode")
    max_z = int(annotation.shape[0] - 1)
    fallback_z = min(DEFAULT_ATLAS_Z_SLICE, max_z)
    try:
        current_z = int(st.session_state.get(z_key, fallback_z))
    except (TypeError, ValueError):
        current_z = fallback_z
    if current_z < 0 or current_z > max_z:
        current_z = fallback_z
    st.session_state[z_key] = current_z
    if z_widget_key not in st.session_state:
        st.session_state[z_widget_key] = current_z

    with st.expander("Visualization controls", expanded=True):
        st.slider(
            "z slice",
            0,
            max_z,
            key=z_widget_key,
            on_change=sync_panel_atlas_z,
            args=(z_widget_key, z_key, max_z),
        )
        flip_col_1, flip_col_2 = st.columns(2)
        with flip_col_1:
            st.checkbox("Flip atlas vertically", key=flip_v_key)
        with flip_col_2:
            st.checkbox("Flip atlas horizontally", key=flip_h_key)
        st.button(
            "Reset atlas orientation",
            key=f"experimental_{side}_reset_atlas_orientation",
            on_click=reset_panel_atlas_orientation,
            args=(side, dataset_name),
        )
        st.radio(
            "Atlas coloring",
            ["Subsection colors", "Parent region colors"],
            horizontal=True,
            key=color_key,
        )
    mode = st.session_state[f"experimental_{side}_toolbar_mode"]
    mode_badge(side)

    try:
        atlas_info = atlas_slice_info(
            annotation,
            annotation_header,
            int(st.session_state[z_key]),
            bool(st.session_state[flip_v_key]),
            bool(st.session_state[flip_h_key]),
        )
    except Exception as error:
        st.error(f"Could not prepare the atlas slice: {error}")
        return
    slice_labels = atlas_info["sl"]
    if slice_labels.size == 0 or not np.any(slice_labels != 0):
        st.error(
            f"Atlas z-slice {st.session_state[z_key]} contains no nonzero region labels."
        )
        return
    crop_bounds = centered_crop_bounds(
        slice_labels.shape[0],
        slice_labels.shape[1],
        1.0,
    )
    crop_y0, crop_y1, crop_x0, crop_x1 = crop_bounds
    view_labels = slice_labels[crop_y0:crop_y1, crop_x0:crop_x1]
    selected_atlas_ids = tuple(sorted(panel_effective_selected_ids(side, dataset_name)))
    assigned_atlas_ids = tuple(sorted(visible_assigned_ids(side, dataset_name)))
    assigned_atlas_colors = visible_assigned_color_map(side, dataset_name)
    selected_atlas_colors = selected_custom_region_color_map(side, dataset_name)
    assigned_atlas_color_signature = tuple(
        (label, rgb_to_hex(color))
        for label, color in sorted(assigned_atlas_colors.items())
    )
    selected_atlas_color_signature = tuple(
        (label, rgb_to_hex(color))
        for label, color in sorted(selected_atlas_colors.items())
    )
    atlas_render_signature = (
        ATLAS_BACKGROUND_STYLE_VERSION,
        annotation_source,
        int(st.session_state[z_key]),
        bool(st.session_state[flip_v_key]),
        bool(st.session_state[flip_h_key]),
        st.session_state[color_key],
        selected_atlas_ids,
        assigned_atlas_ids,
        assigned_atlas_color_signature,
        selected_atlas_color_signature,
    )
    atlas_cache_signature_key = experimental_key(
        side,
        dataset_name,
        "image_cache_signature",
    )
    atlas_cache_image_key = experimental_key(side, dataset_name, "image_cache")
    if (
        st.session_state.get(atlas_cache_signature_key) != atlas_render_signature
        or st.session_state.get(atlas_cache_image_key) is None
    ):
        try:
            st.session_state[atlas_cache_image_key] = atlas_slice_image(
                view_labels,
                metadata_lookup,
                set(selected_atlas_ids),
                st.session_state[color_key],
                set(assigned_atlas_ids),
                assigned_atlas_colors,
                selected_atlas_colors,
            )
        except Exception as error:
            st.error(f"Could not render the atlas image: {error}")
            return
        st.session_state[atlas_cache_signature_key] = atlas_render_signature
    image = st.session_state[atlas_cache_image_key]
    if image is None or image.width == 0 or image.height == 0:
        st.error("The atlas image is empty. Adjust the z-slice or orientation.")
        return
    image_array = np.asarray(image)
    if image_array.size == 0 or not np.any(image_array):
        st.error("The atlas image contains no visible region data.")
        return
    try:
        figure = atlas_plotly_figure(image, view_labels, crop_bounds)
    except Exception as error:
        st.error(f"Could not build the atlas viewer: {error}")
        return
    figure.update_layout(uirevision=f"experimental-{side}-{dataset_name}")
    event = plotly_mode_events(
        figure,
        key=f"experimental_{side}_{dataset_name}_viewer",
        current_mode=mode,
    )
    if process_panel_mode_event(side, event):
        request_experimental_rerun()

    click = unprocessed_panel_click(side, dataset_name, event)
    if click is not None:
        xi = int(np.clip(round(float(click["x"])), crop_x0, crop_x1 - 1))
        yi = int(np.clip(round(float(click["y"])), crop_y0, crop_y1 - 1))
        label = int(slice_labels[yi, xi])
        toggle_panel_atlas(
            side,
            dataset_name,
            label,
            xi,
            yi,
            metadata_lookup,
        )
        if label != 0:
            request_experimental_rerun()

    with st.expander("Selection controls", expanded=True):
        st.markdown("##### Selected Regions")
        selected = panel_selected(side, dataset_name)
        st.caption(
            "Selected regions: "
            + ", ".join(str(item["atlas_region"]) for item in selected)
            if selected
            else "No atlas regions selected."
        )
        st.button(
            "Clear selected regions",
            key=f"experimental_{side}_clear_atlas",
            use_container_width=True,
            disabled=panel_pair_count(side, dataset_name) == 0,
            on_click=clear_panel_selection,
            args=(side, dataset_name),
        )
        st.dataframe(
            display_dataframe(
                pd.DataFrame(
                    selected,
                    columns=[
                        "label",
                        "atlas_region",
                        "name",
                        "parent_structure_id",
                        "color_hex_triplet",
                        "total_voxel_count",
                        "x",
                        "y",
                    ],
                )
            ),
            width="stretch",
            height=220,
            hide_index=True,
        )
    with st.expander("Custom regions", expanded=False):
        render_panel_custom_region_controls(side, dataset_name)
    if metadata_error:
        st.warning(metadata_error)


def render_dataset_panel(
    side: str,
    annotation: Optional[np.ndarray],
    annotation_header: Optional[dict],
    annotation_source: str,
    metadata_lookup: dict[int, dict],
    metadata_error: Optional[str],
) -> None:
    side_title = side.title()
    dataset_key = f"{side}_dataset"
    with st.expander("Dataset settings", expanded=True):
        st.selectbox(
            f"{side_title} dataset",
            list(experimental_datasets()),
            key=dataset_key,
            on_change=change_panel_dataset,
            args=(side,),
        )
    dataset_name = st.session_state[dataset_key]

    st.markdown(f"#### {dataset_name}")
    if dataset_definition(dataset_name)["kind"] == "st":
        render_st_panel(side, dataset_name)
    elif dataset_definition(dataset_name)["kind"] == "histology":
        render_histology_panel(side, dataset_name)
    else:
        render_atlas_panel(
            side,
            dataset_name,
            annotation,
            annotation_header,
            annotation_source,
            metadata_lookup,
            metadata_error,
        )


def custom_regions_dataframe() -> pd.DataFrame:
    rows = []
    active_dataset_names = set(experimental_datasets())
    for dataset_name, regions in st.session_state.experimental_custom_regions.items():
        if dataset_name not in active_dataset_names:
            continue
        definition = dataset_definition(dataset_name)
        for region in regions:
            rows.append(
                {
                    "custom_region_id": region["custom_region_id"],
                    "custom_region_name": region["custom_region_name"],
                    "display_color": region.get(
                        "display_color",
                        custom_region_color(region["custom_region_id"]),
                    ),
                    "source_dataset": dataset_name,
                    "source_dataset_id": definition.get("dataset_id", dataset_name),
                    "source_dataset_display_name": dataset_display_name(dataset_name, definition),
                    "source_dataset_type": definition["kind"],
                    "source_dataset_origin": definition.get("source", "built-in"),
                    "source_panel": region.get("source_panel", ""),
                    "source_region_type": region["source_region_type"],
                    "included_region_ids": json.dumps(region["included_region_ids"]),
                    "assigned_region_ids": json.dumps(
                        region.get("assigned_region_ids", region["included_region_ids"])
                    ),
                    "reuse_mode": region.get("reuse_mode", FLEXIBLE_REUSE_MODE),
                    "point_count": region.get("point_count", ""),
                    "pixel_count": region.get("pixel_count", ""),
                    "total_voxel_count": region.get("total_voxel_count", ""),
                    "created_from": region.get("created_from", "merged_existing_regions"),
                }
            )
    return pd.DataFrame(rows)


def render_custom_regions_table() -> None:
    st.subheader("Custom Regions")
    custom_df = custom_regions_dataframe()
    if custom_df.empty:
        st.info("Select existing colored regions, then create a custom region from the selection.")
        return

    display_df = custom_df[
        [
            "custom_region_name",
            "display_color",
            "source_dataset",
            "source_region_type",
            "included_region_ids",
            "reuse_mode",
            "point_count",
            "pixel_count",
            "total_voxel_count",
        ]
    ]
    st.dataframe(display_dataframe(display_df), width="stretch", hide_index=True)

    with st.expander("Rename or delete custom regions"):
        active_dataset_names = set(experimental_datasets())
        for dataset_name, regions in st.session_state.experimental_custom_regions.items():
            if dataset_name not in active_dataset_names:
                continue
            for region in list(regions):
                region_id = str(region["custom_region_id"])
                row_cols = st.columns([2, 2, 1])
                name_key = f"experimental_rename_{region_id}"
                if name_key not in st.session_state:
                    st.session_state[name_key] = region["custom_region_name"]
                with row_cols[0]:
                    st.text_input(
                        "Name",
                        key=name_key,
                        label_visibility="collapsed",
                    )
                with row_cols[1]:
                    st.caption(
                        f"{dataset_name}: "
                        f"{', '.join(map(str, region['included_region_ids']))}"
                    )
                with row_cols[2]:
                    st.button(
                        "Delete",
                        key=f"experimental_delete_{region_id}",
                        use_container_width=True,
                        on_click=delete_custom_region,
                        args=(dataset_name, region_id),
                    )
                rename_custom_region(dataset_name, region_id, name_key)

    st.download_button(
        "Export CSV",
        data=custom_df.to_csv(index=False),
        file_name="spalign_de_custom_regions.csv",
        mime="text/csv",
        key="experimental_export_custom_regions",
    )


def panel_export_settings(side: str, dataset_name: str) -> dict:
    definition = dataset_definition(dataset_name)
    common = {
        f"{side}_dataset_id": definition.get("dataset_id", dataset_name),
        f"{side}_dataset_display_name": dataset_display_name(dataset_name, definition),
        f"{side}_dataset_source": definition.get("source", "built-in"),
        f"{side}_dataset_kind": definition["kind"],
        f"{side}_dataset_type": definition["kind"],
        f"{side}_uploaded_file_name": definition.get("file_name", ""),
        f"{side}_original_filename": definition.get(
            "original_filename",
            definition.get("file_name", ""),
        ),
    }
    if definition["kind"] in ("st", "histology"):
        return {
            **common,
            f"{side}_rotation_angle": st.session_state[
                experimental_key(side, dataset_name, "rotation")
            ],
            f"{side}_atlas_z_slice": "",
            f"{side}_atlas_flip_vertical": "",
            f"{side}_atlas_flip_horizontal": "",
        }
    return {
        **common,
        f"{side}_rotation_angle": "",
        f"{side}_atlas_z_slice": st.session_state[
            experimental_key(side, dataset_name, "z_slice")
        ],
        f"{side}_atlas_flip_vertical": st.session_state[
            experimental_key(side, dataset_name, "flip_vertical")
        ],
        f"{side}_atlas_flip_horizontal": st.session_state[
            experimental_key(side, dataset_name, "flip_horizontal")
        ],
    }


def selection_export_fields(prefix: str, dataset_name: str, item: dict) -> dict:
    definition = dataset_definition(dataset_name)
    if item.get("_selection_type") == "custom_region":
        included_ids = [int(value) for value in item["included_region_ids"]]
        return {
            f"{prefix}_selection_type": "custom_region",
            f"{prefix}_selected_id": item["custom_region_id"],
            f"{prefix}_selected_name": item["custom_region_name"],
            f"{prefix}_custom_region_id": item["custom_region_id"],
            f"{prefix}_custom_region_name": item["custom_region_name"],
            f"{prefix}_custom_region_display_color": item.get(
                "display_color",
                custom_region_color(item["custom_region_id"]),
            ),
            f"{prefix}_source_dataset": item["source_dataset"],
            f"{prefix}_source_dataset_id": definition.get("dataset_id", dataset_name),
            f"{prefix}_source_dataset_display_name": dataset_display_name(dataset_name, definition),
            f"{prefix}_source_dataset_type": definition["kind"],
            f"{prefix}_source_dataset_origin": definition.get("source", "built-in"),
            f"{prefix}_source_region_type": item["source_region_type"],
            f"{prefix}_included_region_ids": json.dumps(included_ids),
            f"{prefix}_assigned_region_ids": json.dumps(
                item.get("assigned_region_ids", included_ids)
            ),
            f"{prefix}_reuse_mode": item.get("reuse_mode", FLEXIBLE_REUSE_MODE),
            f"{prefix}_point_count": item.get("point_count", ""),
            f"{prefix}_pixel_count": item.get("pixel_count", ""),
            f"{prefix}_total_voxel_count": item.get("total_voxel_count", ""),
            f"{prefix}_created_from": item.get("created_from", "merged_existing_regions"),
            f"{prefix}_x": "",
            f"{prefix}_y": "",
        }

    raw_common = {
        f"{prefix}_selection_type": "raw_region",
        f"{prefix}_custom_region_id": "",
        f"{prefix}_custom_region_name": "",
        f"{prefix}_custom_region_display_color": "",
        f"{prefix}_source_dataset": dataset_name,
        f"{prefix}_source_dataset_id": definition.get("dataset_id", dataset_name),
        f"{prefix}_source_dataset_display_name": dataset_display_name(dataset_name, definition),
        f"{prefix}_source_dataset_type": definition["kind"],
        f"{prefix}_source_dataset_origin": definition.get("source", "built-in"),
        f"{prefix}_source_region_type": source_region_type(dataset_name),
        f"{prefix}_included_region_ids": "",
        f"{prefix}_assigned_region_ids": "",
        f"{prefix}_reuse_mode": custom_region_reuse_mode(dataset_name),
        f"{prefix}_created_from": "",
    }
    kind = definition["kind"]
    if kind == "st":
        return {
            **raw_common,
            f"{prefix}_selected_id": item["cluster_id"],
            f"{prefix}_selected_name": f"cluster_{item['cluster_id']}",
            f"{prefix}_point_count": item["cluster_point_count"],
            f"{prefix}_pixel_count": "",
            f"{prefix}_total_voxel_count": "",
            f"{prefix}_x": item["x"],
            f"{prefix}_y": item["y"],
        }
    if kind == "histology":
        return {
            **raw_common,
            f"{prefix}_selected_id": item["label"],
            f"{prefix}_selected_name": item["histology_region"],
            f"{prefix}_point_count": "",
            f"{prefix}_pixel_count": item["pixel_count"],
            f"{prefix}_total_voxel_count": "",
            f"{prefix}_x": item["x"],
            f"{prefix}_y": item["y"],
        }
    return {
        **raw_common,
        f"{prefix}_selected_id": item["label"],
        f"{prefix}_selected_name": item["atlas_region"],
        f"{prefix}_full_name": item["name"],
        f"{prefix}_parent_structure_id": item["parent_structure_id"],
        f"{prefix}_structure_id_path": item["structure_id_path"],
        f"{prefix}_color_hex_triplet": item["color_hex_triplet"],
        f"{prefix}_point_count": "",
        f"{prefix}_pixel_count": "",
        f"{prefix}_total_voxel_count": item["total_voxel_count"],
        f"{prefix}_x": item["x"],
        f"{prefix}_y": item["y"],
    }


def panel_pair_items(side: str, dataset_name: str) -> list[dict]:
    raw_items = [
        {"_selection_type": "raw_region", **item}
        for item in panel_selected(side, dataset_name)
    ]
    custom_items = [
        {"_selection_type": "custom_region", **region}
        for region in panel_selected_custom_regions(side, dataset_name)
    ]
    return raw_items + custom_items


def panel_pair_count(side: str, dataset_name: str) -> int:
    return len(panel_selected(side, dataset_name)) + len(
        panel_selected_custom_region_ids(side, dataset_name)
    )


def combined_atlas_export_settings(
    left_dataset: str,
    right_dataset: str,
) -> dict:
    atlas_sides = [
        side
        for side, dataset_name in (
            ("left", left_dataset),
            ("right", right_dataset),
        )
        if dataset_definition(dataset_name)["kind"] == "atlas"
    ]
    if not atlas_sides:
        return {
            "atlas_z_slice": "",
            "atlas_flip_vertical": "",
            "atlas_flip_horizontal": "",
        }

    values = {}
    for field in ("z_slice", "flip_vertical", "flip_horizontal"):
        side_values = {
            side: st.session_state[
                experimental_key(side, "Allen CCF Atlas", field)
            ]
            for side in atlas_sides
        }
        values[f"atlas_{field}"] = (
            next(iter(side_values.values()))
            if len(side_values) == 1
            else json.dumps(side_values)
        )
    return values


def clear_all_experimental_selections() -> None:
    left_dataset = st.session_state.left_dataset
    right_dataset = st.session_state.right_dataset
    clear_panel_selection("left", left_dataset)
    clear_panel_selection("right", right_dataset)


def save_experimental_pair() -> None:
    left_dataset = st.session_state.left_dataset
    right_dataset = st.session_state.right_dataset
    left_items = panel_pair_items("left", left_dataset)
    right_items = panel_pair_items("right", right_dataset)
    group_id = f"group_{st.session_state.experimental_next_group_id}"
    left_ids = sorted(panel_effective_selected_ids("left", left_dataset))
    right_ids = sorted(panel_effective_selected_ids("right", right_dataset))
    left_custom_ids = panel_selected_custom_region_ids("left", left_dataset)
    right_custom_ids = panel_selected_custom_region_ids("right", right_dataset)
    shared = {
        "group_id": group_id,
        "left_dataset_type": left_dataset,
        "right_dataset_type": right_dataset,
        "left_selected_ids": json.dumps(left_ids),
        "right_selected_ids": json.dumps(right_ids),
        "left_raw_selected_ids": json.dumps(sorted(panel_selected_ids("left", left_dataset))),
        "right_raw_selected_ids": json.dumps(sorted(panel_selected_ids("right", right_dataset))),
        "left_selected_custom_region_ids": json.dumps(left_custom_ids),
        "right_selected_custom_region_ids": json.dumps(right_custom_ids),
        "left_paired_to_dataset": right_dataset,
        "left_paired_to_ids": json.dumps(right_ids),
        "right_paired_to_dataset": left_dataset,
        "right_paired_to_ids": json.dumps(left_ids),
        **panel_export_settings("left", left_dataset),
        **panel_export_settings("right", right_dataset),
        **combined_atlas_export_settings(left_dataset, right_dataset),
    }
    for left_item in left_items:
        for right_item in right_items:
            st.session_state.experimental_pairs.append(
                {
                    **shared,
                    **selection_export_fields("left", left_dataset, left_item),
                    **selection_export_fields("right", right_dataset, right_item),
                }
            )
    st.session_state.experimental_next_group_id += 1
    clear_all_experimental_selections()


def clear_experimental_pairs() -> None:
    st.session_state.experimental_pairs = []
    st.session_state.experimental_next_group_id = 1


st.set_page_config(page_title="spAlignDE UI Experimental", layout="wide")
initialize_experimental_state()

st.title("spAlignDE Interactive Region Pairing Tool")
st.caption(
    "Compare spatial transcriptomics, Allen CCF atlas, and histology datasets; "
    "define custom regions; and export paired region mappings."
)

with st.expander("How to use this app", expanded=False):
    st.markdown(
        """
1. Choose datasets for the left and right panels.
2. Use Pan mode to navigate and Select mode to choose regions.
3. Create custom regions by grouping selected regions.
4. Pair selected or custom regions across datasets.
5. Export saved pairings as CSV.
        """.strip()
    )

render_custom_dataset_upload()

left_dataset = st.session_state.left_dataset
right_dataset = st.session_state.right_dataset
needs_atlas = any(
    dataset_definition(dataset_name)["kind"] == "atlas"
    for dataset_name in (left_dataset, right_dataset)
)

if needs_atlas:
    metadata_df, metadata_path, metadata_error = load_region_metadata()
    region_lookup = metadata_by_id(metadata_df)
    annotation, annotation_header, annotation_source = load_annotation(None)
else:
    metadata_path = None
    metadata_error = None
    region_lookup = {}
    annotation = None
    annotation_header = None
    annotation_source = "Not loaded"

st.markdown("### Dataset Panels")
left_col, right_col = st.columns([1, 1])
with left_col:
    render_dataset_panel(
        "left",
        annotation,
        annotation_header,
        annotation_source,
        region_lookup,
        metadata_error,
    )
with right_col:
    render_dataset_panel(
        "right",
        annotation,
        annotation_header,
        annotation_source,
        region_lookup,
        metadata_error,
    )

if st.session_state.pop("experimental_deferred_rerun", False):
    st.rerun()

st.divider()
left_dataset = st.session_state.left_dataset
right_dataset = st.session_state.right_dataset
left_count = panel_pair_count("left", left_dataset)
right_count = panel_pair_count("right", right_dataset)

st.markdown("### Pairing")
action_col_1, action_col_2, _action_spacer = st.columns([1, 1, 2])
with action_col_1:
    st.button(
        "Clear all current selections",
        key="experimental_clear_all",
        use_container_width=True,
        disabled=left_count == 0 and right_count == 0,
        on_click=clear_all_experimental_selections,
    )
with action_col_2:
    st.button(
        "Save grouped pair",
        key="experimental_save_pair",
        type="primary",
        use_container_width=True,
        disabled=left_count == 0 or right_count == 0,
        on_click=save_experimental_pair,
    )

render_custom_regions_table()

st.subheader("Saved Pairs")
pairs_df = pd.DataFrame(st.session_state.experimental_pairs)
if pairs_df.empty:
    st.info("Select one or more items in both panels, then save a grouped pair.")
else:
    st.dataframe(display_dataframe(pairs_df), width="stretch", hide_index=True)
    st.markdown("#### Export")
    st.download_button(
        "Export CSV",
        data=pairs_df.to_csv(index=False),
        file_name="spalign_de_experimental_pairs.csv",
        mime="text/csv",
        key="experimental_export_pairs",
    )
    st.button(
        "Clear saved pairs",
        key="experimental_clear_pairs",
        on_click=clear_experimental_pairs,
    )
