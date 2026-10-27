# Imports
import warnings

warnings.filterwarnings("ignore")  # Silence TqdmWarning
import os
import string
from pathlib import Path

import cmocean
import echoregions as er
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.dates import ConciseDateFormatter
from matplotlib.figure import Figure

# Set paths (rel. to project root)
SV_PATH = "./data/demo.zarr"  # Acoustic dataset
REGIONS_PATH = "./data/demo_regions.csv"  # Echogram regions shape

# Open / load data
da_Sv = (
    xr.open_dataset(SV_PATH, engine="zarr")
    .set_coords("frequency_nominal")
    .where(lambda x: x.Sv != -150)
    .Sv.sel(
        ping_time=slice("2017-04-26 14:00", "2017-04-26 17:00"),
        depth=slice(None, 300),
    )
)
regions = er.read_regions_csv(REGIONS_PATH)
regions_closed = regions.close_region()

# Compute masks for
region_mask_ds, region_points = regions.region_mask(
    da_Sv.isel(channel=0).drop_vars("channel").compute(),  # type: ignore
    collapse_to_2d=False,
)
mask_3d = region_mask_ds["mask_3d"]


# Helper functions
def plot_rgb(
    da_Sv: xr.DataArray,
    channel_idx: list[int],
    ax: Axes | None = None,
    **plot_kwrgs,
) -> QuadMesh:

    default_plot_kwrgs = {
        "vmin": -90,
        "vmax": -50,
        "shading": "nearest",
        "figsize": (8, 4),
        "rasterize": True,
    }
    plot_kwrgs = default_plot_kwrgs | plot_kwrgs

    # Create meshgrid
    time = da_Sv["ping_time"].values
    depth = da_Sv["depth"].values
    X, Y = np.meshgrid(time, depth)

    # Create Z values
    da_Sv = da_Sv.isel(channel=channel_idx).transpose("depth", "ping_time", "channel")
    Z = (da_Sv.values - plot_kwrgs["vmin"]) / (plot_kwrgs["vmax"] - plot_kwrgs["vmin"])
    Z = Z.clip(0, 1)
    Z = np.nan_to_num(Z, nan=0.0)

    # Mask invalid data (mostly bottom)
    valid_data_mask = ~np.all(np.isnan(da_Sv.values), axis=-1)
    Z = np.ma.array(Z, mask=np.dstack([~valid_data_mask] * 3))

    # Create figure (if ax was not passed)
    if ax is not None:
        plot_kwrgs.pop("figsize")
    else:
        _, ax = plt.subplots(figsize=plot_kwrgs.pop("figsize"))
    im = ax.pcolormesh(X, Y, Z, **plot_kwrgs)

    # Format
    ax.set_ylabel("Depth [m]")
    ax.set_xlabel("Time")
    ax.xaxis.set_major_formatter(ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.yaxis.set_inverted(True)

    return im


# Figure

fig = plt.figure(figsize=(7.0, 3.0), layout="constrained")
gs = fig.add_gridspec(
    2,
    4,
    width_ratios=[1, 0.02, 0.02, 0.02],
    height_ratios=[0.1, 1],
    wspace=0.0,
)
ax = fig.add_subplot(gs[:, 0])
cbar_axes = [fig.add_subplot(gs[1, i]) for i in range(1, 4)]
cbar_title_ax = fig.add_subplot(gs[0, 1:])

# RGB-composite echogram
im = plot_rgb(da_Sv, [0, 1, 2], ax)

# RGB colorbar
colors = ["red", "green", "blue"]
channel_labels = ["38", "70", "120"]
vmin, vmax = -90, -50

norm = Normalize(vmin, vmax)

for i, (color, label, cax) in enumerate(zip(colors, channel_labels, cbar_axes)):
    cmap = LinearSegmentedColormap.from_list(f"black_{color}", ["black", color])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.set_title(label, fontsize=8)
    if i < 2:
        cbar.set_ticks([])
    else:
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label("MVBS [dB re 1 m-1]")

cbar_title_ax.set_axis_off()
cbar_title_ax.text(
    0.5,
    0.5,
    "Channel [kHz]",
    ha="center",
    va="center",
    fontsize=10,
)

# Annotated regions
cdict = {"School": "C6", "Unknown": "C1", "Layer": "C2", "Artifact": "C3"}
used_labels = set()

for _, row in regions_closed.iterrows():
    label = row["region_class"] if row["region_class"] not in used_labels else None
    ax.plot(
        row["time"],
        row["depth"],
        fillstyle="full",
        markersize=0.5,
        c=cdict[row["region_class"]],  # type: ignore
        label=label,
        zorder=2,
    )
    used_labels.add(row["region_class"])


ax.set_ylim(300, 0)
times = da_Sv.ping_time.values
ax.set_xlim(times[0], times[-1])
ax.legend(loc="upper left", title="Region class", fontsize=8, title_fontsize=8)

plt.show()
