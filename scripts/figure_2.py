# Imports
import warnings

warnings.filterwarnings("ignore")  # Silence TqdmWarning

from pathlib import Path

import echoregions as er
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection, QuadMesh
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.dates import DateFormatter, HourLocator, MinuteLocator


# Helper functions
def build_rgb_array(
    da_Sv: xr.DataArray,
    channel_idx: list[int],
    vmin: float,
    vmax: float,
):
    # Create meshgrid
    time = da_Sv["ping_time"].values
    depth = da_Sv["depth"].values
    X, Y = np.meshgrid(time, depth)

    # Create Z values
    da_Sv = da_Sv.isel(channel=channel_idx).transpose("depth", "ping_time", "channel")
    Z = (da_Sv.values - vmin) / (vmax - vmin)
    Z = Z.clip(0, 1)
    Z = np.nan_to_num(Z, nan=0.0)

    # Mask invalid data (mostly bottom)
    valid_data_mask = ~np.all(np.isnan(da_Sv.values), axis=-1)
    Z = np.ma.array(Z, mask=np.dstack([~valid_data_mask] * 3))

    return X, Y, Z


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
    }
    plot_kwrgs = default_plot_kwrgs | plot_kwrgs

    # Create meshgrid
    X, Y, Z = build_rgb_array(
        da_Sv,
        channel_idx,
        plot_kwrgs["vmin"],
        plot_kwrgs["vmax"],
    )

    # Create figure (if ax was not passed)
    if ax is not None:
        plot_kwrgs.pop("figsize")
    else:
        _, ax = plt.subplots(figsize=plot_kwrgs.pop("figsize"))
    im = ax.pcolormesh(X, Y, Z, **plot_kwrgs)

    # Format
    ax.set_ylabel("Depth [m]")
    ax.set_xlabel("Time")
    ax.xaxis.set_major_locator(HourLocator())
    ax.xaxis.set_minor_locator(MinuteLocator(byminute=30))
    ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
    ax.yaxis.set_inverted(True)

    return im


# Set paths (rel. to project root)
SV_PATH = "./data/demo.zarr"  # Acoustic dataset
REGIONS_PATH = "./data/demo_regions.csv"  # Echogram regions shape

# Open / load data
ds_Sv = (
    xr.open_dataset(SV_PATH, engine="zarr")
    .set_coords("frequency_nominal")
    .where(lambda x: x.Sv != -150)
)
regions = er.read_regions_csv(REGIONS_PATH)
regions_closed = regions.close_region()

# Compute masks for
region_mask_ds, region_points = regions.region_mask(
    ds_Sv["Sv"].isel(channel=0).drop_vars("channel").compute(),
    collapse_to_2d=False,
)
mask_3d = region_mask_ds["mask_3d"]


# Figure
fig = plt.figure(figsize=(7.0, 8.0), layout="constrained")
n_rows = 4
gs_outer = fig.add_gridspec(
    nrows=n_rows,
    ncols=2,
    height_ratios=[3, 2, 2, 2],
)

# First row: RGB echogram spanning 2 columns
first = gs_outer[0, :].subgridspec(
    2,
    4,
    width_ratios=[1, 0.02, 0.02, 0.02],
    height_ratios=[0.1, 1],
    wspace=0.0,
)
ax = fig.add_subplot(first[:, 0])
cbar_axs = [fig.add_subplot(first[1, i]) for i in range(1, 4)]
cbar_title_ax = fig.add_subplot(first[0, 1:])

# RGB-composite echogram
da_Sv = ds_Sv.Sv.sel(
    ping_time=slice("2017-04-26 14:00", "2017-04-26 17:00"),
    depth=slice(None, 300),
)

im = plot_rgb(da_Sv, [0, 1, 2], ax)

# RGB colorbar
colors = ["red", "green", "blue"]
channel_labels = ["38", "70", "120"]
vmin, vmax = -90, -50

norm = Normalize(vmin, vmax)

for i, (color, label, cax) in enumerate(zip(colors, channel_labels, cbar_axs)):
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
for _, row in regions_closed.iterrows():
    if row["region_class"] == "Layer":
        continue
    ax.plot(
        row["time"],
        row["depth"],
        fillstyle="full",
        markersize=0.5,
        c="red",
        zorder=2,
    )

ax.set_ylim(300, 0)
times = da_Sv.ping_time.values
ax.set_xlim(times[0], times[-1])

# Remaining 2-column rows

# --- Relative frequency response --- #
left_ax = fig.add_subplot(gs_outer[1, 0])
right_ax = fig.add_subplot(gs_outer[1, 1])

unknown_regions = regions_closed[regions_closed["region_class"] == "Unknown"]
unknown_regions.sort_values("region_bbox_left", inplace=True)  # type: ignore
agg_dict = {
    f"Unknown {letter}": ds_Sv.Sv.where(mask_3d.sel(region_id=i), drop=True)
    for letter, i in zip(["a", "b", "c", "d"], unknown_regions["region_id"])
}

# Left: zoom on the unknown regions
da_Sv_zoomed = da_Sv.sel(
    ping_time=slice("2017-04-26 16:15", "2017-04-26 17:00"),
    depth=slice(None, 200),
)
X, Y, Z = build_rgb_array(da_Sv_zoomed, [0, 1, 2], -90, -50)
left_ax.pcolormesh(X, Y, Z, shading="nearest")
left_ax.set_ylabel("Depth [m]")
left_ax.set_xlabel("Time")
left_ax.xaxis.set_major_locator(MinuteLocator(byminute=[15, 30, 45, 0]))
left_ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
left_ax.yaxis.set_inverted(True)

# Add shapes
for _, row in unknown_regions.iterrows():
    left_ax.plot(
        row["time"],
        row["depth"],
        fillstyle="full",
        markersize=0.5,
        zorder=2,
    )


# Right: R(f)
for label, data in agg_dict.items():
    freqs = data["frequency_nominal"].values / 1000

    delta_Sv = data - data.isel(channel=0)
    mean = delta_Sv.mean(dim=["ping_time", "depth"])
    std = delta_Sv.std(dim=["ping_time", "depth"])

    right_ax.errorbar(freqs, mean, yerr=std, fmt="o-", capsize=5, label=label)

right_ax.set_xticks(freqs)
right_ax.set_xlabel("Channel [kHz]")
right_ax.set_ylabel(r"$\Delta$MVBS 38 kHz [dB]")
right_ax.set_ylim(-15, 15)

# --- Artifact removal --- #
left_ax = fig.add_subplot(gs_outer[2, 0])
right_ax = fig.add_subplot(gs_outer[2, 1])

row = regions_closed[regions_closed["region_class"] == "Artifact"].iloc[0]
ds_zoomed = ds_Sv.sel(
    ping_time=slice("2017-04-26 14:20", "2017-04-26 14:25"),
    depth=slice(75, 250),
)

# Plot RGB echogram with shape
X, Y, Z = build_rgb_array(ds_zoomed.Sv, [0, 1, 2], -90, -50)
left_ax.pcolormesh(X, Y, Z, shading="nearest")
left_ax.set_ylabel("Depth [m]")
left_ax.set_xlabel("Time")
left_ax.xaxis.set_major_locator(MinuteLocator(byminute=[20, 21, 22, 23, 24, 25]))
left_ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
left_ax.yaxis.set_inverted(True)

left_ax.plot(row["time"], row["depth"], fillstyle="full", markersize=0.5, c="red")

# Plot RGB echogram cleaned
ds_clean = ds_zoomed.where(mask_3d.sel(region_id=2) == 0)
X, Y, Z = build_rgb_array(ds_clean.Sv, [0, 1, 2], -90, -50)
right_ax.pcolormesh(X, Y, Z, shading="nearest")
right_ax.set_ylabel("Depth [m]")
right_ax.set_xlabel("Time")
right_ax.xaxis.set_major_locator(MinuteLocator(byminute=[20, 21, 22, 23, 24, 25]))
right_ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
right_ax.yaxis.set_inverted(True)

# --- School analysis --- #
left_ax = fig.add_subplot(gs_outer[3, 0])
right_ax = fig.add_subplot(gs_outer[3, 1])

row = regions_closed[regions_closed["region_class"] == "School"].iloc[0]
ds_zoomed = ds_Sv.sel(
    ping_time=slice("2017-04-26 16:30", "2017-04-26 16:43"),
    depth=slice(40, 100),
)
# Zoomed RGB echogram with shape
X, Y, Z = build_rgb_array(ds_zoomed.Sv, [0, 1, 2], -90, -50)
left_ax.pcolormesh(X, Y, Z, shading="nearest")
left_ax.set_ylabel("Depth [m]")
left_ax.set_xlabel("Time")
left_ax.xaxis.set_major_locator(MinuteLocator(byminute=[30, 35, 40]))
left_ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
left_ax.yaxis.set_inverted(True)

left_ax.plot(row["time"], row["depth"], fillstyle="full", markersize=0.5, c="red")

# Binary mask
mask_zoomed = mask_3d.sel(
    region_id=5,
    ping_time=slice("2017-04-26 16:30", "2017-04-26 16:43"),
    depth=slice(40, 100),
)
im = mask_zoomed.plot(ax=right_ax, yincrease=False)
right_ax.set_ylabel("Depth [m]")
right_ax.set_xlabel("Time")
right_ax.set_title("")
im.colorbar.set_label("")
right_ax.xaxis.set_major_locator(MinuteLocator(byminute=[30, 35, 40]))
right_ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))

plt.show()
fig.savefig("./images/basic-annotation-tasks.png", dpi=300)


# --- NEW FIG --- #

# Mask dataset to keep only migrant layer
ds_migrant = ds_Sv.where(mask_3d.sel(region_id=1), drop=True)
sA_time = ds_migrant.sA.isel(channel=0).sum(dim="depth")
# Compute mean depth weighted by NASC
weights = ds_migrant.sA.fillna(0)
depth_time = ds_migrant.depth.weighted(weights).mean(dim="depth")
# Compute delta Sv 38 kHz over time
delta_Sv = ds_migrant.Sv.drop_isel(channel=0) - ds_migrant.Sv.isel(channel=0)
delta_Sv_time = delta_Sv.mean(dim="depth")
# Smooth the depth time series
depth_time_smooth = depth_time.rolling(ping_time=50, center=True, min_periods=1).mean(
    dim="ping_time"
)
# Compute vertical speed
speed_time = depth_time_smooth.differentiate("ping_time", datetime_unit="s")

# Plot comparison between depth and R(f)
fig = plt.figure(figsize=(7, 2), layout="constrained")
gs = fig.add_gridspec(ncols=2, width_ratios=[1, 1], wspace=0.1)

left_ax = fig.add_subplot(gs[0])
right_ax = fig.add_subplot(gs[1])

# RGB plot
im = plot_rgb(ds_migrant.Sv, [0, 1, 2], vmin=-90, vmax=-50, ax=left_ax)
depth_time_smooth.isel(channel=0).plot(ax=left_ax, color="red", label="mean")
left_ax.xaxis.set_major_locator(HourLocator(byhour=[8, 10, 12, 14, 16]))
left_ax.xaxis.set_minor_locator(MinuteLocator(byminute=[0, 30]))
left_ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
left_ax.yaxis.set_inverted(True)
left_ax.legend(loc="upper right")
left_ax.set_title("")
left_ax.set_ylabel("Depth [m]")
right_ax.set_xlabel("Time")

ax1 = right_ax
ax2 = ax1.twinx()

# Line plot of speed
ax2.axhline(0, c="grey")  # horizontal line for reference
speed_time.isel(channel=0).plot(yincrease=False, c="black", ax=ax2)
ax1.set_title("")
ax1.set_ylabel("Speed [m s$^{-1}$]")

# Stripe plot of delta Sv for background
im = (
    delta_Sv_time.sel(channel=70e3)
    .expand_dims(dim={"y": [-0.2, 0.2]}, axis=0)  # fake 2D: 2 rows, same value
    .plot.pcolormesh(x="ping_time", y="y", yincrease=False, ax=ax2, add_colorbar=True)
)
ax2.set_yticks([])
ax2.set_ylabel("")
ax2.set_title("")
im.colorbar.set_label(r"$\Delta$MVBS$_{70-38 \text{kHz}}$ [dB]")

# Set ylim for speed plot
ax1.set_ylim(0.2, -0.2)
ax2.set_ylim(0.2, -0.2)

right_ax.xaxis.set_major_locator(HourLocator(byhour=[8, 10, 12, 14, 16]))
right_ax.xaxis.set_minor_locator(MinuteLocator(byminute=[0, 30]))
right_ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
right_ax.yaxis.set_inverted(True)
right_ax.legend(loc="upper right")
right_ax.set_title("")
right_ax.set_xlabel("Time")

plt.show()
fig.savefig("./images/fuzzy-layer-analysis.png", dpi=300)


# --- NEW FIG --- #
fig = plt.figure(figsize=(7, 2), layout="constrained")
gs = fig.add_gridspec(ncols=2, width_ratios=[1, 1], wspace=0.1)

left_ax = fig.add_subplot(gs[0])
right_ax = fig.add_subplot(gs[1])

time_vals = ds_migrant.ping_time.values
depth_vals = depth_time_smooth.isel(channel=0)
delta_Sv_vals = delta_Sv_time.sel(channel=70e3).values
speed_vals = speed_time.isel(channel=0).values

coll = left_ax.scatter(
    time_vals,
    depth_vals,
    c=delta_Sv_vals,
    vmin=-2.5,
    vmax=2.5,
    s=1,
    cmap="managua",
)


def get_extend_flag(coll: PathCollection) -> str | None:
    values = coll.get_array()
    if values is None:
        raise ValueError("Collection has no color values")
    vmin, vmax = coll.get_clim()
    extends = {(True, True): "both", (True, False): "min", (False, True): "max"}
    return extends.get((values.min() < vmin, vmax < values.max()))


extend = get_extend_flag(coll)
cbar = plt.colorbar(coll, ax=left_ax, extend=extend)
cbar.set_label(r"$\Delta$MVBS$_{70-38 \text{kHz}}$ [dB]")

left_ax.xaxis.set_major_locator(HourLocator(byhour=[8, 10, 12, 14, 16]))
left_ax.xaxis.set_minor_locator(MinuteLocator(byminute=[0, 30]))
left_ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
left_ax.yaxis.set_inverted(True)

left_ax.set_ylabel("Depth [m]")
left_ax.set_xlabel("Ping Time")

right_ax.scatter(
    speed_vals,
    delta_Sv_vals,
    s=1,
    color="black",
    alpha=0.5,
)
right_ax.axvline(x=speed_vals.mean(), color="grey", alpha=0.8, linewidth=1)
right_ax.axhline(y=delta_Sv_vals.mean(), color="grey", alpha=0.8, linewidth=1)

from scipy.stats import linregress
from sklearn.linear_model import LinearRegression

x, y = speed_vals, delta_Sv_vals
X = np.array(x).reshape((-1, 1))
model = LinearRegression()
model.fit(X, y)
xseq = np.linspace(x.min(), x.max(), num=100)
right_ax.plot(xseq, model.predict(xseq.reshape((-1, 1))), color="red")

eq = rf"$y = {model.intercept_:.2f} + {model.coef_[0]:.2f} x$"
right_ax.text(
    x.min() + 5e-3,
    2,
    eq,
    fontsize=7,
    horizontalalignment="left",
)

right_ax.set_ylabel(r"$\Delta$MVBS$_{70-38 \text{kHz}}$ [dB]")
right_ax.set_xlabel(r"Vertical speed [m.s$^{-1}$]")

plt.show()
fig.savefig("./images/fuzzy-layer-analysis_2.png", dpi=300)

# Save for statistical analysis in R
import pandas as pd

# Get simple arrays
data = {
    "time": ds_migrant.ping_time.values,
    "delta_Sv": delta_Sv_time.sel(channel=70e3).values,
    "depth": depth_time.isel(channel=0).values,
    "depth_smooth": depth_time_smooth.isel(channel=0).values,
    "speed_smooth": speed_time.isel(channel=0).values,
}

for name, a in data.items():
    print(f"{name} - type {type(a)} - shape {a.shape} - dtype {a.dtype}")

df = pd.DataFrame(data)

df.to_csv(Path(__file__).parent.parent / "data/layer.csv")
