"""
Plot a region of the year-count GeoTIFF and a ~50 km transect across it.

Produces a two-panel PNG:
  (left)  the regional map (cropped to a margin around the transect), with the
          transect line and its endpoints drawn on top;
  (right) the year-count profile sampled along the transect vs. distance (km).

The default transect runs west-to-east across California's Central Valley near
38.5 N, crossing from the low-count valley floor up into the higher-count Sierra
foothills. Override the endpoints on the command line to pick another region.

Usage:
    python plot_region.py [out_png] [lat0 lon0 lat1 lon1]

Examples:
    python plot_region.py                         # default Central Valley transect
    python plot_region.py cv.png 38.5 -122.1 38.5 -121.52
"""

import sys

import numpy as np
import rasterio
from rasterio.transform import rowcol
import matplotlib

matplotlib.use("Agg")  # headless / batch (no display)
import matplotlib.pyplot as plt

DEFAULT_SRC = (
    "/projects/rev/projects/ntps/fy26/underground_transmission/data/fwi/"
    "fwi_tc_pcthist2000_2014_cnt2025_2059_p98_year_count_map.tiff"
)
DEFAULT_OUT = "central_valley_transect.png"

# West-to-east across the Central Valley near 38.5 N (~50 km long).
DEFAULT_SEG = (38.5, -122.10, 38.5, -121.52)

MARGIN_DEG = 0.6  # padding around the transect for the map panel
N_SAMPLES = 200  # points sampled along the transect


def haversine_km(lat0, lon0, lat1, lon1):
    """Great-circle distance in km between two lon/lat points."""
    r = 6371.0
    p0, p1 = np.radians(lat0), np.radians(lat1)
    dphi = np.radians(lat1 - lat0)
    dlmb = np.radians(lon1 - lon0)
    a = np.sin(dphi / 2) ** 2 + np.cos(p0) * np.cos(p1) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def plot(src, out, seg):
    lat0, lon0, lat1, lon1 = seg
    with rasterio.open(src) as r:
        nodata = r.nodata
        # Window the raster to a margin around the transect.
        west = min(lon0, lon1) - MARGIN_DEG
        east = max(lon0, lon1) + MARGIN_DEG
        south = min(lat0, lat1) - MARGIN_DEG
        north = max(lat0, lat1) + MARGIN_DEG
        row_top, col_left = r.index(west, north)
        row_bot, col_right = r.index(east, south)
        row_top, row_bot = sorted((row_top, row_bot))
        col_left, col_right = sorted((col_left, col_right))
        win = rasterio.windows.Window(
            col_left, row_top, col_right - col_left + 1, row_bot - row_top + 1
        )
        region = r.read(1, window=win).astype(float)
        win_left, win_bottom, win_right, win_top = rasterio.windows.bounds(
            win, r.transform
        )

        # Sample the transect on the full raster (cell-center nearest values).
        lons = np.linspace(lon0, lon1, N_SAMPLES)
        lats = np.linspace(lat0, lat1, N_SAMPLES)
        full = r.read(1)
        rows, cols = rowcol(r.transform, lons, lats)
        profile = full[np.array(rows), np.array(cols)].astype(float)

    region[region == nodata] = np.nan
    profile[profile == nodata] = np.nan
    dist_km = haversine_km(lat0, lon0, lats, lons)

    extent = [win_left, win_right, win_bottom, win_top]
    fig, (ax_map, ax_prof) = plt.subplots(1, 2, figsize=(13, 5.5))

    im = ax_map.imshow(
        region, extent=extent, origin="upper", cmap="YlOrRd", aspect="auto"
    )
    ax_map.plot([lon0, lon1], [lat0, lat1], "-", color="black", lw=2)
    ax_map.plot([lon0, lon1], [lat0, lat1], color="cyan", lw=1, ls="--")
    ax_map.scatter([lon0], [lat0], c="blue", s=40, zorder=5, label="start (W)")
    ax_map.scatter([lon1], [lat1], c="black", s=40, zorder=5, label="end (E)")
    ax_map.set_xlabel("longitude")
    ax_map.set_ylabel("latitude")
    ax_map.set_title("p98 FWI year count (region)")
    ax_map.legend(loc="lower left", fontsize=8)
    fig.colorbar(im, ax=ax_map, label="years (of 35)", shrink=0.85)

    ax_prof.plot(dist_km, profile, "-", color="firebrick", lw=2)
    ax_prof.fill_between(dist_km, profile, alpha=0.2, color="firebrick")
    ax_prof.set_xlabel("distance along transect (km)")
    ax_prof.set_ylabel("years (of 35)")
    ax_prof.set_title(f"transect profile ({dist_km[-1]:.0f} km)")
    ax_prof.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(
        f"Wrote {out}  (transect {dist_km[-1]:.1f} km, "
        f"count {np.nanmin(profile):.0f}->{np.nanmax(profile):.0f})"
    )


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    if len(sys.argv) > 5:
        seg = tuple(float(x) for x in sys.argv[2:6])
    else:
        seg = DEFAULT_SEG
    plot(DEFAULT_SRC, out, seg)
