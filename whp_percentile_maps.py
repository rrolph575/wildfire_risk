"""
Wildfire Hazard Potential (WHP) percentile-exceedance maps.

The WHP raster is a single static field (one value per cell, no time
dimension), so -- unlike the FWI script (fwi_percentile_maps.py), which computed
a percentile per cell *over years* -- here a percentile is a single scalar
threshold computed over all CONUS cells. "Map of the Nth percentile" therefore
means: a map of where WHP is at or above that threshold (the top (100-N)% most
hazardous cells).

Produces, for each percentile in PERCENTILES:
  * a GeoTIFF "data" file  -- full-resolution, WHP value kept where it is >= the
    threshold, nodata elsewhere (preserves CRS/transform for GIS use), and
  * a PNG map of that exceedance field over CONUS.
Plus a combined 1xN PNG and a small JSON/NPZ of the threshold values.

Input raster (USFS continuous WHP, RDS-2020-0016, 90 m resample):
    /projects/.../data/rasters/wildfire/resampled/wildfire_hazard_potential_conus_90m.tif
  EPSG:5070, int32, nodata = -2147483647. Value 0 = non-burnable
  (water/urban/agriculture/barren); values > 0 are continuous hazard.

Why this structure / efficiency notes
--------------------------------------
The 90 m raster is ~1.76e9 cells (~7 GB as int32), which is borderline for RAM
and grows with intermediate copies, so we stream it in horizontal stripes:
  * Pass 1 builds an exact integer histogram of the burnable (>0) values and
    derives the percentile thresholds from its CDF -- no need to hold all values.
  * Pass 2 re-reads stripe by stripe and writes the three exceedance GeoTIFFs
    in one go (windowed writes), so peak memory is one stripe (~0.4 GB).
The PNGs are rendered from a single decimated read (~2400 px wide) via imshow,
which is the right tool for a regular grid.

Run inside the `rev` conda env (which has rasterio + matplotlib):
    conda activate rev
    python whp_percentile_maps.py
"""

import json
import os
import warnings

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
WHP_PATH = ("/projects/rev/projects/ntps/fy26/underground_transmission/"
            "data/rasters/wildfire/resampled/"
            "wildfire_hazard_potential_conus_90m.tif")

PERCENTILES = [90, 95, 98]

# Compute percentiles over burnable cells only (WHP > 0), excluding the
# non-burnable 0 cells (water/urban/agriculture/barren). Set False to include
# every valid (non-nodata) cell in the percentile population.
EXCLUDE_ZERO = True

# Streaming stripe height (rows) for the full-res passes.
STRIPE_ROWS = 2048

# Target width (px) of the decimated array used for the PNG maps.
PLOT_MAX_PX = 2400

OUT_DIR = ("/projects/rev/projects/ntps/fy26/underground_transmission/"
           "data/whp")

POP = "burnable" if EXCLUDE_ZERO else "allvalid"
TAG = f"whp_pct_{POP}"   # e.g. whp_pct_burnable


# ----------------------------------------------------------------------------
# Percentile from an integer histogram (exact, memory-cheap)
# ----------------------------------------------------------------------------
def hist_percentiles(counts, qs):
    """Percentiles (numpy 'linear' convention) of the population described by
    `counts`, where counts[v] = number of cells with integer value v.
    Returns one float per q in `qs`."""
    counts = counts.astype(np.int64)
    n = int(counts.sum())
    if n == 0:
        return [float("nan")] * len(qs)
    cdf = np.cumsum(counts)                 # cdf[v] = # cells with value <= v
    vals = np.nonzero(counts)[0]            # the distinct values present
    cval = cdf[vals]                        # cumulative count up to each value
    out = []
    for q in qs:
        # numpy 'linear': position h = (n-1)*q/100 in the sorted population.
        h = (n - 1) * (q / 100.0)
        lo_i, hi_i = int(np.floor(h)), int(np.ceil(h))
        # ordered index k -> value = first distinct value whose cumulative > k.
        v_lo = vals[np.searchsorted(cval, lo_i, side="right")]
        v_hi = vals[np.searchsorted(cval, hi_i, side="right")]
        out.append(float(v_lo) + (h - lo_i) * float(v_hi - v_lo))
    return out


# ----------------------------------------------------------------------------
# Compute
# ----------------------------------------------------------------------------
def compute_thresholds(src):
    """Pass 1: stream the raster and build the burnable-value histogram, then
    derive the percentile thresholds from it."""
    nd = src.nodata
    h, w = src.height, src.width
    hist = np.zeros(0, dtype=np.int64)      # grows (zero-padded) as needed
    n_pop = 0
    for r0 in range(0, h, STRIPE_ROWS):
        nrows = min(STRIPE_ROWS, h - r0)
        a = src.read(1, window=Window(0, r0, w, nrows))
        m = a != nd
        if EXCLUDE_ZERO:
            m &= a > 0
        vals = a[m]
        if vals.size:
            bc = np.bincount(vals)                  # values are >= 0 here
            if bc.size > hist.size:
                grown = np.zeros(bc.size, dtype=np.int64)
                grown[:hist.size] = hist
                hist = grown
            hist[:bc.size] += bc.astype(np.int64)
            n_pop += int(vals.size)
        print(f"  hist rows {r0:,}-{r0 + nrows:,}", flush=True)
    thresholds = hist_percentiles(hist, PERCENTILES)
    return thresholds, n_pop


def write_exceedance_geotiffs(src, thresholds):
    """Pass 2: stream the raster and write one GeoTIFF per percentile, keeping
    the WHP value where value >= threshold and nodata elsewhere."""
    nd = src.nodata
    h, w = src.height, src.width
    profile = src.profile.copy()
    profile.update(driver="GTiff", dtype="int32", count=1, nodata=nd,
                   compress="deflate", predictor=2, tiled=True,
                   blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")

    paths = [os.path.join(OUT_DIR, f"{TAG}_p{int(p)}_exceedance.tif")
             for p in PERCENTILES]
    dsts = [rasterio.open(pth, "w", **profile) for pth in paths]
    try:
        for r0 in range(0, h, STRIPE_ROWS):
            nrows = min(STRIPE_ROWS, h - r0)
            win = Window(0, r0, w, nrows)
            a = src.read(1, window=win)
            valid = a != nd
            for thr, dst in zip(thresholds, dsts):
                out = np.full(a.shape, nd, dtype=np.int32)
                keep = valid & (a >= thr)
                out[keep] = a[keep]
                dst.write(out, 1, window=win)
            print(f"  geotiff rows {r0:,}-{r0 + nrows:,}", flush=True)
    finally:
        for dst in dsts:
            dst.close()
    return paths


def decimated_read(src):
    """Read the whole raster downsampled to ~PLOT_MAX_PX wide (nearest, to keep
    threshold logic faithful). Returns (array, extent) for imshow."""
    dec = max(1, int(np.ceil(src.width / PLOT_MAX_PX)))
    ow, oh = src.width // dec, src.height // dec
    a = src.read(1, out_shape=(oh, ow), resampling=Resampling.nearest)
    b = src.bounds
    extent = [b.left, b.right, b.bottom, b.top]
    return a, extent, dec


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
def plot_maps(disp, extent, thresholds, n_pop):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, ListedColormap

    grey_cmap = ListedColormap(["0.82"])   # light grey for context cells

    nd = WHP_NODATA
    valid = disp != nd
    burn = valid & (disp > 0)
    base = "burnable cells" if EXCLUDE_ZERO else "all valid cells"

    # Color scale shared across panels: log (WHP is heavy-tailed), from the
    # smallest plotted threshold up to a high percentile of burnable values.
    vmin = max(1, min(thresholds))
    vmax = float(np.nanpercentile(disp[burn], 99.9)) if burn.any() else vmin * 10
    norm = LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10))

    def panel(ax, thr, p):
        # Context: burnable-but-below-threshold cells in light grey.
        ctx = np.where(burn & (disp < thr), 1.0, np.nan)
        ax.imshow(ctx, extent=extent, origin="upper", cmap=grey_cmap,
                  interpolation="none")
        # Exceedance: WHP value where >= threshold.
        hi = np.where(valid & (disp >= thr), disp, np.nan).astype(float)
        im = ax.imshow(hi, extent=extent, origin="upper", cmap="inferno",
                       norm=norm, interpolation="none")
        ax.set_title(f"WHP >= p{int(p)} threshold ({thr:.0f})\n"
                     f"top {100 - p}% of {base}")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        return im

    n = len(thresholds)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6), constrained_layout=True)
    if n == 1:
        axes = [axes]
    im = None
    for ax, thr, p in zip(axes, thresholds, PERCENTILES):
        im = panel(ax, thr, p)
    cb = fig.colorbar(im, ax=axes, shrink=0.85)
    cb.set_label("Wildfire Hazard Potential")
    combo = os.path.join(OUT_DIR, f"{TAG}_percentile_maps.png")
    fig.savefig(combo, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {combo}")

    # Individual panels.
    for thr, p in zip(thresholds, PERCENTILES):
        f1, a1 = plt.subplots(figsize=(8, 6), constrained_layout=True)
        im = panel(a1, thr, p)
        cb = f1.colorbar(im, ax=a1, shrink=0.85)
        cb.set_label("Wildfire Hazard Potential")
        p1 = os.path.join(OUT_DIR, f"{TAG}_p{int(p)}.png")
        f1.savefig(p1, dpi=150, bbox_inches="tight")
        plt.close(f1)
    print(f"Saved {len(thresholds)} individual panels in {OUT_DIR}")


WHP_NODATA = None  # set in main() from the source


# ----------------------------------------------------------------------------
def main():
    global WHP_NODATA
    os.makedirs(OUT_DIR, exist_ok=True)
    with rasterio.open(WHP_PATH) as src:
        WHP_NODATA = src.nodata
        print(f"WHP raster {src.width:,} x {src.height:,} "
              f"({src.width * src.height:,} cells), nodata={src.nodata}")

        print("Pass 1: percentile thresholds from histogram...")
        thresholds, n_pop = compute_thresholds(src)
        for p, thr in zip(PERCENTILES, thresholds):
            print(f"  p{p} = {thr:.1f}")

        # Save the threshold scalars + metadata.
        meta = {
            "source": WHP_PATH,
            "population": "burnable (WHP>0)" if EXCLUDE_ZERO else "all valid",
            "n_population_cells": n_pop,
            "percentiles": PERCENTILES,
            "thresholds": {f"p{p}": thr
                           for p, thr in zip(PERCENTILES, thresholds)},
        }
        jpath = os.path.join(OUT_DIR, f"{TAG}_thresholds.json")
        with open(jpath, "w") as fh:
            json.dump(meta, fh, indent=2)
        print(f"Saved thresholds -> {jpath}")

        print("Pass 2: writing exceedance GeoTIFFs...")
        gpaths = write_exceedance_geotiffs(src, thresholds)
        for gp in gpaths:
            print(f"  -> {gp}")

        print("Rendering decimated maps...")
        disp, extent, dec = decimated_read(src)
        print(f"  decimation factor {dec} -> {disp.shape[1]}x{disp.shape[0]} px")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        plot_maps(disp, extent, thresholds, n_pop)
    print("Done.")


if __name__ == "__main__":
    main()
