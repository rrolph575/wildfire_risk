"""
FWI percentile and exceedance-frequency maps from Sup3rCC daily data.

Reads the 45 yearly FWI files (2015-2059) in
    /datasets/sup3rcc/conus_ecearth3veg_ssp245_r1i1p1f1/v0.2.2/daily/*fwi*.h5
and produces 6 CONUS maps:

  Percentile maps (per grid cell, over all days of all years):
    1. 90th percentile of daily FWI
    2. 95th percentile of daily FWI
    3. 98th percentile of daily FWI

  Exceedance-frequency maps (per grid cell):
    4. # years that exceed the 90th-percentile threshold "more than baseline"
    5. # years that exceed the 95th-percentile threshold "more than baseline"
    6. # years that exceed the 98th-percentile threshold "more than baseline"

  where a year "counts" if the number of days that year above the cell's
  percentile threshold is GREATER than the baseline rate for that percentile
  (10% of days for p90, 5% for p95, 2% for p98). By construction the pooled
  record exceeds the threshold at exactly that rate on average, so this map
  highlights which cells have an above-baseline number of fire-prone years
  (and, under SSP245 warming, where the distribution is shifting).

Why this structure / efficiency notes
--------------------------------------
The full dataset is ~151 GB (2.3M grid cells x ~16,400 days x float32), so it
cannot be loaded into memory at once. The HDF5 datasets are chunked as
(full_time, 800 cells), meaning "all days for a block of columns" is a
chunk-aligned read. We therefore:
  * keep all 45 file handles open,
  * stream over spatial blocks of cells (a multiple of 800),
  * for each block hold every day of every year in memory (a few GB),
    compute per-cell percentile thresholds in one pass, then tally the
    per-year exceedance counts.
This keeps memory bounded (~3 GB/block) while reading each byte from disk
exactly once.

Run inside the `sup3r` conda env:
    conda activate sup3r
    python fwi_percentile_maps.py
"""

import glob
import os
import warnings

import h5py
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DATA_DIR = ("/datasets/sup3rcc/conus_ecearth3veg_ssp245_r1i1p1f1/"
            "v0.2.2/daily")
FILE_GLOB = os.path.join(DATA_DIR, "*fwi*.h5")
VARIABLE = "fwi_tc"               # "fwi" or "fwi_tc" (trend-corrected)
PERCENTILES = [90, 95, 98]     # also defines baseline rates: (100 - p) / 100

# Spatial block size for streaming (multiple of the 800-cell HDF5 chunk width).
# 800 * 60 = 48,000 cells/block -> ~3 GB per block for the full time series.
CELLS_PER_BLOCK = 800 * 60

OUT_DIR = ("/projects/rev/projects/ntps/fy26/underground_transmission/"
           "data/fwi")
NPZ_PATH = os.path.join(OUT_DIR, f"{VARIABLE}_percentile_maps.npz")


# ----------------------------------------------------------------------------
# Compute
# ----------------------------------------------------------------------------
def compute_maps():
    files = sorted(glob.glob(FILE_GLOB))
    if not files:
        raise FileNotFoundError(f"No files match {FILE_GLOB}")
    print(f"Found {len(files)} files: "
          f"{os.path.basename(files[0])} ... {os.path.basename(files[-1])}")

    pctiles = np.asarray(PERCENTILES, dtype=float)
    baseline_rates = (100.0 - pctiles) / 100.0   # p90->0.10, p95->0.05, p98->0.02
    n_p = len(pctiles)

    # Open every yearly file once and keep the handle.
    handles = [h5py.File(f, "r") for f in files]
    try:
        dsets = [h[VARIABLE] for h in handles]
        n_days_per_year = [d.shape[0] for d in dsets]   # 365 or 366
        n_cells = dsets[0].shape[1]
        total_days = int(np.sum(n_days_per_year))
        n_years = len(files)
        print(f"{n_cells:,} grid cells, {n_years} years, "
              f"{total_days:,} total days, variable '{VARIABLE}'")

        # lat/lon for plotting (read once from the first file's meta table).
        meta = pd.DataFrame(handles[0]["meta"][:])
        lat = meta["latitude"].to_numpy(dtype=np.float32)
        lon = meta["longitude"].to_numpy(dtype=np.float32)

        # Output maps.
        pct_maps = np.full((n_p, n_cells), np.nan, dtype=np.float32)
        year_count_maps = np.zeros((n_p, n_cells), dtype=np.int16)

        # Per-year baseline thresholds (days): rate * n_days_in_that_year.
        # shape (n_p, n_years)
        baseline_days = baseline_rates[:, None] * np.asarray(n_days_per_year)[None, :]

        block_starts = range(0, n_cells, CELLS_PER_BLOCK)
        n_blocks = len(block_starts)
        for bi, c0 in enumerate(block_starts):
            c1 = min(c0 + CELLS_PER_BLOCK, n_cells)
            nc = c1 - c0

            # Gather every day of every year for this block: (total_days, nc).
            block = np.empty((total_days, nc), dtype=np.float32)
            year_slices = []
            off = 0
            for d, nd in zip(dsets, n_days_per_year):
                block[off:off + nd] = d[:, c0:c1]
                year_slices.append((off, off + nd))
                off += nd

            with warnings.catch_warnings():
                # all-NaN columns (e.g. ocean) -> NaN threshold, expected.
                warnings.simplefilter("ignore", category=RuntimeWarning)
                thresholds = np.nanpercentile(block, pctiles, axis=0)  # (n_p, nc)
            pct_maps[:, c0:c1] = thresholds.astype(np.float32)

            # For each year, count days above each threshold; a year "counts"
            # if that count exceeds the year's baseline expectation.
            counts = np.zeros((n_p, nc), dtype=np.int16)
            for yi, (s, e) in enumerate(year_slices):
                year_block = block[s:e]                      # (nd, nc)
                # days_above[p, cell] for this year
                days_above = np.empty((n_p, nc), dtype=np.int32)
                for pi in range(n_p):
                    days_above[pi] = np.sum(year_block > thresholds[pi], axis=0)
                counts += (days_above > baseline_days[:, yi][:, None]).astype(np.int16)
            year_count_maps[:, c0:c1] = counts

            del block
            print(f"  block {bi + 1}/{n_blocks} "
                  f"(cells {c0:,}-{c1:,}) done", flush=True)

    finally:
        for h in handles:
            h.close()

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(
        NPZ_PATH,
        lat=lat, lon=lon,
        percentiles=pctiles,
        pct_maps=pct_maps,
        year_count_maps=year_count_maps,
        n_years=n_years,
        variable=VARIABLE,
    )
    print(f"Saved arrays -> {NPZ_PATH}")
    return lat, lon, pctiles, pct_maps, year_count_maps, n_years


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
def plot_maps(lat, lon, pctiles, pct_maps, year_count_maps, n_years):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_p = len(pctiles)
    fig, axes = plt.subplots(2, n_p, figsize=(6 * n_p, 9), constrained_layout=True)

    # Common scatter style. 2.3M points -> tiny markers, rasterized output.
    def scatter(ax, values, title, cbar_label, cmap, vmin=None, vmax=None):
        sc = ax.scatter(lon, lat, c=values, s=0.3, marker="s",
                        cmap=cmap, vmin=vmin, vmax=vmax,
                        linewidths=0, rasterized=True)
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal")
        cb = fig.colorbar(sc, ax=ax, shrink=0.85)
        cb.set_label(cbar_label)

    # Shared color scale for the three percentile maps for easy comparison.
    pmax = np.nanpercentile(pct_maps, 99.5)
    for pi, p in enumerate(pctiles):
        scatter(axes[0, pi], pct_maps[pi],
                f"{int(p)}th percentile of daily FWI",
                "FWI", "inferno", vmin=0, vmax=pmax)
        scatter(axes[1, pi], year_count_maps[pi],
                f"# years above baseline (p{int(p)})",
                f"years (0-{n_years})", "viridis", vmin=0, vmax=n_years)

    out_png = os.path.join(OUT_DIR, f"{VARIABLE}_percentile_maps.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved figure -> {out_png}")

    # Also save each of the 6 panels individually.
    for pi, p in enumerate(pctiles):
        for kind, values, title, cbar, cmap, vmax in [
            ("pct", pct_maps[pi], f"{int(p)}th percentile of daily FWI",
             "FWI", "inferno", pmax),
            ("years", year_count_maps[pi], f"# years above baseline (p{int(p)})",
             f"years (0-{n_years})", "viridis", n_years),
        ]:
            f1, a1 = plt.subplots(figsize=(8, 6), constrained_layout=True)
            scatter_single(f1, a1, lon, lat, values, title, cbar, cmap, vmax)
            p1 = os.path.join(OUT_DIR, f"{VARIABLE}_{kind}_p{int(p)}.png")
            f1.savefig(p1, dpi=150, bbox_inches="tight")
            plt.close(f1)
    print(f"Saved 6 individual panels in {OUT_DIR}")


def scatter_single(fig, ax, lon, lat, values, title, cbar_label, cmap, vmax):
    sc = ax.scatter(lon, lat, c=values, s=0.3, marker="s", cmap=cmap,
                    vmin=0, vmax=vmax, linewidths=0, rasterized=True)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    cb = fig.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label(cbar_label)


# ----------------------------------------------------------------------------
def main():
    if os.path.exists(NPZ_PATH):
        print(f"Loading cached arrays from {NPZ_PATH} "
              f"(delete it to recompute).")
        d = np.load(NPZ_PATH, allow_pickle=True)
        lat, lon, pctiles = d["lat"], d["lon"], d["percentiles"]
        pct_maps, year_count_maps = d["pct_maps"], d["year_count_maps"]
        n_years = int(d["n_years"])
    else:
        lat, lon, pctiles, pct_maps, year_count_maps, n_years = compute_maps()
    plot_maps(lat, lon, pctiles, pct_maps, year_count_maps, n_years)


if __name__ == "__main__":
    main()
