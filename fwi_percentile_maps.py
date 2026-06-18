"""
FWI percentile and exceedance-frequency maps from Sup3rCC daily data.

Reads the historical FWI files (2000-2014) for the percentile thresholds and the
ssp245 FWI files (2015-2059) for the exceedance counts, both under
    /datasets/sup3rcc/conus_ecearth3veg_<run>_r1i1p1f1/v0.2.2/daily/*fwi*.h5
and produces 6 CONUS maps:

  Percentile maps (per grid cell, over daily FWI from the HISTORICAL run,
  all years 2000-2014):
    1. 90th percentile of daily FWI
    2. 95th percentile of daily FWI
    3. 98th percentile of daily FWI

  Exceedance-frequency maps (per grid cell, counted over the SSP245 run,
  2015-2059):
    4. # years that exceed the 90th-percentile threshold "more than baseline"
    5. # years that exceed the 95th-percentile threshold "more than baseline"
    6. # years that exceed the 98th-percentile threshold "more than baseline"

  where a year "counts" if the number of days that year above the cell's
  percentile threshold is GREATER than the historical baseline rate for that
  percentile (10% of days for p90, 5% for p95, 2% for p98). The threshold is the
  historical (2000-2014) climatology, so SSP245 years increasingly exceed it
  under warming -- these maps surface WHERE and HOW FAST extreme fire-weather
  days are becoming more frequent, without using the scenario period as its own
  baseline.

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
VARIABLE = "fwi_tc"               # "fwi" or "fwi_tc" (trend-corrected)
PERCENTILES = [90, 95, 98]     # also defines baseline rates: (100 - p) / 100

# Percentile THRESHOLDS are computed from the HISTORICAL run (all years), and the
# per-cell "# years above threshold" counts use the SSP245 future run. Using the
# historical climatology as the yardstick lets the exceedance maps reveal where
# and how fast extreme fire-weather days become more frequent under warming,
# without using the scenario period as its own baseline.
PCT_FILE_GLOB = ("/datasets/sup3rcc/conus_ecearth3veg_historical_r1i1p1f1/"
                 "v0.2.2/daily/*fwi*.h5")     # baseline for thresholds
COUNT_FILE_GLOB = ("/datasets/sup3rcc/conus_ecearth3veg_ssp245_r1i1p1f1/"
                   "v0.2.2/daily/*fwi*.h5")   # years counted for exceedance

# Spatial block size for streaming (multiple of the 800-cell HDF5 chunk width).
# 800 * 60 = 48,000 cells/block.
CELLS_PER_BLOCK = 800 * 60

OUT_DIR = ("/projects/rev/projects/ntps/fy26/underground_transmission/"
           "data/fwi")


def _years(glob_pat):
    return sorted(int(os.path.basename(f).split("_")[-1].split(".")[0])
                  for f in glob.glob(glob_pat))


# Tag outputs with the variable and the historical baseline span so methodology
# variants don't overwrite each other.
_pyrs = _years(PCT_FILE_GLOB)
PCT_LABEL = f"hist{_pyrs[0]}_{_pyrs[-1]}" if _pyrs else "histNA"
TAG = f"{VARIABLE}_pct{PCT_LABEL}"
NPZ_PATH = os.path.join(OUT_DIR, f"{TAG}_percentile_maps.npz")


# ----------------------------------------------------------------------------
# Compute
# ----------------------------------------------------------------------------
def compute_maps():
    pct_files = sorted(glob.glob(PCT_FILE_GLOB))
    count_files = sorted(glob.glob(COUNT_FILE_GLOB))
    if not pct_files:
        raise FileNotFoundError(f"No files match {PCT_FILE_GLOB}")
    if not count_files:
        raise FileNotFoundError(f"No files match {COUNT_FILE_GLOB}")

    pctiles = np.asarray(PERCENTILES, dtype=float)
    baseline_rates = (100.0 - pctiles) / 100.0   # p90->0.10, p95->0.05, p98->0.02
    n_p = len(pctiles)

    # Open all baseline (historical) and count (ssp245) file handles once.
    pct_h = [h5py.File(f, "r") for f in pct_files]
    count_h = [h5py.File(f, "r") for f in count_files]
    try:
        pct_dsets = [h[VARIABLE] for h in pct_h]
        count_dsets = [h[VARIABLE] for h in count_h]

        pct_ndays = [d.shape[0] for d in pct_dsets]
        count_ndays = [d.shape[0] for d in count_dsets]
        pct_total_days = int(np.sum(pct_ndays))

        n_cells = count_dsets[0].shape[1]
        if pct_dsets[0].shape[1] != n_cells:
            raise ValueError("Baseline and count grids differ "
                             f"({pct_dsets[0].shape[1]} vs {n_cells} cells)")

        pct_years = [int(os.path.basename(f).split("_")[-1].split(".")[0])
                     for f in pct_files]
        count_years = [int(os.path.basename(f).split("_")[-1].split(".")[0])
                       for f in count_files]
        n_years = len(count_files)
        print(f"{n_cells:,} grid cells, variable '{VARIABLE}'")
        print(f"Thresholds from {len(pct_files)} historical years "
              f"({pct_years[0]}-{pct_years[-1]}, {pct_total_days:,} days).")
        print(f"Year counts over {n_years} ssp245 years "
              f"({count_years[0]}-{count_years[-1]}).")

        # lat/lon for plotting (count-run grid; identical to baseline grid).
        meta = pd.DataFrame(count_h[0]["meta"][:])
        lat = meta["latitude"].to_numpy(dtype=np.float32)
        lon = meta["longitude"].to_numpy(dtype=np.float32)

        # Output maps.
        pct_maps = np.full((n_p, n_cells), np.nan, dtype=np.float32)
        year_count_maps = np.zeros((n_p, n_cells), dtype=np.int16)

        # Per-year exceedance-day threshold: rate * n_days_in_that_count_year.
        baseline_days = baseline_rates[:, None] * np.asarray(count_ndays)[None, :]

        block_starts = range(0, n_cells, CELLS_PER_BLOCK)
        n_blocks = len(block_starts)
        for bi, c0 in enumerate(block_starts):
            c1 = min(c0 + CELLS_PER_BLOCK, n_cells)
            nc = c1 - c0

            # 1) Historical block -> per-cell percentile thresholds.
            hist_block = np.empty((pct_total_days, nc), dtype=np.float32)
            off = 0
            for d, nd in zip(pct_dsets, pct_ndays):
                hist_block[off:off + nd] = d[:, c0:c1]
                off += nd
            with warnings.catch_warnings():
                # all-NaN columns (e.g. ocean) -> NaN threshold, expected.
                warnings.simplefilter("ignore", category=RuntimeWarning)
                thresholds = np.nanpercentile(hist_block, pctiles, axis=0)
            pct_maps[:, c0:c1] = thresholds.astype(np.float32)
            del hist_block

            # 2) For each ssp245 year, count days above threshold; a year
            #    "counts" if that exceeds the historical baseline rate.
            counts = np.zeros((n_p, nc), dtype=np.int16)
            for yi, d in enumerate(count_dsets):
                year_block = d[:, c0:c1]                      # (nd, nc)
                days_above = np.empty((n_p, nc), dtype=np.int32)
                for pi in range(n_p):
                    days_above[pi] = np.sum(year_block > thresholds[pi], axis=0)
                counts += (days_above > baseline_days[:, yi][:, None]).astype(np.int16)
            year_count_maps[:, c0:c1] = counts

            print(f"  block {bi + 1}/{n_blocks} "
                  f"(cells {c0:,}-{c1:,}) done", flush=True)

    finally:
        for h in pct_h + count_h:
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
        pct_baseline=np.asarray([pct_years[0], pct_years[-1]]),
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

    base = f"{_pyrs[0]}-{_pyrs[-1]} hist" if _pyrs else "hist"
    # Shared color scale for the three percentile maps for easy comparison.
    pmax = np.nanpercentile(pct_maps, 99.5)
    for pi, p in enumerate(pctiles):
        scatter(axes[0, pi], pct_maps[pi],
                f"{int(p)}th pctile of daily FWI ({base} baseline)",
                "FWI", "inferno", vmin=0, vmax=pmax)
        scatter(axes[1, pi], year_count_maps[pi],
                f"# ssp245 years above p{int(p)} ({base})",
                f"years (0-{n_years})", "viridis", vmin=0, vmax=n_years)

    out_png = os.path.join(OUT_DIR, f"{TAG}_percentile_maps.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved figure -> {out_png}")

    # Also save each of the 6 panels individually.
    for pi, p in enumerate(pctiles):
        for kind, values, title, cbar, cmap, vmax in [
            ("pct", pct_maps[pi],
             f"{int(p)}th pctile of daily FWI ({base} baseline)",
             "FWI", "inferno", pmax),
            ("years", year_count_maps[pi],
             f"# ssp245 years above p{int(p)} ({base})",
             f"years (0-{n_years})", "viridis", n_years),
        ]:
            f1, a1 = plt.subplots(figsize=(8, 6), constrained_layout=True)
            scatter_single(f1, a1, lon, lat, values, title, cbar, cmap, vmax)
            p1 = os.path.join(OUT_DIR, f"{TAG}_{kind}_p{int(p)}.png")
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
