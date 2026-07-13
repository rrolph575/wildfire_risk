"""
FWI exceedance-rate change maps (historical minus future) from Sup3rCC data.

Companion to fwi_percentile_maps.py. Same inputs and streaming strategy, but a
different per-cell metric. For each percentile p in PERCENTILES (95 and 98):

  threshold  = the p-th percentile of daily FWI over the HISTORICAL run
               (2000-2014), per grid cell -- the "historical (pre-2015)"
               climatological extreme.

  A = future exceedance rate (days/year)
      = (total # days across 2025-2059 with FWI > threshold) / n_future_years
        where n_future_years = 35 (the 2025-2059 ssp245 files).

  B = historical exceedance rate (days/year)
      = (total # historical days with FWI > threshold) / n_hist_years.
        By construction this is ~ (100 - p)% of 365, i.e. ~7.3 days/yr for p98
        and ~18.25 days/yr for p95 (computed empirically per cell here).

  Plot metric = A - B   (days/year) = the increase in the number of days per
  year above the historical p-th percentile, future minus historical.

Under warming the future period sees MORE extreme-FWI days, so A > B and A - B
is positive; the larger a cell's value, the bigger its increase in extreme
fire-weather days per year. Cells where A - B < 0 saw fewer extreme days than
the historical baseline.

Efficiency: identical to fwi_percentile_maps.py -- the ~151 GB cannot fit in
memory, so we stream chunk-aligned spatial blocks of cells, holding every day of
every year for a block, computing per-cell thresholds and the A/B tallies in one
pass, reading each byte from disk exactly once.

Run inside the `sup3r` conda env:
    conda activate sup3r
    python fwi_exceedance_change_maps.py
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
VARIABLE = "fwi_tc"              # "fwi" or "fwi_tc" (trend-corrected)
PERCENTILES = [95, 98]          # start with the 95th and 98th

# Thresholds + the historical rate B come from the HISTORICAL run (2000-2014);
# the future rate A is counted over the ssp245 run from COUNT_YEAR_MIN onward.
PCT_FILE_GLOB = ("/datasets/sup3rcc/conus_ecearth3veg_historical_r1i1p1f1/"
                 "v0.2.2/daily/*fwi*.h5")     # baseline (2000-2014)
COUNT_FILE_GLOB = ("/datasets/sup3rcc/conus_ecearth3veg_ssp245_r1i1p1f1/"
                   "v0.2.2/daily/*fwi*.h5")   # future (ssp245)
COUNT_YEAR_MIN = 2025           # future window start; 2025-2059 -> 35 years

# Spatial block size for streaming (multiple of the 800-cell HDF5 chunk width).
CELLS_PER_BLOCK = 800 * 60      # 48,000 cells/block

OUT_DIR = ("/projects/rev/projects/ntps/fy26/underground_transmission/"
           "data/fwi")


def _years(glob_pat):
    return sorted(int(os.path.basename(f).split("_")[-1].split(".")[0])
                  for f in glob.glob(glob_pat))


# Tag outputs with the variable, historical baseline span, and future window so
# this doesn't collide with the other scripts' outputs.
_pyrs = _years(PCT_FILE_GLOB)
_cyrs = [y for y in _years(COUNT_FILE_GLOB) if y >= COUNT_YEAR_MIN]
PCT_LABEL = f"hist{_pyrs[0]}_{_pyrs[-1]}" if _pyrs else "histNA"
CNT_LABEL = f"cnt{_cyrs[0]}_{_cyrs[-1]}" if _cyrs else "cntNA"
TAG = f"{VARIABLE}_AminusB_pct{PCT_LABEL}_{CNT_LABEL}"
NPZ_PATH = os.path.join(OUT_DIR, f"{TAG}_maps.npz")


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

    def _file_year(f):
        return int(os.path.basename(f).split("_")[-1].split(".")[0])
    count_files = [f for f in count_files if _file_year(f) >= COUNT_YEAR_MIN]
    if not count_files:
        raise FileNotFoundError(
            f"No count files at/after {COUNT_YEAR_MIN} in {COUNT_FILE_GLOB}")

    pctiles = np.asarray(PERCENTILES, dtype=float)
    n_p = len(pctiles)

    pct_h = [h5py.File(f, "r") for f in pct_files]
    count_h = [h5py.File(f, "r") for f in count_files]
    try:
        pct_dsets = [h[VARIABLE] for h in pct_h]
        count_dsets = [h[VARIABLE] for h in count_h]

        pct_ndays = [d.shape[0] for d in pct_dsets]
        pct_total_days = int(np.sum(pct_ndays))

        n_cells = count_dsets[0].shape[1]
        if pct_dsets[0].shape[1] != n_cells:
            raise ValueError("Baseline and count grids differ "
                             f"({pct_dsets[0].shape[1]} vs {n_cells} cells)")

        pct_years = [_file_year(f) for f in pct_files]
        count_years = [_file_year(f) for f in count_files]
        n_hist_years = len(pct_files)
        n_future_years = len(count_files)      # 2025-2059 -> 35
        print(f"{n_cells:,} grid cells, variable '{VARIABLE}'")
        print(f"Thresholds + B from {n_hist_years} historical years "
              f"({pct_years[0]}-{pct_years[-1]}, {pct_total_days:,} days).")
        print(f"A over {n_future_years} future ssp245 years "
              f"({count_years[0]}-{count_years[-1]}).")

        meta = pd.DataFrame(count_h[0]["meta"][:])
        lat = meta["latitude"].to_numpy(dtype=np.float32)
        lon = meta["longitude"].to_numpy(dtype=np.float32)

        # Output maps (NaN over ocean / all-NaN columns).
        pct_maps = np.full((n_p, n_cells), np.nan, dtype=np.float32)
        A_maps = np.full((n_p, n_cells), np.nan, dtype=np.float32)   # days/yr
        B_maps = np.full((n_p, n_cells), np.nan, dtype=np.float32)   # days/yr

        block_starts = range(0, n_cells, CELLS_PER_BLOCK)
        n_blocks = len(block_starts)
        for bi, c0 in enumerate(block_starts):
            c1 = min(c0 + CELLS_PER_BLOCK, n_cells)
            nc = c1 - c0

            # 1) Historical block -> per-cell thresholds + historical rate B.
            hist_block = np.empty((pct_total_days, nc), dtype=np.float32)
            off = 0
            for d, nd in zip(pct_dsets, pct_ndays):
                hist_block[off:off + nd] = d[:, c0:c1]
                off += nd
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                thresholds = np.nanpercentile(hist_block, pctiles, axis=0)
            pct_maps[:, c0:c1] = thresholds.astype(np.float32)

            nan_col = np.isnan(thresholds)     # (n_p, nc) ocean mask per pctile

            # B: historical days above threshold, per year. NaN > x is False,
            # so NaN days never count.
            for pi in range(n_p):
                hist_above = np.sum(hist_block > thresholds[pi], axis=0)
                B = hist_above.astype(np.float32) / n_hist_years
                B[nan_col[pi]] = np.nan
                B_maps[pi, c0:c1] = B
            del hist_block

            # 2) A: total future days above threshold across all years / n_yrs.
            future_above = np.zeros((n_p, nc), dtype=np.int64)
            for d in count_dsets:
                year_block = d[:, c0:c1]                       # (nd, nc)
                for pi in range(n_p):
                    future_above[pi] += np.sum(year_block > thresholds[pi],
                                               axis=0)
            for pi in range(n_p):
                A = future_above[pi].astype(np.float32) / n_future_years
                A[nan_col[pi]] = np.nan
                A_maps[pi, c0:c1] = A

            print(f"  block {bi + 1}/{n_blocks} "
                  f"(cells {c0:,}-{c1:,}) done", flush=True)

    finally:
        for h in pct_h + count_h:
            h.close()

    metric_maps = A_maps - B_maps      # days/year: increase future vs historical

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(
        NPZ_PATH,
        lat=lat, lon=lon,
        percentiles=pctiles,
        pct_maps=pct_maps,
        A_maps=A_maps,
        B_maps=B_maps,
        metric_maps=metric_maps,
        n_hist_years=n_hist_years,
        n_future_years=n_future_years,
        variable=VARIABLE,
        pct_baseline=np.asarray([pct_years[0], pct_years[-1]]),
        future_window=np.asarray([count_years[0], count_years[-1]]),
    )
    print(f"Saved arrays -> {NPZ_PATH}")
    return lat, lon, pctiles, metric_maps


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
def plot_maps(lat, lon, pctiles, metric_maps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_p = len(pctiles)
    base = f"{_pyrs[0]}-{_pyrs[-1]} hist" if _pyrs else "hist"
    fut = f"{_cyrs[0]}-{_cyrs[-1]}" if _cyrs else "future"

    # Diverging, symmetric-about-zero scale (ColorBrewer RdBu is colorblind
    # safe). A - B > 0 (more future extreme days) -> red; < 0 -> blue.
    vlim = float(np.nanpercentile(np.abs(metric_maps), 99))
    if not np.isfinite(vlim) or vlim == 0:
        vlim = 1.0

    def scatter(fig, ax, values, title, cbar_label):
        sc = ax.scatter(lon, lat, c=values, s=0.3, marker="s",
                        cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                        linewidths=0, rasterized=True)
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal")
        cb = fig.colorbar(sc, ax=ax, shrink=0.85)
        cb.set_label(cbar_label)

    fig, axes = plt.subplots(1, n_p, figsize=(7 * n_p, 6),
                             constrained_layout=True)
    if n_p == 1:
        axes = [axes]
    for pi, p in enumerate(pctiles):
        scatter(fig, axes[pi], metric_maps[pi],
                f"Increase in days/yr above historical p{int(p)}\n"
                f"({base} baseline, {fut} future)",
                f"increase in days/yr above historical p{int(p)} (A - B)")
    combo = os.path.join(OUT_DIR, f"{TAG}_maps.png")
    fig.savefig(combo, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {combo}")

    for pi, p in enumerate(pctiles):
        f1, a1 = plt.subplots(figsize=(8, 6), constrained_layout=True)
        scatter(f1, a1, metric_maps[pi],
                f"Increase in days/yr above historical p{int(p)}\n"
                f"({base} baseline, {fut} future)",
                f"increase in days/yr above historical p{int(p)} (A - B)")
        p1 = os.path.join(OUT_DIR, f"{TAG}_p{int(p)}.png")
        f1.savefig(p1, dpi=150, bbox_inches="tight")
        plt.close(f1)
    print(f"Saved {n_p} individual panels in {OUT_DIR}")


# ----------------------------------------------------------------------------
def main():
    if os.path.exists(NPZ_PATH):
        print(f"Loading cached arrays from {NPZ_PATH} (delete to recompute).")
        d = np.load(NPZ_PATH, allow_pickle=True)
        lat, lon, pctiles = d["lat"], d["lon"], d["percentiles"]
        metric_maps = d["metric_maps"]
    else:
        lat, lon, pctiles, metric_maps = compute_maps()
    plot_maps(lat, lon, pctiles, metric_maps)


if __name__ == "__main__":
    main()
