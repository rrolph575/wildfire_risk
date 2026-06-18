# FWI percentile & exceedance-frequency maps

Scripts to build 6 CONUS maps from the Sup3rCC daily Fire Weather Index (FWI)
data.

## Files
- `fwi_percentile_maps.py` — reads the data, computes the maps, saves data + PNGs.
- `submit_fwi_percentile_maps.sh` — SLURM batch script to run it on an HPC node.
- `logs/` — SLURM job logs land here (`slurm-<jobid>.out`).

## Input data
Two Sup3rCC runs (same grid, 2.3M cells):
- **Historical** (2000–2014, 15 files) — used for the percentile thresholds:
  `/datasets/sup3rcc/conus_ecearth3veg_historical_r1i1p1f1/v0.2.2/daily/*fwi*.h5`
- **SSP245** (2015–2059, 45 files, ~151 GB) — used for the year counts:
  `/datasets/sup3rcc/conus_ecearth3veg_ssp245_r1i1p1f1/v0.2.2/daily/*fwi*.h5`

NREL rex-format H5. Each file has datasets:
- `fwi` — daily FWI, shape `(days, gridcell)` = `(365 or 366, 2,300,000)`.
- `fwi_tc` — **trend-corrected** FWI (corrects the humidity trend). NOT
  temperature-corrected.
- `meta` — per-cell table incl. `latitude`, `longitude`.
- `time_index` — daily timestamps (leap years have 366 days).

HDF5 chunking is `(full_time, 800 cells)`, so reading all days for a block of
columns is a chunk-aligned (fast) read.

## What the 6 maps are
**Percentile maps** (per grid cell, over daily FWI from the HISTORICAL run, all
years 2000–2014):
1. 90th percentile of daily FWI
2. 95th percentile of daily FWI
3. 98th percentile of daily FWI

**Exceedance-frequency maps** (per grid cell, counted over the SSP245 run
2015–2059, value = number of years, 0–45):
4. # years above the p90 historical threshold
5. # years above the p95 historical threshold
6. # years above the p98 historical threshold

A year "counts" for a cell if the number of days that year above the cell's
percentile threshold is GREATER than the historical baseline rate for that
percentile (10% of days for p90, 5% for p95, 2% for p98). Because the threshold
is the historical (2000–2014) climatology, SSP245 years increasingly exceed it
under warming — so the maps surface WHERE and HOW FAST extreme fire-weather days
become more frequent, without using the scenario period as its own baseline
(e.g. a California sample averages ~32/45 years above the p90 threshold, rising
from mixed in the 2010s to nearly all cells by the 2050s).

## How it runs efficiently
The full dataset can't fit in memory, so the script streams over spatial blocks
of cells (`CELLS_PER_BLOCK = 800 * 60 = 48,000`), holding every day of every
year for that block (~3 GB), computing per-cell percentile thresholds in one
pass, then tallying the per-year exceedance counts. All 45 file handles are kept
open; each byte is read from disk exactly once. Peak memory ~7 GB/block.

Runtime: ~20–45 min (≈6–13 min reading + percentile compute + tallies).

## How to run
```bash
cd /home/rrolph/wildfire
sbatch submit_fwi_percentile_maps.sh
```
The batch script activates the `sup3r` conda env (which has rex, h5py, xarray,
numpy, matplotlib). To run interactively instead:
```bash
conda activate sup3r
python fwi_percentile_maps.py
```

## Outputs
Written to:
```
/projects/rev/projects/ntps/fy26/underground_transmission/data/fwi/
```
Filenames are tagged `<TAG>` = `<var>_pcthist<start>_<end>` (e.g.
`fwi_tc_pcthist2000_2014`), so different variables/baseline periods don't
overwrite each other.
- `<TAG>_percentile_maps.npz` — the map data: `lat`, `lon`, `percentiles`,
  `pct_maps` (3 × n_cells), `year_count_maps` (3 × n_cells), `n_years`,
  `pct_baseline`.
- `<TAG>_percentile_maps.png` — combined 2×3 figure.
- 6 individual PNGs: `<TAG>_pct_p90/95/98.png`, `<TAG>_years_p90/95/98.png`.

## Config knobs (top of fwi_percentile_maps.py)
- `VARIABLE` — `"fwi"` or `"fwi_tc"`.
- `PCT_FILE_GLOB` — files whose pooled days define the percentile thresholds
  (default: the historical 2000–2014 run).
- `COUNT_FILE_GLOB` — files whose years are counted for exceedance (default: the
  ssp245 2015–2059 run).
- `PERCENTILES` — defaults `[90, 95, 98]`; baseline rate is `(100 - p)/100`.
- `CELLS_PER_BLOCK` — memory/speed tradeoff (keep a multiple of 800).
- `OUT_DIR` — output directory.

Re-running: if the `.npz` already exists, compute is skipped and only the plots
are regenerated. Delete the `.npz` to force a full recompute.

## Notes
- Maps use plain matplotlib scatter (no cartopy in the `sup3r` env), so there
  are no coastlines — points are placed by lat/lon. Install cartopy if you want
  basemap features.
