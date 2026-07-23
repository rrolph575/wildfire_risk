# Methods: fire-weather exceedance → transmission undergrounding

Pipeline from the Sup3rCC daily Fire Weather Index (FWI) to low/medium/high
undergrounding thresholds for transmission lines. Three scripts, run in order.

Data: NREL Sup3rCC daily `fwi_tc` (trend-corrected FWI) on a ~4 km CONUS grid
(2.3M cells). Historical run = 2000–2014; SSP245 future run = 2015–2059.

## 1. Percentile thresholds — `fwi_percentile_maps.py`
For each grid cell, compute the **90/95/98th percentile of daily FWI over the
historical period (2000–2014)**. This is the cell's climatological "extreme
fire-weather" level. Because the data (~151 GB) can't fit in memory, cells are
streamed in chunk-aligned spatial blocks and percentiles taken per block with
`np.nanpercentile`. Ocean / all-NaN cells give NaN.

## 2. Exceedance-rate change (A − B) — `fwi_exceedance_change_maps.py`
Using the historical percentile as a fixed per-cell threshold, for the 95th and
98th percentiles compute, per cell:

- **Threshold** = historical (2000–2014) p-th percentile of daily FWI.
- **B** = historical exceedance rate = (historical days above threshold) /
  15 years. By construction ≈ (100 − p)% of 365 (≈ 7.3 days/yr for p98,
  ≈ 18.3 days/yr for p95).
- **A** = future exceedance rate = (all days in 2025–2059 above threshold) /
  35 years.
- **A − B** = the increase in days per year above the historical extreme, i.e.
  how much more often extreme fire-weather days occur in the future vs. the
  historical baseline. Positive under warming.

Same block-streaming strategy. Maps use a diverging colour scale (red = larger
increase). Arrays (`A_maps`, `B_maps`, `metric_maps`, thresholds) are saved to
an `.npz`.

## 3. Undergrounding thresholds — `undergrounding_thresholds.py`
A line segment (or grid cell) is undergrounded where the **future exceedance
rate A** (days/yr above the historical p98) meets or exceeds a threshold. A
lower threshold catches more area → more undergrounding. A set of fixed
thresholds is evaluated (no low/medium/high labels):

- **8, 10, 15, 16, 18, 20 days/yr** — undergrounding ~51 / 27 / 3.6 / 2.5 / 1.4 /
  0.8% of CONUS cells respectively. For reference the historical baseline rate is
  ~7.3 days/yr, so thresholds above that select cells where future extreme
  fire-weather days are more frequent than they were historically.

Line coverage is computed by sampling A along each line at 500 m spacing
(reprojected to EPSG:5070 so equal spacing ⇒ equal length weight; 500 m is well
below the 4 km grid, so it resolves every cell the line crosses — finer spacing
would not change the percentages).

Outputs:
- `<TAG>_national.png` — CONUS maps of undergrounding (A ≥ threshold) vs. not,
  vertically stacked one per threshold (8, 10, 15, 16, 18, 20 days/yr;
  ascending), with state and Canada/Mexico country boundaries and the
  transmission lines overlaid; titles report the % of CONUS cells undergrounded.
- `<TAG>_scenarios.png` — grid (rows = threshold, cols = line).
- `<TAG>_exceedance.png` — the lines coloured by A.
- `<TAG>_summary.csv` — each threshold + total/per-line coverage.
- `<TAG>_points.gpkg` — densified sample points with a boolean `ug_<threshold>`
  column per threshold (`ug_8`, `ug_10`, `ug_15`, `ug_16`, `ug_18`, `ug_20`)
  flagging where A ≥ that threshold (GIS).

## Key knobs
- Percentile driving undergrounding: `PCTILE` (98 by default; 95 available).
- Metric for undergrounding: future rate **A** (not A − B).
- Thresholds: `THRESHOLDS = [8, 10, 15, 16, 18, 20]` days/yr.
- Baseline / future split: historical 2000–2014 for thresholds, 2025–2059 for A.
