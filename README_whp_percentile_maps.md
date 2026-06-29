# WHP percentile-exceedance maps

Builds CONUS maps of where the **Wildfire Hazard Potential (WHP)** index is at or
above its 90th / 95th / 98th percentile. Companion to the FWI percentile script,
but WHP is a single static raster (no time dimension), so a percentile is one
scalar threshold over all cells rather than a per-cell value over years.

## Files
- `whp_percentile_maps.py` — computes thresholds, writes data + PNGs.
- `submit_whp_percentile_maps.sh` — SLURM batch script (uses the `rev` env).
- `logs/` — SLURM job logs land here (`slurm-<jobid>.out`).

## Input data
USFS continuous Wildfire Hazard Potential (RDS-2020-0016), 90 m resample:
```
/projects/rev/projects/ntps/fy26/underground_transmission/data/rasters/wildfire/resampled/wildfire_hazard_potential_conus_90m.tif
```
- EPSG:5070, int32, `nodata = -2147483647`, ~1.76e9 cells.
- Value `0` = non-burnable (water/urban/agriculture/barren); values `> 0` are
  the continuous hazard score (max ~95,000).
- Native 30 m (`.../wildfire/wildfire_hazard_potential_conus.tif`, 15.9e9 cells,
  ~63 GB) and a 500 m resample also exist; switch via `WHP_PATH`.

## What the maps are
For each percentile p in `[90, 95, 98]`:
1. A single threshold = the p-th percentile of WHP over the chosen population
   (default: **burnable cells only**, WHP > 0). At 90 m these are
   p90 = 572, p95 = 1564, p98 = 5412.
2. A map showing where WHP >= that threshold — i.e. the top (100 - p)% most
   hazardous burnable cells (top 10% / 5% / 2%). Exceedance cells are colored by
   WHP value (log scale, inferno); burnable-but-below-threshold cells are light
   grey; non-burnable / nodata are white.

## How it runs efficiently
The 90 m raster (~7 GB as int32) is streamed in horizontal stripes
(`STRIPE_ROWS`):
- **Pass 1** accumulates an exact integer histogram of the burnable values and
  derives the percentile thresholds from its CDF (validated to match
  `np.percentile` exactly) — no need to hold all values in memory.
- **Pass 2** re-reads stripe by stripe and writes the three exceedance GeoTIFFs
  in one pass (windowed writes).
Peak memory is one stripe (well under 1 GB). The PNGs come from a single
decimated read (~2400 px wide) rendered with `imshow`.

Runtime: ~5–15 min at 90 m.

## How to run
```bash
cd /home/rrolph/wildfire
sbatch submit_whp_percentile_maps.sh
```
Or interactively:
```bash
conda activate rev      # the env with rasterio + matplotlib
python whp_percentile_maps.py
```

## Outputs
Written to:
```
/projects/rev/projects/ntps/fy26/underground_transmission/data/whp/
```
Filenames are tagged `<TAG>` = `whp_pct_<population>` (e.g. `whp_pct_burnable`):
- `<TAG>_thresholds.json` — the three threshold values + metadata (source,
  population, cell count).
- `<TAG>_p90/95/98_exceedance.tif` — full-resolution GeoTIFF per percentile;
  keeps the WHP value where it is >= the threshold, nodata elsewhere (preserves
  CRS/transform for GIS).
- `<TAG>_p90/95/98.png` — individual map per percentile.
- `<TAG>_percentile_maps.png` — combined 1×3 figure.

## Config knobs (top of whp_percentile_maps.py)
- `WHP_PATH` — input raster (default the 90 m resample).
- `PERCENTILES` — defaults `[90, 95, 98]`.
- `EXCLUDE_ZERO` — `True` (default) computes percentiles over burnable cells
  only; `False` includes the non-burnable 0 cells (lowers the thresholds).
- `STRIPE_ROWS` — streaming stripe height (memory/speed tradeoff).
- `PLOT_MAX_PX` — target width of the decimated PNG render.
- `OUT_DIR` — output directory.

## Notes
- Maps use `imshow` in the raster's native EPSG:5070 (equal-area) projection;
  axes ticks are hidden. No coastlines (no cartopy needed).
