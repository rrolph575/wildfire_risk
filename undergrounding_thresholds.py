"""
Fire-driven undergrounding thresholds for transmission lines.

Idea: a transmission-line segment is undergrounded where the FUTURE fire-weather
exceedance rate (A = days/year above the historical p-th percentile FWI, from
fwi_exceedance_change_maps.py) crosses a threshold. Undergrounding is expensive,
so we want three thresholds -- "low", "medium", "high" -- that underground
progressively LESS line:

    low threshold  -> more of the line exceeds it   -> MOST undergrounding
    high threshold -> little of the line exceeds it -> LEAST undergrounding

We pick each threshold by targeting a total undergrounded coverage (fraction of
line length), so the amount of undergrounding is controlled. A single threshold
value is applied to all lines (a uniform policy); the resulting coverage is the
share of total line length above it.

Inputs
------
* Exceedance field: A_maps in
    .../data/fwi/fwi_tc_AminusB_pcthist2000_2014_cnt2025_2059_maps.npz
  (per-cell future exceedance rate, days/yr, for p95 and p98; lat/lon points).
* Lines: AC_Lines_Group1_3_lines_from_Jianyu_orig_from_NTP.gpkg (3 AC lines).

Outputs (to OUT_DIR)
--------------------
* <TAG>_national.png     -- CONUS map per threshold (A >= threshold vs not).
* <TAG>_scenarios.png    -- grid: rows = thresholds, cols = the 3 lines,
                            undergrounded segments in red.
* <TAG>_exceedance.png   -- the 3 lines colored by A (context).
* <TAG>_summary.csv      -- threshold value + coverage per line.
* <TAG>_points.gpkg      -- densified sample points with A and a ug_<threshold>
                            flag column per threshold (GIS).

Run inside the `reeds2` conda env (geopandas + scipy + matplotlib):
    conda activate reeds2
    python undergrounding_thresholds.py
"""

import os

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import MultiLineString, Point
from scipy.spatial import cKDTree

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
NPZ_PATH = ("/projects/rev/projects/ntps/fy26/underground_transmission/"
            "data/fwi/fwi_tc_AminusB_pcthist2000_2014_cnt2025_2059_maps.npz")
LINES_PATH = ("/home/rrolph/wildfire/"
              "AC_Lines_Group1_3_lines_from_Jianyu_orig_from_NTP.gpkg")

# Which future exceedance rate to threshold on: 95 or 98 (the p-th percentile).
PCTILE = 98

# Fixed exceedance-rate thresholds (days/yr above the historical p-th
# percentile). A line/cell is undergrounded where A >= threshold. Lower
# threshold -> more undergrounding. Each gets a ug_<threshold> flag column and a
# national map panel; the historical baseline for reference is ~7.3 days/yr.
THRESHOLDS = [8.0, 10.0, 15.0, 16.0, 18.0, 20.0]

# CONUS state boundaries for the national maps.
STATES_PATH = "/projects/rev/projects/scapes/maps/conus_state_boundaries.gpkg"

# Sample spacing along the lines, in metres (equal spacing => coverage measured
# by length). Lines are reprojected to EPSG:5070 (Albers, metres) for this.
# Only needs to be well below the ~4 km exceedance grid; finer just smooths the
# plotted segment boundaries without changing the undergrounded percentages.
SAMPLE_M = 500.0

OUT_DIR = ("/projects/rev/projects/ntps/fy26/underground_transmission/"
           "data/undergrounding")

TAG = f"undergrounding_A_p{PCTILE}"
EQUAL_AREA = 5070   # CONUS Albers for length-correct densification


# ----------------------------------------------------------------------------
def densify_metres(geom, step):
    """Return equal-spacing sample points (in the geom's CRS units) along a
    (Multi)LineString, one point roughly every `step` units."""
    parts = geom.geoms if isinstance(geom, MultiLineString) else [geom]
    pts = []
    for ln in parts:
        L = ln.length
        n = max(2, int(np.ceil(L / step)) + 1)
        for dd in np.linspace(0.0, L, n):
            p = ln.interpolate(dd)
            pts.append((p.x, p.y))
    return pts


def build_samples(lines_5070):
    """Densify every line; return a GeoDataFrame of sample points (EPSG:5070)
    with a line id, tagged later with the exceedance value."""
    recs = []
    for i, geom in enumerate(lines_5070.geometry):
        for (x, y) in densify_metres(geom, SAMPLE_M):
            recs.append({"line": i, "geometry": Point(x, y)})
    gdf = gpd.GeoDataFrame(recs, crs=f"EPSG:{EQUAL_AREA}")
    return gdf


# ----------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    d = np.load(NPZ_PATH, allow_pickle=True)
    lat, lon = d["lat"], d["lon"]
    pex = list(d["percentiles"].astype(int))
    pidx = pex.index(PCTILE)
    A = d["A_maps"][pidx]            # future exceedance rate, days/yr
    print(f"Loaded exceedance field: A_maps p{PCTILE} "
          f"({np.isfinite(A).sum():,} valid cells)")

    tree = cKDTree(np.column_stack([lon, lat]))

    lines = gpd.read_file(LINES_PATH).to_crs(4326)
    names = [str(n) for n in lines["NAME"]]
    lines_5070 = lines.to_crs(EQUAL_AREA)

    samples = build_samples(lines_5070)
    # Query the exceedance field at each sample point (in lon/lat).
    samp_ll = samples.to_crs(4326)
    xy = np.column_stack([samp_ll.geometry.x, samp_ll.geometry.y])
    _, idx = tree.query(xy)
    samples["A"] = A[idx]
    samp_ll["A"] = A[idx]

    valid = np.isfinite(samples["A"].to_numpy())

    # Fixed thresholds; a segment is undergrounded where A >= threshold.
    flag_thr = sorted(float(t) for t in THRESHOLDS)

    def col(t):
        return f"ug_{int(round(t))}"

    # Flag undergrounding at each threshold, and tabulate coverage.
    rows = []
    for t in flag_thr:
        flag = samples["A"].to_numpy() >= t
        samples[col(t)] = flag
        samp_ll[col(t)] = flag
        tot_cov = flag[valid].mean()
        rows.append({"threshold_days_per_yr": t, "line": "ALL",
                     "coverage_total": round(float(tot_cov), 3)})
        for i, nm in enumerate(names):
            m = (samples["line"].to_numpy() == i) & valid
            cov = flag[m].mean() if m.any() else np.nan
            rows.append({"threshold_days_per_yr": t, "line": nm,
                         "coverage_total": round(float(cov), 3)})
    summary = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, f"{TAG}_summary.csv")
    summary.to_csv(csv_path, index=False)
    print(f"\nThresholds (days/yr above historical p{PCTILE}) and coverage:")
    print(summary[summary.line == "ALL"].to_string(index=False))
    print(f"Saved summary -> {csv_path}")

    # Save the sample points (lon/lat) with flags for GIS.
    gpkg_path = os.path.join(OUT_DIR, f"{TAG}_points.gpkg")
    samp_ll.to_file(gpkg_path, driver="GPKG")
    print(f"Saved points  -> {gpkg_path}")

    _plot(samples, samp_ll, lines_5070, names, flag_thr)
    _plot_national(lat, lon, A, lines, flag_thr)


# ----------------------------------------------------------------------------
def _plot(samples, samp_ll, lines_5070, names, flag_thr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_line = len(names)
    sx = samples.geometry.x.to_numpy()
    sy = samples.geometry.y.to_numpy()
    line_id = samples["line"].to_numpy()
    Aval = samples["A"].to_numpy()

    def line_pad(i):
        b = lines_5070.geometry.iloc[i].bounds
        padx = max((b[2] - b[0]) * 0.08, 3000)
        pady = max((b[3] - b[1]) * 0.08, 3000)
        return (b[0] - padx, b[2] + padx, b[1] - pady, b[3] + pady)

    # ---- Figure 1: threshold x line grid ----
    n_thr = len(flag_thr)
    fig, axes = plt.subplots(n_thr, n_line,
                             figsize=(5 * n_line, 4.2 * n_thr),
                             constrained_layout=True, squeeze=False)
    for r, thr in enumerate(flag_thr):
        flag = samples[f"ug_{int(round(thr))}"].to_numpy()
        for c in range(n_line):
            ax = axes[r, c]
            m = line_id == c
            ax.plot(sx[m], sy[m], "-", color="0.75", lw=3, zorder=1)
            ug = m & flag
            ax.scatter(sx[ug], sy[ug], c="crimson", s=14, zorder=3,
                       label="undergrounded")
            cov = flag[m].mean()
            xmin, xmax, ymin, ymax = line_pad(c)
            ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(names[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(f"A >= {thr:.0f} days/yr", fontsize=10)
            ax.text(0.5, 0.02, f"{cov*100:.0f}% undergrounded",
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=9, color="crimson")
    fig.suptitle(f"Undergrounded line segments by threshold "
                 f"(A, days/yr > historical p{PCTILE})", fontsize=13)
    p1 = os.path.join(OUT_DIR, f"{TAG}_scenarios.png")
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure  -> {p1}")

    # ---- Figure 2: lines colored by exceedance value ----
    vmin = float(np.nanpercentile(Aval, 2))
    vmax = float(np.nanpercentile(Aval, 98))
    fig, axes = plt.subplots(1, n_line, figsize=(5 * n_line, 4.4),
                             constrained_layout=True)
    if n_line == 1:
        axes = [axes]
    sc = None
    for c in range(n_line):
        ax = axes[c]
        m = line_id == c
        sc = ax.scatter(sx[m], sy[m], c=Aval[m], s=16, cmap="inferno",
                        vmin=vmin, vmax=vmax)
        xmin, xmax, ymin, ymax = line_pad(c)
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(names[c], fontsize=10)
    cb = fig.colorbar(sc, ax=axes, shrink=0.85)
    cb.set_label(f"A: future days/yr above historical p{PCTILE}")
    p2 = os.path.join(OUT_DIR, f"{TAG}_exceedance.png")
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure  -> {p2}")


def _line_coords(geom):
    """Yield (xs, ys) arrays for each part of a (Multi)LineString in lon/lat."""
    parts = geom.geoms if isinstance(geom, MultiLineString) else [geom]
    for ln in parts:
        xs, ys = ln.xy
        yield np.asarray(xs), np.asarray(ys)


def _plot_national(lat, lon, A, lines_ll, thr_list):
    """CONUS maps of undergrounding (A >= threshold) vs not, one per threshold
    (ascending), with state/country boundaries and the lines overlaid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    states = gpd.read_file(STATES_PATH).to_crs(4326)
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    countries = world[world["name"].isin(["Canada", "Mexico"])].to_crs(4326)

    finite = np.isfinite(A)
    lon_f, lat_f, A_f = lon[finite], lat[finite], A[finite]

    # Crop to CONUS so the Canada/Mexico borders show as edges, not full extent.
    xlim = (-125.5, -66.5)
    ylim = (23.5, 51.0)

    thr_list = sorted(thr_list)
    n = len(thr_list)
    fig, axes = plt.subplots(n, 1, figsize=(8.5, 4.6 * n),
                             constrained_layout=True)
    if n == 1:
        axes = [axes]
    for c, thr in enumerate(thr_list):
        ax = axes[c]
        ug = A_f >= thr
        # Not-undergrounded (light grey) then undergrounded (crimson) on top.
        ax.scatter(lon_f[~ug], lat_f[~ug], c="0.85", s=0.3, marker="s",
                   linewidths=0, rasterized=True)
        ax.scatter(lon_f[ug], lat_f[ug], c="crimson", s=0.3, marker="s",
                   linewidths=0, rasterized=True)
        # Country (Canada/Mexico) then state boundaries, then the lines on top.
        countries.boundary.plot(ax=ax, color="0.2", linewidth=0.8, zorder=4)
        states.boundary.plot(ax=ax, color="0.35", linewidth=0.4, zorder=4)
        for geom in lines_ll.geometry:
            for xs, ys in _line_coords(geom):
                ax.plot(xs, ys, color="black", lw=2.5, zorder=5)
        conus_pct = ug.mean() * 100
        ax.set_title(f"A >= {thr:.0f} days/yr  |  "
                     f"{conus_pct:.1f}% of CONUS cells undergrounded",
                     fontsize=11)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"National undergrounding extent from future fire-weather "
                 f"exceedance (A > historical p{PCTILE})", fontsize=13)
    p3 = os.path.join(OUT_DIR, f"{TAG}_national.png")
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure  -> {p3}")


if __name__ == "__main__":
    main()
