"""
Convert the 98th-percentile year-count map (.npz) to a GeoTIFF.

The source .npz stores the map as flattened, per-cell arrays: `lat`, `lon`, and
`year_count_map`, each of length n_cells. The cells form a regular lat/lon grid
in row-major order with latitude as the outer (slowest-varying) dimension, so
the data reshapes to (n_lat, n_lon) with row 0 at the northernmost latitude.

We rebuild that grid, derive a north-up affine transform from the cell-center
lat/lon, and write a single-band GeoTIFF in EPSG:4326 (WGS84 lon/lat).

Usage:
    python npz_to_tiff.py [src_npz] [out_tiff]
"""

import sys

import numpy as np
import rasterio
from rasterio.transform import from_origin

DEFAULT_SRC = (
    "/projects/rev/projects/ntps/fy26/underground_transmission/data/fwi/"
    "fwi_tc_pcthist2000_2014_cnt2025_2059_p98_year_count_map.npz"
)
DEFAULT_OUT = (
    "/projects/rev/projects/ntps/fy26/underground_transmission/data/fwi/"
    "fwi_tc_pcthist2000_2014_cnt2025_2059_p98_year_count_map.tiff"
)

# year_count_map is int16; reserve a value that can't occur as a real count
# (counts are >= 0) so masked/off-grid cells read as nodata in GIS tools.
NODATA = -1


def convert(src, out):
    with np.load(src, allow_pickle=False) as data:
        lat = data["lat"].astype(np.float64)
        lon = data["lon"].astype(np.float64)
        values = data["year_count_map"]

    ulat = np.unique(lat)  # ascending
    ulon = np.unique(lon)  # ascending
    n_lat, n_lon = ulat.size, ulon.size
    if n_lat * n_lon != values.size:
        raise ValueError(
            f"Grid {n_lat}x{n_lon} != {values.size} cells; data is not a full "
            "regular grid"
        )

    # Mean cell sizes (the grid spacing varies only at the float32 round-off
    # level, so the mean is an accurate, robust pixel size).
    dlon = float(np.diff(ulon).mean())
    dlat = float(np.diff(ulat).mean())

    # Place each cell value at its (row=lat, col=lon) index. Row 0 must be the
    # northernmost latitude for a north-up raster, so flip the lat axis.
    lat_idx = np.searchsorted(ulat, lat)
    lon_idx = np.searchsorted(ulon, lon)
    grid = np.full((n_lat, n_lon), NODATA, dtype=values.dtype)
    grid[lat_idx, lon_idx] = values
    grid = grid[::-1, :]  # north-up: row 0 = max latitude

    # lat/lon are cell centers; the transform origin is the top-left *corner*.
    west = ulon.min() - dlon / 2.0
    north = ulat.max() + dlat / 2.0
    transform = from_origin(west, north, dlon, dlat)

    with rasterio.open(
        out,
        "w",
        driver="GTiff",
        height=n_lat,
        width=n_lon,
        count=1,
        dtype=grid.dtype,
        crs="EPSG:4326",
        transform=transform,
        nodata=NODATA,
        compress="deflate",
    ) as dst:
        dst.write(grid, 1)

    print(f"Wrote {out} ({n_lat}x{n_lon}, EPSG:4326, nodata={NODATA})")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SRC
    out = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT
    convert(src, out)
