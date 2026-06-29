"""
Extract the 98th-percentile year-count map from the FWI percentile maps file
and save it to its own .npz.

The source file stores year_count_maps with shape (3, n_cells), one row per
percentile in `percentiles` ([90, 95, 98]). The 98th percentile is the last
row (index 2). We pull that row out and save it -- with lat/lon so the map can
still be plotted -- to a new file.

Usage:
    python extract_p98_year_count.py [src_npz] [out_npz]
"""

import sys

import numpy as np

DEFAULT_SRC = (
    "/projects/rev/projects/ntps/fy26/underground_transmission/data/fwi/"
    "fwi_tc_pcthist2000_2014_cnt2025_2059_percentile_maps.npz"
)
DEFAULT_OUT = (
    "/projects/rev/projects/ntps/fy26/underground_transmission/data/fwi/"
    "fwi_tc_pcthist2000_2014_cnt2025_2059_p98_year_count_map.npz"
)


def extract(src, out):
    with np.load(src, allow_pickle=False) as data:
        percentiles = data["percentiles"]
        # Find the row for the 98th percentile rather than assuming it's last.
        (idx,) = np.where(percentiles == 98)
        if idx.size == 0:
            raise ValueError(f"No 98th percentile in {percentiles}")
        i = int(idx[0])

        np.savez(
            out,
            lat=data["lat"],
            lon=data["lon"],
            percentile=np.int64(98),
            year_count_map=data["year_count_maps"][i],
            n_years=data["n_years"],
            variable=data["variable"],
            pct_baseline=data["pct_baseline"],
        )
    print(f"Wrote {out} (year_count_map from row {i}, percentile=98)")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SRC
    out = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT
    extract(src, out)
