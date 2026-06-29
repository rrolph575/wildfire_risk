"""
List the arrays stored in an .npz file.

For each array, prints its name, shape, dtype, and -- for small (0-d / 1-d, few
elements) arrays -- its value, so you can quickly inspect the contents of a
saved .npz without loading it elsewhere.

Usage:
    python list_npz.py [path_to_npz]

If no path is given, defaults to the FWI percentile maps file.
"""

import sys

import numpy as np

DEFAULT_NPZ = (
    "/projects/rev/projects/ntps/fy26/underground_transmission/data/fwi/"
    "fwi_tc_pcthist2000_2014_cnt2025_2059_percentile_maps.npz"
)


def list_npz(path):
    with np.load(path, allow_pickle=False) as data:
        names = data.files
        print(f"{path}")
        print(f"{len(names)} array(s):\n")
        width = max(len(n) for n in names)
        for name in names:
            arr = data[name]
            line = f"  {name:<{width}}  shape={str(arr.shape):<14} dtype={arr.dtype}"
            # Show the value when it's small enough to be informative.
            if arr.ndim == 0 or arr.size <= 5:
                line += f"  value={arr}"
            print(line)


if __name__ == "__main__":
    npz_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_NPZ
    list_npz(npz_path)
