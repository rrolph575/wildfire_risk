#!/bin/bash

#SBATCH --account=alcaps
#SBATCH --partition=debug
#SBATCH --time=0-01:00:00 # debug partition caps walltime at 1h; run is minutes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user=rebecca.fuchs@nlr.gov
#SBATCH --mail-type=FAIL
#SBATCH --mem=32000 # RAM in MB (only the cached A field + region subsets)
#SBATCH --job-name=region_ug_maps
#SBATCH --output=logs/slurm-%j.out

#load your default settings
. $HOME/.bashrc

cd /home/rrolph/wildfire/wildfire_risk_changes_capacity

# One-time WHP burnable sampling needs rasterio, which lives in the `rev` env.
# It caches to _burnable_cache_<region>.npz; only run if the caches are missing.
if [ ! -f _burnable_cache_socal.npz ] || [ ! -f _burnable_cache_tva.npz ]; then
    conda activate rev
    python regional_undergrounding_maps.py
    conda deactivate
fi

# Main run in reeds2 (loads the cached burnable flags; no rasterio needed).
conda activate reeds2
python regional_undergrounding_maps.py
