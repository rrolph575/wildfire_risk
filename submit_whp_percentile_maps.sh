#!/bin/bash

#SBATCH --account=rev
#SBATCH --time=0-02:00:00 # walltime; the 90m run is ~5-15 min, 2h is headroom
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user=rebecca.fuchs@nlr.gov
#SBATCH --mail-type=FAIL
#SBATCH --mem=64000 # RAM in MB (streams in stripes; peak well under this)
#SBATCH --job-name=whp_pctile
#SBATCH --output=logs/slurm-%j.out

#load your default settings
. $HOME/.bashrc

conda activate rev

cd /home/rrolph/wildfire
python whp_percentile_maps.py
