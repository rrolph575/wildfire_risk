#!/bin/bash

#SBATCH --account=rev
#SBATCH --partition=debug
#SBATCH --time=0-01:00:00 # debug partition caps walltime at 1h; run is ~5-15 min
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
