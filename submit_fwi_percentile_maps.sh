#!/bin/bash

#SBATCH --account=rev
#SBATCH --time=0-04:00:00 # walltime; run takes ~20-45 min, 4h is headroom
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user=rebecca.fuchs@nlr.gov
#SBATCH --mail-type=FAIL
#SBATCH --mem=172000 # RAM in MB (peak usage is only ~7 GB/block)
#SBATCH --job-name=fwi_pctile
#SBATCH --output=logs/slurm-%j.out

#load your default settings
. $HOME/.bashrc

conda activate sup3r

cd /home/rrolph/wildfire
python fwi_percentile_maps.py
