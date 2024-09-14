#!/bin/bash
#SBATCH -n 8
#SBATCH -J med
#SBATCH -o %j.out
#SBATCH -p publicgpu
#SBATCH -q wildfire
#SBATCH -e %j.ERROR
#SBATCH -t 6-24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ssganta1@asu.edu

module load anaconda/py3
source activate med
python gastricNB_20.py -i resultsGastricCancer_20.csv


