#!/bin/bash
#
#SBATCH --job-name=compute_bk_dataN1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ophilcox@princeton.edu
#SBATCH --ntasks=1
#SBATCH --time=05:58:59
#SBATCH --mem-per-cpu=10GB
#SBATCH --cpus-per-task=16
#SBATCH --array=0-0
#SBATCH --output=/home/ophilcox/out/compute_bk_dataN1_%a.log

# Load modules
module load anaconda3
source activate ptenv

## PARAMETERS
# gf = grid-factor
# wtype = 0 for FKP weights or 1 for ML weights

PARAMFILE=../bk/patchy_n1_bk.param
python2 -u ../bk/compute_bk_fisher_general.py $PARAMFILE