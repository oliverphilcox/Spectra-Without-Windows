#!/bin/bash

#SBATCH --job-name=boss_fish
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ohp2105@columbia.edu
#SBATCH --ntasks=1
#SBATCH --time=23:58:59
#SBATCH --cpus-per-task=40
#SBATCH -N 1
#SBATCH -p cca
#SBATCH --output=/mnt/home/ophilcox/out/boss_fish.log

# Load modules
module load modules/2.1 openmpi python-mpi
source /mnt/home/ophilcox/mpivenv2/bin/activate

cd /mnt/home/ophilcox/BOSS-Without-Windows

## SL
PARAMFILE=paramfiles/boss_sL.param
python -u bk/compute_bk_fisher.py $PARAMFILE

## SC
PARAMFILE=paramfiles/boss_sC.param
python -u bk/compute_bk_fisher.py $PARAMFILE

## NL
PARAMFILE=paramfiles/boss_nL.param
python -u bk/compute_bk_fisher.py $PARAMFILE

## NC
PARAMFILE=paramfiles/boss_nC.param
python -u bk/compute_bk_fisher.py $PARAMFILE
