#!/bin/bash
#
#SBATCH --job-name=compute_pk_randoms
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ophilcox@princeton.edu
#SBATCH --ntasks=1
#SBATCH --time=23:58:59
#SBATCH --mem-per-cpu=25GB
#SBATCH --cpus-per-task=1
#SBATCH --array=1-100
#SBATCH --output=/home/ophilcox/out/pk_randoms_%a.log

### Sample SLURM script
# Here we compute contributions to the Pk Fisher matrix using random catalogs
# A similar script would work for the other Python scripts...

# Load modules
module load anaconda3/2021.11
source activate ptenv

## Define parameters
rand_it = $SLURM_ARRAY_TASK_ID
paramfile = ../params/boss_n1.param

## Run script
python2 -u ../pk/compute_pk_randoms.py $rand_it $paramfile