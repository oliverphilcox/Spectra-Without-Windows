#!/bin/bash

#SBATCH --job-name=compute_bk_randoms_nseries
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ophilcox@princeton.edu
#SBATCH --ntasks=1
#SBATCH --time=23:58:59
#SBATCH --mem-per-cpu=25gb
#SBATCH --cpus-per-task=4
#SBATCH -N 1
#SBATCH --array=0-99
#SBATCH --output=/home/ophilcox/out/compute_bk_randoms_nseries_%a.log

# Run 1 job per task
N_JOB=$SLURM_NTASKS                # create as many jobs as tasks

# Load modules
module load anaconda3/2021.11
source activate ptenv

PARAMFILE=../paramfiles/nseries_l4.param

for((i=1;i<=$N_JOB;i++))
do
  (( task_id = SLURM_ARRAY_TASK_ID * N_JOB + i ))

  echo "Running on random set $task_id"

  python2 -u ../bk/compute_bk_randoms.py $task_id $PARAMFILE &

done

#Wait for all
wait

echo
echo "All done!"


