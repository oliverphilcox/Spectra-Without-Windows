#!/bin/bash

#SBATCH --job-name=boss_bk_nL
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ohp2105@columbia.edu
#SBATCH --ntasks=1
#SBATCH --time=23:58:59
#SBATCH --mem=500gb
#SBATCH --cpus-per-task=40
#SBATCH -N 1
#SBATCH -p cca
#SBATCH --array=0-1
#SBATCH --output=/mnt/home/ophilcox/out/boss_bk_nL_%a.log

# Run 1 job per CPU
N_JOB=$SLURM_CPUS_PER_TASK                # create as many jobs as tasks

# Load modules
module load modules/2.1 openmpi python-mpi
source /mnt/home/ophilcox/mpivenv2/bin/activate

PARAMFILE=../paramfiles/boss_nL.param

for((i=1;i<=$N_JOB;i++))
do
  (( task_id = SLURM_ARRAY_TASK_ID * N_JOB + i ))

  echo "Running on random set $task_id"

  python -u ../bk/compute_bk_randoms.py $task_id $PARAMFILE &

done

#Wait for all
wait

echo
echo "All done!"


