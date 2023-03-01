#!/bin/bash

#SBATCH --job-name=patchy_pk_nC
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ohp2105@columbia.edu
#SBATCH --ntasks=1
#SBATCH --time=71:58:59
#SBATCH --mem=500gb
#SBATCH --cpus-per-task=20
#SBATCH -N 1
#SBATCH -p cca
#SBATCH --array=0-4
#SBATCH --output=/mnt/home/ophilcox/out/patchy_pk_nC_%a.log

# Run 1 job per CPU
N_JOB=$SLURM_CPUS_PER_TASK                # create as many jobs as tasks

# Load modules
module load modules/2.1 openmpi python-mpi
source /mnt/home/ophilcox/mpivenv2/bin/activate

PARAMFILE=../paramfiles/patchy_nC.param

for((i=1;i<=$N_JOB;i++))
do
  (( task_id = SLURM_ARRAY_TASK_ID * N_JOB + i ))

  echo "Running on random set $task_id"

  python -u ../pk/compute_pk_randoms.py $task_id $PARAMFILE &

done

#Wait for all
wait

echo
echo "All done!"


