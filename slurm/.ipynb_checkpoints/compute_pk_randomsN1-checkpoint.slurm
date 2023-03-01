#!/bin/bash
#
#SBATCH --job-name=compute_pk_randomsN1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ophilcox@princeton.edu
#SBATCH --ntasks=10
#SBATCH --time=11:58:59
#SBATCH --mem=100gb
#SBATCH --cpus-per-task=1
#SBATCH -N 1
#SBATCH --array=0-9
#SBATCH --output=/home/ophilcox/out/compute_pk_randomsN1_%a.log

### COMPUTE Pk FUNCTIONS ON RANDOMS

# Run 1 job per task
N_JOB=$SLURM_NTASKS                # create as many jobs as tasks

# Load modules
module load anaconda3
source activate ptenv

## PARAMETERS
patch=ngc
ztype=z1
wtype=0
gf=1.3

for((i=1;i<=$N_JOB;i++))
do
  (( task_id = SLURM_ARRAY_TASK_ID * N_JOB + i ))
  echo "Running on random set $task_id"
  python2 -u ../pk/compute_pk_randoms.py $task_id $patch $ztype $wtype $gf &
done

#Wait for all
wait

echo
echo "All done!"
