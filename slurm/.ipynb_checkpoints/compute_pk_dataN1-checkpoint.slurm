#!/bin/bash
#
#SBATCH --job-name=compute_pk_dataN1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ophilcox@princeton.edu
#SBATCH --ntasks=1
#SBATCH --time=15:59:59
#SBATCH --mem=100gb
#SBATCH --cpus-per-task=1
#SBATCH --array=1-1
#SBATCH --output=/home/ophilcox/out/compute_pk_dataN1_%a.log

### COMPUTE Pk FUNCTIONS ON DATA

# Load modules
module load anaconda3
source activate ptenv

if [ "$SLURM_ARRAY_TASK_ID" -eq "0" ]
then
    SLURM_ARRAY_TASK_ID=-1
fi

## PARAMETERS
# gf = grid-factor
# wtype = 0 for FKP weights or 1 for ML weights
patch=ngc
ztype=z1
wtype=0

#gf=2.0
#python2 -u ../pk/compute_pk_data.py $SLURM_ARRAY_TASK_ID $patch $ztype $wtype $gf

gf=1.3
python2 -u ../pk/corr_compute_pk_data.py $SLURM_ARRAY_TASK_ID $patch $ztype $wtype $gf


#!/bin/bash
#
#SBATCH --job-name=compute_pk_dataN1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ophilcox@princeton.edu
#SBATCH --ntasks=16
#SBATCH -N 1
#SBATCH --time=01:58:59
#SBATCH --mem-per-cpu=150gb
#SBATCH --cpus-per-task=1
#SBATCH --array=0-128
#SBATCH --output=/home/ophilcox/out/compute_pk_dataN1_%a.log

# Run 1 job per task
N_JOB=$SLURM_NTASKS                # create as many jobs as tasks

# Load modules
module load anaconda3
source activate ptenv

patch = ngc
ztype = z1
wtype=0
gf=1.3

for((i=1;i<=$N_JOB;i++))
do
  (( task_id = SLURM_ARRAY_TASK_ID * N_JOB + i ))
  #
  # OUTFILE=/projects/QUIJOTE/Oliver/nseries_patchy_pkbk_fish/pk_corr_nseries_patchy${task_id}_fkp_N100_g2.0_k0.000_0.160_0.010.txt
  # if [ -f "$OUTFILE" ]; then
  #   echo "$task_id already computed!"
  #   continue
  # fi

  echo "Running on simulation $task_id"

  python2 -u ../pk/corr_compute_pk_data_nseries_patchy.py $task_id $wtype $gf &

done

#Wait for all
wait

echo
echo "All done!"
