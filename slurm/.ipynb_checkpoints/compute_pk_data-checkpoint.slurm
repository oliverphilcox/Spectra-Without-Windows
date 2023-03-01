#!/bin/bash
#
#SBATCH --job-name=compute_pk_data
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ophilcox@princeton.edu
#SBATCH --ntasks=1
#SBATCH --time=01:59:59
#SBATCH --mem=50gb
#SBATCH --cpus-per-task=4
#SBATCH --array=0-24
#SBATCH --output=/home/ophilcox/out/compute_pk_data_all_%a.log

### COMPUTE Pk FUNCTIONS ON DATA

# Load modules
module load anaconda3
source activate ptenv

## PARAMETERS
# gf = grid-factor
# wtype = 0 for FKP weights or 1 for ML weights
patch=ngc
ztype=z1
wtype=0

batch_size=$SLURM_CPUS_PER_TASK
sim_min=`expr $SLURM_ARRAY_TASK_ID \* $batch_size `
sim_max=`expr $SLURM_ARRAY_TASK_ID \* $batch_size + $batch_size`

#sleep 5

if [[ $SLURM_ARRAY_TASK_ID -eq 13 ]]
then
  gf=3.0
  python2 -u ../pk/compute_pk_data.py -1 $patch $ztype $wtype $gf &> /home/ophilcox/out/compute_pk_data_boss.log

  gf=2.0
  python2 -u ../pk/compute_pk_data.py -1 $patch $ztype $wtype $gf  &>> /home/ophilcox/out/compute_pk_data_boss.log
  #
  # gf=1.3
  # python2 -u ../pk/compute_pk_data.py -1 $patch $ztype $wtype $gf  &>> /home/ophilcox/out/compute_pk_data_boss.log

else
  ## Run g = 3
  gf=3.0
  for (( i=$sim_min; i<=$sim_max; i++ ))
  do
   python2 -u ../pk/compute_pk_data.py $i $patch $ztype $wtype $gf &> /home/ophilcox/out/compute_pk_data_$i.log &
  done
  wait

  ### Run g = 2
  gf=2.0
  for (( i=$sim_min; i<=$sim_max; i++ ))
  do
    python2 -u ../pk/compute_pk_data.py $i $patch $ztype $wtype $gf  &>> /home/ophilcox/out/compute_pk_data_$i.log &
  done
  wait
  #
  # ### Run g = 1.3
  # gf=1.3
  # for (( i=$sim_min; i<=$sim_max; i++ ))
  # do
  #   python2 -u ../pk/compute_pk_data.py $i $patch $ztype $wtype $gf  &>> /home/ophilcox/out/compute_pk_data_$i.log &
  # done
  # wait
fi
