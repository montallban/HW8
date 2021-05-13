#!/bin/bash

#SBATCH --partition=normal
#SBATCH --ntasks=1
# memory in MB
#SBATCH --mem=35000
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=results_hw7/unet_%04a.txt
#SBATCH --error=results_hw7/error_unet_%04a.txt
#SBATCH --time=24:00:00
#SBATCH --job-name=unet_hw7
#SBATCH --mail-user=michael.montalbano@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/mcmontalbano/homework7
#SBATCH --array=0-5
#
#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up
source ~fagg/pythonenv/tensorflow/bin/activate

python base.py -exp_index $SLURM_ARRAY_TASK_ID -epochs 2000 -patience 1000 -L2_regularizer 0.001 -experiment_type 'basic' -dataset '/scratch/fagg/chesapeake/' -network 'unet'
