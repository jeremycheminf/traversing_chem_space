#!/bin/bash
#SBATCH --job-name=30_mlp_batch_bald_random
#SBATCH --output=/home/tilborgd/projects/Active_Learning_Simulation/out/30_mlp_batch_bald_random.out
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=18
#SBATCH --gpus-per-node=1
#SBATCH --time=120:00:00

source $HOME/anaconda3/etc/profile.d/conda.sh
export PYTHONPATH="$PYTHONPATH:$HOME/projects/Active_Learning_Simulation"
$HOME/anaconda3/envs/molml/bin/python -u $HOME/projects/Active_Learning_Simulation/experiments/active_learning.py -o /home/tilborgd/projects/Active_Learning_Simulation/results -acq batch_bald -bias random -arch mlp > $HOME/projects/Active_Learning_Simulation/results/mlp_batch_bald_random.log