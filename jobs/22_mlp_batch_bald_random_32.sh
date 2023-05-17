#!/bin/bash
#SBATCH --job-name=22_mlp_batch_bald_random_32
#SBATCH --output=/home/tilborgd/projects/Active_Learning_Simulation/out/22_mlp_batch_bald_random_32.out
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=18
#SBATCH --gpus-per-node=1
#SBATCH --time=120:00:00

source $HOME/anaconda3/etc/profile.d/conda.sh
export PYTHONPATH="$PYTHONPATH:$HOME/projects/Active_Learning_Simulation"
$HOME/anaconda3/envs/molml/bin/python -u $HOME/projects/Active_Learning_Simulation/experiments/main.py -o /home/tilborgd/projects/Active_Learning_Simulation/results -acq batch_bald -bias random -arch mlp -batch_size 32 > $HOME/projects/Active_Learning_Simulation/results/mlp_batch_bald_random_32.log
