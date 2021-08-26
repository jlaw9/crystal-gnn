#!/bin/bash
#SBATCH --account=rlmolecule
#SBATCH --time=10:00:00
#SBATCH --job-name=test-crystals
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH -o log.out
#SBATCH -e err.out
#SBATCH --mail-user=jlaw@nrel.gov
#SBATCH --mail-type=END

source ~/.bashrc
module load cudnn/8.1.1/cuda-11.2
#conda activate crystals
conda activate ~/.conda-envs/rlmol

srun python train_model.py
