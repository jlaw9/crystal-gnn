#!/bin/bash
#SBATCH --account=rlmolecule
#SBATCH --time=1:00:00
#SBATCH --partition=debug
#SBATCH --job-name=eval-crystals
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH -o outputs/eval_log.out
#SBATCH -e outputs/eval_err.out
#SBATCH --mail-user=jlaw@nrel.gov
#SBATCH --mail-type=END

source ~/.bashrc_conda
module load cudnn/8.1.1/cuda-11.2
conda activate crystals

#srun python run_test.py --config config/zintl-unrelaxed.yaml
#srun python run_test.py --config config/battery-unrelaxed.yaml
#srun python run_test.py --config config/battery-relaxed.yaml
srun python src/eval_test.py --config config/battery-relaxed-unrelaxed.yaml
