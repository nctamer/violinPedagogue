#!/bin/bash
#SBATCH --job-name=PEdagogue
#SBATCH -n 4
#SBATCH --mem 40000
#SBATCH --gres=gpu:1
#SBATCH -p high                     # Partition to submit to
#SBATCH -o %x-%j.out # File to which STDOUT will be written
#SBATCH -e %x-%j.err # File to which STDERR will be written

module load libsndfile
module load RubberBand
module load FFmpeg
module load Miniconda3
eval "$(conda shell.bash hook)"
conda init bash
conda activate supervised

module load CUDA
module load cuDNN

python train.py --no-augment --epochs 5000 --batch-size 32

