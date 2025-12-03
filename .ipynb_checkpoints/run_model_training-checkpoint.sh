#!/bin/bash
#SBATCH --job-name=GAT_experiments
#SBATCH --partition=DGX
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --output=GAT_training.out
#SBATCH --mem=100G

cd /orfeo/scratch/dssc/cmoret/RNAErnie/


source ernia_dgx/bin/activate
# source ernia_h100/bin/activate


cd /orfeo/scratch/dssc/cmoret/miRNA_GAT


python train_models.py 