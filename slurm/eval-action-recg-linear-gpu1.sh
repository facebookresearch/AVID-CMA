#! /bin/bash
#################################################################################
#     File Name           :     submit-singlenode-training.sh
#################################################################################
#
#SBATCH --job-name=ActionLinear
#SBATCH --output=/checkpoint/%u/jobs/%x-%j.out
#SBATCH --error=/checkpoint/%u/jobs/%x-%j.err
#SBATCH --partition=learnfair
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --mail-user=pmorgado@fb.com
#SBATCH --partition=priority
#SBATCH --comment="ECCV conference deadline."

srun --label python eval-action-recg-linear-head.py $1 $2 $3 $4 $5 $6 $7 --quiet