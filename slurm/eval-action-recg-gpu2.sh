#! /bin/bash
#################################################################################
#     File Name           :     submit-singlenode-training.sh
#################################################################################
#
#SBATCH --job-name=EvalAction
#SBATCH --output=/checkpoint/%u/jobs/%x-%j.out
#SBATCH --error=/checkpoint/%u/jobs/%x-%j.err
#SBATCH --partition=learnfair
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=40
#SBATCH --mem=250G
#SBATCH --time=24:00:00
#SBATCH --mail-user=pmorgado@fb.com
##SBATCH --partition=priority
##SBATCH --comment="ECCV."

srun --label python eval-action-recg.py $1 $2 $3 $4 $5 $6 $7 --quiet