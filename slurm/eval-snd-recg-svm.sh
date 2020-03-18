#! /bin/bash
#################################################################################
#     File Name           :     submit-singlenode-training.sh
#################################################################################
#
#SBATCH --job-name=EvalSnd
#SBATCH --output=/checkpoint/%u/jobs/%x-%j.out
#SBATCH --error=/checkpoint/%u/jobs/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=250G
#SBATCH --time=24:00:00
#SBATCH --mail-user=pmorgado@fb.com
#SBATCH --partition=priority
#SBATCH --comment="ECCV."

srun --label python eval-snd-recg-svm.py $1 $2 --quiet
