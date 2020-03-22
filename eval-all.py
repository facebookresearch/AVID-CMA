import os
import argparse

parser = argparse.ArgumentParser(description='Launch all evals')
parser.add_argument('cfg', metavar='CFG', help='model config file')
parser.add_argument('--small', action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()
    if not args.small:
        eval_cmds = [
            f'sbatch slurm/eval-action-recg-gpu2.sh configs/benchmark/ucf/r2plus1d/r2plus1d-wucls-8at16-fold1.yaml {args.cfg}',
            f'sbatch slurm/eval-action-recg-gpu2.sh configs/benchmark/ucf/r2plus1d/r2plus1d-wucls-8at16-fold2.yaml {args.cfg}',
            f'sbatch slurm/eval-action-recg-gpu2.sh configs/benchmark/ucf/r2plus1d/r2plus1d-wucls-8at16-fold3.yaml {args.cfg}',
            f'sbatch slurm/eval-action-recg-gpu2.sh configs/benchmark/hmdb51/r2plus1d/r2plus1d-wucls-8at16-fold1.yaml {args.cfg}',
            f'sbatch slurm/eval-action-recg-gpu2.sh configs/benchmark/hmdb51/r2plus1d/r2plus1d-wucls-8at16-fold2.yaml {args.cfg}',
            f'sbatch slurm/eval-action-recg-gpu2.sh configs/benchmark/hmdb51/r2plus1d/r2plus1d-wucls-8at16-fold3.yaml {args.cfg}',
            f'sbatch slurm/eval-action-recg-gpu2.sh configs/benchmark/ucf/r2plus1d/r2plus1d-wucls-32at16-fold1.yaml {args.cfg}',
            f'sbatch slurm/eval-action-recg-gpu2.sh configs/benchmark/ucf/r2plus1d/r2plus1d-wucls-32at16-fold2.yaml {args.cfg}',
            f'sbatch slurm/eval-action-recg-gpu2.sh configs/benchmark/ucf/r2plus1d/r2plus1d-wucls-32at16-fold3.yaml {args.cfg}',
            f'sbatch slurm/eval-action-recg-gpu2.sh configs/benchmark/hmdb51/r2plus1d/r2plus1d-wucls-32at16-fold1.yaml {args.cfg}',
            f'sbatch slurm/eval-action-recg-gpu2.sh configs/benchmark/hmdb51/r2plus1d/r2plus1d-wucls-32at16-fold2.yaml {args.cfg}',
            f'sbatch slurm/eval-action-recg-gpu2.sh configs/benchmark/hmdb51/r2plus1d/r2plus1d-wucls-32at16-fold3.yaml {args.cfg}',
            f'sbatch slurm/eval-action-recg-linear-gpu2.sh configs/benchmark/kinetics/r2plus1d/8x224x224-linear.yaml {args.cfg}',
            f'sbatch slurm/eval-snd-recg-svm.sh configs/benchmark/dcase/audio-svm.yaml {args.cfg}',
            f'sbatch slurm/eval-snd-recg-svm.sh configs/benchmark/esc50/audio-svm.yaml {args.cfg}'
        ]
    else:
        eval_cmds = [
            f'sbatch slurm/eval-action-recg-gpu1.sh configs/benchmark/ucf/r2plus1d-small/r2plus1d-wucls-8at16.yaml {args.cfg}',
            f'sbatch slurm/eval-action-recg-gpu1.sh configs/benchmark/hmdb51/r2plus1d-small/r2plus1d-wucls-8at16.yaml {args.cfg}',
            f'sbatch slurm/eval-action-recg-linear-gpu2.sh configs/benchmark/kinetics/r2plus1d-small/8x128x128-linear.yaml {args.cfg}',
            f'sbatch slurm/eval-snd-recg-svm.sh configs/benchmark/esc50/audio-small-svm.yaml {args.cfg}',
            f'sbatch slurm/eval-snd-recg-svm.sh configs/benchmark/dcase/audio-small-svm.yaml {args.cfg}'
        ]

    for cmd in eval_cmds:
        os.system(cmd)