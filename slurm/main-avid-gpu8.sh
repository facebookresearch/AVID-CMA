#! /bin/bash
#SBATCH --job-name=AVID
#SBATCH --output=/checkpoint/%u/jobs/%x-%j.out
#SBATCH --error=/checkpoint/%u/jobs/%x-%j.err
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=80
#SBATCH --mem=250G
#SBATCH --time=48:00:00
#SBATCH --mail-user=pmorgado@fb.com
#SBATCH --partition=priority
#SBATCH --comment="ECCV conference deadline."

#SBATCH --signal=B:USR1@60
#SBATCH --open-mode=append

# Install signal handler
trap_handler () {
   echo "Caught signal: " $1
   # SIGTERM must be bypassed
   if [ "$1" = "TERM" ]; then
       echo "bypass sigterm"
   else
     # Submit a new job to the queue
     echo "Requeuing " $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID
     # SLURM_JOB_ID is a unique representation of the job, equivalent
     # to above
     scontrol requeue $SLURM_JOB_ID
   fi
}
trap 'trap_handler USR1' USR1
trap 'trap_handler TERM' TERM

srun --label python main-avid.py $1 --quiet --dist-url tcp://localhost:1234 --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0
