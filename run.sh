#!/bin/bash
#SBATCH --account=def-bengioy
#SBATCH --gres=gpu:v100l:4             # Number of GPUs (per node)
#SBATCH --mem=32G               # memory (per node)
#SBATCH --time=00-10:00            # time (DD-HH:MM)
#SBATCH --job-name=biasly_classifer
#SBATCH -o /home/jangeunb/projects/def-bengioy/jangeunb/lstm/output/train-%j.out
module load cuda/10.0.130
module load cuda cudnn
ulimit -c unlimited
start_time=$(date)
echo "Start Time: $start_time"
python3 main.py
end_time=$(date)
echo "Start Time: $start_time"
echo "End Time: $end_time"
