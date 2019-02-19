#!/bin/sh
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=sent_analysis
#SBATCH --output=sent_analysis.out
echo 'test started\n\n'
start=$(date +%s.%N)

python main.py

dur=$(echo "$(date +%s.%N) - $start" | bc)
printf "Execution time: %.6f seconds" $dur
echo '\n\ntest finished'
~







~
~                                                                                                                                                             
~                                                                                                                                                             
~               
