#!/bin/bash
#SBATCH --job-name RUN_EXP
#SBATCH --account=luts
#SBATCH --nodes 1
#SBATCH --partition=bigmem
#SBATCH --qos=serial
#SBATCH --time 96:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --output "out/slurm-%A_%a.log"
#SBATCH --mem=512G
#SBATCH --cpus-per-task=72

module load gcc/13.2.0
echo "${@:1}"
python -u "${@:1}"