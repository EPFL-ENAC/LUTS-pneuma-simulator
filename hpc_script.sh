#!/bin/bash
#SBATCH --job-name RUN_EXP
#SBATCH --account=luts
#SBATCH --nodes 1
#SBATCH --partition=hugemem
#SBATCH --qos=serial
#SBATCH --time 64:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --output "out/slurm-%A_%a.log"
#SBATCH --mem=1700G
#SBATCH --cpus-per-task=64

module load gcc/13.2.0
echo "${@:1}"
python -u "${@:1}"