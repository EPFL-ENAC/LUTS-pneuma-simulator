#!/bin/bash
#SBATCH --job-name RUN_EXP
#SBATCH --account=luts
#SBATCH --nodes 4
#SBATCH --partition=standard
#SBATCH --qos=parallel
#SBATCH --time 06:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --output "out/slurm-%A_%a.log"
#SBATCH --mem=64G
#SBATCH --cpus-per-task=72

module load gcc/13.2.0
echo "${@:1}"
python -u "${@:1}"