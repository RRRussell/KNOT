#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --partition=zhanglab.p
#SBATCH --nodelist=galaxy
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00
#SBATCH --output=jupyter_%j.log

# module load slurm
module load anaconda
source activate knot  # 你的环境

PORT=8888
HOST=$(hostname)

echo "Hostname: $HOST"
echo "PORT: $PORT"

jupyter lab --no-browser --ip=0.0.0.0 --port=$PORT
