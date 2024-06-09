#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=5:30:00
#SBATCH --output=%N-%j.out
#SBATCH --account=def-mlecuyer


echo "Starting script..."

source $HOME/FedScale/env_setup.sh ~/projects/def-mlecuyer/qfyan/data/femnist.tar.gz

bash ${FEDSCALE_HOME}/fedscale.sh driver start ./benchmark/configs/femnist/conf.yml

sleep 19800

echo "Script completed."
