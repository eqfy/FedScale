#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=5:30:00
#SBATCH --output=%N-%j.out
#SBATCH --account=def-mlecuyer
#SBATCH --mail-user=andrewl02y@gmail.com
#SBATCH --mail-type=ALL


echo "Starting script..."

source $HOME/FedScale/env_setup.sh ~/projects/def-mlecuyer/qfyan/data/femnist.tar.gz

bash ${FEDSCALE_HOME}/fedscale.sh driver start ./benchmark/configs/baseline/fedavg_femnist_shf.yml
    
sleep 30

bash $HOME/FedScale/job_monitor.sh

echo "Script completed."
