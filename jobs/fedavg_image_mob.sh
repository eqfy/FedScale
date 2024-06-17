#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:30:00
#SBATCH --output=%N-%j.out
#SBATCH --account=def-mlecuyer


echo "Starting script..."

source $HOME/FedScale/env_setup.sh ~/projects/def-mlecuyer/qfyan/data/open_images.tar.gz

bash ${FEDSCALE_HOME}/fedscale.sh driver start ./benchmark/configs/baseline/fedavg_image_mob.yml
    
sleep 30

bash $HOME/FedScale/job_monitor.sh

echo "Script completed."
