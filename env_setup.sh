#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 data_unzip_location"
    exit 1
fi

DATA_UNZIP_LOCATION=$1

echo "Setting up python environment"
module load python/3.10
virtualenv --no-download --seeder=pip $SLURM_TMPDIR/env 
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r ~/FedScale/requirements.txt 
echo "Done setting up python environment"

echo "Unzipping training data"
tar -xf $DATA_UNZIP_LOCATION -C $SLURM_TMPDIR
echo "Done unzipping training data"

if [ -f $SLURM_TMPDIR/env/bin/activate ]; then
    source $SLURM_TMPDIR/env/bin/activate
else
    echo "Virtual environment not found!"
    exit 1
fi

cd $HOME/FedScale || { echo "FedScale directory not found!"; ls; exit 1; }

pip install -e .
FEDSCALE_HOME=$(pwd)
echo "FEDSCALE_HOME set to $FEDSCALE_HOME"
echo "export FEDSCALE_HOME=$FEDSCALE_HOME" >> ~/.bashrc
