#!/bin/bash

tar -xf ~/projects-qfyan/data/femnist.tar.gz -C $SLURM_TMPDIR
echo "Done unzipping training data"

module load python/3.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r ~/FedScale/requirements.txt
echo "Done setting up python environment"
