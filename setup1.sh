#!/bin/bash

sudo chmod 777 /mnt
mkdir /mnt/fl
cd /mnt/fl
mkdir benchmark

cp /home/ubuntu/FedScale/benchmark/dataset/download.sh download.sh
bash download.sh download femnist
bash download.sh download speech

mkdir data/device_info
cp /home/ubuntu/FedScale/benchmark/dataset/data/device_info/* ./data/device_info
cp /home/ubuntu/FedScale/client_device_capacity_ul_dl ./data/device_info