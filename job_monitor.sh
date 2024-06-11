#!/bin/bash

while true; do
    ps_out=$(ps -ef | grep python | grep -v 'grep' | grep FedScale)
    
    if [[ -n "$ps_out" ]]; then
        # FedScale process is running, wait before checking again
        sleep 60
    else
        # FedScale process is not running, exit the loop
        echo "FedScale process not found. Exiting monitor script."
        break
    fi
done
