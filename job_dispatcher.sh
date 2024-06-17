#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 script_or_directory"
    exit 1
fi

TARGET=$1

if [ -d "$TARGET" ]; then
    for SCRIPT in $TARGET/*.sh; do
        echo "Submitting $SCRIPT"
        sbatch $SCRIPT
    done
elif [ -f "$TARGET" ]; then
    echo "Submitting $TARGET"
    sbatch $TARGET
else
    echo "Invalid target: $TARGET"
    exit 1
fi