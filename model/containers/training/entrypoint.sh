#!/bin/bash

# Argument options
LONG=dataset_size:,sample_distortion:,sweep_id:,force_dataset_download

# Parse through arguments
function parse_args()
{
    options=$(getopt --long $LONG --long name: -- "$@")
    [ $? -eq 0 ] || {
        echo "entrypoint: Incorrect option provided. Stopping..."
        exit 1
    }
    eval set -- "$options"
    while true; do
        case "$1" in
        --dataset_size)
            shift
            DATASET_SIZE="$1"
            ;;
        --sample_distortion)
            shift;
            SAMPLE_DISTORTION="$1"
            ;;
        --sweep_id)
            shift
            SWEEP_ID="$1"
            ;;
        --force_dataset_download)
            FORCE_DATASET_DOWNLOAD=1
            ;;
        --)
            break
            ;;
        esac
        shift
    done
}
 
DATASET_SIZE=small
SAMPLE_DISTORTION=generic-updated
SWEEP_ID=
FORCE_DATASET_DOWNLOAD=
parse_args $0 "$@"

if [ -z "$AWS_ACCESS_KEY_ID" ]
then 
    echo "entrypoint: AWS_ACCESS_KEY_ID env var is not set. Stopping..."
    exit 1
fi

if [ -z "$AWS_SECRET_ACCESS_KEY" ]
then 
    echo "entrypoint: AWS_SECRET_ACCESS_KEY env var is not set. Stopping..."
    exit 1
fi

if [ -z "$AWS_DEFAULT_REGION" ]
then 
    echo "entrypoint: AWS_DEFAULT_REGION env var is not set. Stopping..."
    exit 1
fi

if [ -z "$WANDB_API_KEY" ]
then 
    echo "entrypoint: Weights and Biases WANDB_API_KEY env var is not set. Stopping..."
    exit 1
fi

ls -A dataset

if [ "$(ls -A dataset)" ]
then
    echo "entrypoint: Dataset already present..."
    if [ "$FORCE_DATASET_DOWNLOAD" ]
    then
        echo "entrypoint: Force download flag set. Removing dataset..."
        rm -r dataset/*
    fi
fi

if [ -z "$(ls -A dataset)" ]
then
    echo "entrypoint: Downloading dataset $DATASET_SIZE/$SAMPLE_DISTORTION..."
    aws s3 cp s3://ssa-data/dataset/$DATASET_SIZE/$SAMPLE_DISTORTION dataset --recursive --quiet
fi

echo 'entrypoint: ' && find dataset/training -type d -exec sh -c "echo \"{}\" && ls -l {} | wc -l" \;
echo 'entrypoint: ' && find dataset/validation -type d -exec sh -c "echo \"{}\" && ls -l {} | wc -l" \;

wandb login $WANDB_API_KEY
wandb on

# Handle script execution
if [ -z "$SWEEP_ID" ]
then
    echo "entrypoint: Performing normal training session..."
    python model.py
else
    echo "entrypoint: Performing sweep training session using $SWEEP_ID..."
    wandb agent $SWEEP_ID
fi