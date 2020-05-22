#!/bin/bash
 
# Argument options
LONG=api_key:,sweep_id:,config:,entity:,project:

# Parse through arguments
function args()
{
    options=$(getopt --long $LONG --long name: -- "$@")
    [ $? -eq 0 ] || {
        echo "wandb_setup: Incorrect option provided"
        exit 1
    }
    eval set -- "$options"
    while true; do
        case "$1" in
        --api_key)
            shift
            export WANDB_API_KEY="$1"
            ;;
        --sweep_id)
            shift;
            SWEEP_ID="$1" 
            ;;
        --project)
            shift
            export WANDB_PROJECT="$1"
            ;;
        --entity)
            shift
            export WANDB_ENTITY="$1"
            ;;
        --config)
            shift;
            CONFIG="$1" 
            ;;
        --)
            break
            ;;
        esac
        shift
    done
}
 
args $0 "$@"

# Colors fail on remote instance?

# Handle login
if [ "$WANDB_API_KEY" == "" ]; then
    echo -e "\e[1;35mwandb_setup\e[0m: \e[31mERROR\e[0m api_key is a required parameter. "
    echo -e "\e[1;35mwandb_setup\e[0m: \e[31mERROR\e[0m Please add your api_key to your estimator's hyperparameter argument."
    exit 0
else
    wandb login $WANDB_API_KEY
fi

# Handle script execution
if [ "$SWEEP_ID" == "" ]; then
    echo -e "\e[1;35mwandb_setup\e[0m: \U1F3CB Performing normal training session."
    if [ "$CONFIG" == "" ]; then
        echo -e "\e[1;35mwandb_setup\e[0m: \U1F3CB Using default configuration."
        python3 model.py
    else
        echo -e "\e[1;35mwandb_setup\e[0m: \U1F984 Using custom configuration."
        python3 model.py --config $CONFIG
    fi
else
    echo -e "\e[1;35mwandb_setup\e[0m: \U1F9F9 Performing sweep training session."
    python3 model.py --wandb-sweep-id $SWEEP_ID
fi