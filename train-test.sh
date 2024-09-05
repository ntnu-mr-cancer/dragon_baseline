#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

# Maximum is currently 30g, configurable in your algorithm image settings on grand challenge
MEM_LIMIT="4g"

for fold in 0 1 2 3 4
do
for task_name in "Task101_Example_sl_bin_clf" "Task102_Example_sl_mc_clf" "Task103_Example_mednli" "Task104_Example_ml_bin_clf" "Task105_Example_ml_mc_clf" "Task106_Example_sl_reg" "Task107_Example_ml_reg" "Task108_Example_sl_ner" "Task109_Example_ml_ner"
do
    jobname="$task_name-fold$fold"

    echo "========================================"
    echo "Training $jobname..."

    # skip if the output file already exists
    if [ -f "$SCRIPTPATH/test-output/$jobname/nlp-predictions-dataset.json" ]; then
        echo "Skipping $jobname, output predictions already exist"
        continue
    fi

    # Create the output directory if it does not exist
    mkdir -p $SCRIPTPATH/test-output/$jobname
    chmod 777 $SCRIPTPATH/test-output/$jobname

    # Do not change any of the parameters to docker run, these are fixed
    docker run --rm \
        --gpus=all \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        -v $SCRIPTPATH/test-input/$jobname:/input:ro \
        -v $SCRIPTPATH/test-output/$jobname:/output \
        joeranbosma/dragon_baseline

    docker run --rm \
        -v $SCRIPTPATH/test-output/$jobname:/output/ \
        python:3.10-slim cat /output/nlp-predictions-dataset.json

    echo ""

done
done
