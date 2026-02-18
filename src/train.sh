#!/bin/bash

###############################################################
# Script for training  ESCAPE using main.py
#  Fold 1 y Fold 2 with ensemble 
###############################################################



DEVICE=2
MODE="MultiModal"
SEED=42
BATCH=64

# Path for FILES)
FOLD1_FILE="path/to/Fold1.csv" 
FOLD2_FILE="path/to/Fold2.csv" 

########################################
# =============== FOLD 1 ===============
########################################

FOLD=1
RUN_NAME="${MODE}_fold${FOLD}_seed${SEED}"

CUDA_VISIBLE_DEVICES=$DEVICE python -u main.py \
    --fold $FOLD \
    --mode $MODE \
    --run_name $RUN_NAME \
    --batch_size $BATCH \
    --wandb True \
    --seed $SEED \
    --fold1_file $FOLD1_FILE \
    --fold2_file $FOLD2_FILE

########################################
# =============== FOLD 2 ===============
########################################

FOLD=2
RUN_NAME="${MODE}_fold${FOLD}_seed${SEED}"

CUDA_VISIBLE_DEVICES=$DEVICE python -u main.py \
    --fold $FOLD \
    --mode $MODE \
    --run_name $RUN_NAME \
    --batch_size $BATCH \
    --wandb True \
    --seed $SEED \
    --fold1_file $FOLD1_FILE \
    --fold2_file $FOLD2_FILE

###############################################################
echo "✔ Training completed for Fold1 and Fold2."
###############################################################
