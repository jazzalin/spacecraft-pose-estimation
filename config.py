# Network configuration and parameters
import numpy as np


class Config:
    # Paths
    DATASET_ROOT='../speed'
    ANNOTATIONS_ROOT='./annotations'

    # Dataset
    SPLIT_TRAINING_INDEX=1000 # default: None
    SANITY_CHECK_INDEX=100

    # Training
    USE_GPU=True
    WORKERS=6
    TRAIN_BS=16
    TEST_BS=1

    EPOCH=200
    PATIENCE=10
    BETA=1

    # Attitude regression parameters
    ATT="prv" # Options: "ep", "prv", "ea"

    # Euler Parameters (EP)
    EP_LR_T=0.1
    EP_LR_A=0.001

    # Principal Rotation Vectors (PRV)
    PRV_LR_T=0.1
    PRV_LR_A=0.001
