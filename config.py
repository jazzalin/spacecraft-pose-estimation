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

    BETA=1