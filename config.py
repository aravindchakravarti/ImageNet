# config.py

import os

class Config:
    # Data
    DATA_SET_PATH = 'C:/Users/achakravarti/Documents/ERAv4/imagenette2-320'
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    IMAGE_SIZE = 32
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]
    NUM_CLASSES = 10

    # Model
    RESNET_LAYERS = [2, 2, 3, 2]
    USE_DEPTHWISE = (False, False, True, True)
    BASE_CHANNELS = 64

    # Training
    TOTAL_EPOCHS = 15
    LEARNING_RATE = 1e-3  # Default LR, can be overridden by LR finder
    WEIGHT_DECAY = 1e-4
    PCT_START = 0.3
    ANNEAL_STRATEGY = "cos"
    DIV_FACTOR = 25.0
    FINAL_DIV_FACTOR = 1e4

    # Checkpointing
    CHECKPOINT_DIR = os.path.join(os.getcwd(), "checkpoints")
    CHECKPOINT_FILE = "checkpoint_imagenette.pt"
    BEST_MODEL_FILE = "best_model_imagenette.pt"

    # LR Finder
    LR_FINDER_END_LR = 0.01
    LR_FINDER_NUM_ITER = 60

    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
