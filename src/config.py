import torch


class Config:
    """Central configuration for Green AI experiments."""

    DATA_ROOT = "./data"
    RESULT_DIR = "./results"
    LOG_FILE = "./results/experiment_log.csv"

    BATCH_SIZE = 64
    LEARNING_RATE = 0.1
    NUM_EPOCHS = 10
    NUM_WORKERS = 2
    SEED = 42

    IMG_SIZE = 32
    NUM_CLASSES = 100
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def get_device(technique: str):
        return "cpu" if technique == "quantization" else Config.DEVICE
