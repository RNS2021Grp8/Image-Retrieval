import os

ROOT_DIR = os.path.abspath(os.path.join("..", os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, "data"))
MODEL_PATH = os.path.abspath(os.path.join(ROOT_DIR, "model"))
BATCH_SIZE = 256
NUM_OF_EPOCHS = 30