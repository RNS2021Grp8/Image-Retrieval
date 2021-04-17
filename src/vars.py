import os
from deploy.settings import BASE_DIR

DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "data"))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "models"))
BATCH_SIZE = 256
NUM_OF_EPOCHS = 30