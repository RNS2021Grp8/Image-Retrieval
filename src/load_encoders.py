import os
from tensorflow import keras
from .vars import MODEL_PATH


def load_vision_encoder():
    vision_encoder = keras.models.load_model(str(os.path.join(MODEL_PATH, "vision_encoder")))
    return vision_encoder


def load_text_encoder():
    text_encoder = keras.models.load_model(str(os.path.join(MODEL_PATH, "text_encoder")))
    return text_encoder
