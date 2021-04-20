import os
from tensorflow import keras
from .vars import MODEL_PATH
import tensorflow_text as text


def load_vision_encoder():
    vision_encoder = keras.models.load_model(str(os.path.join(MODEL_PATH, "vision_encoder")), compile=False)
    print("vision encoder successfully loaded")
    return vision_encoder


def load_text_encoder():
    text_encoder = keras.models.load_model(str(os.path.join(MODEL_PATH, "text_encoder")), compile=False)
    print("text encoder successfully loaded")
    return text_encoder
