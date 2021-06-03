import os
import pickle
import tensorflow as tf
from os import walk
from collections import OrderedDict
from .vars import BASE_DIR, BATCH_SIZE, DATA_DIR
from .load_encoders import load_vision_encoder



def get_image_paths():
    images_dir = os.path.join(DATA_DIR, "images")
    image_paths = []
    for _, _, files in walk(images_dir):
        for filename in files:
            image_path = os.path.join(images_dir, filename)
            image_paths.append(image_path)
    return image_paths


def read_image(image_path):
    image_array = tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3)
    return tf.image.resize(image_array, (299, 299))


def generate_embeddings():
    image_paths = get_image_paths()
    print(f"Generating embeddings for {len(image_paths)} images...")
    vision_encoder = load_vision_encoder()
    image_embeddings = vision_encoder.predict(
        tf.data.Dataset.from_tensor_slices(image_paths).map(read_image).batch(BATCH_SIZE),
        verbose=1,
    )
    print(f"Image embeddings shape: {image_embeddings.shape}.")

    image_embeddings_dict = OrderedDict(zip(image_paths, image_embeddings))
    output_file = open(os.path.join(BASE_DIR, "output.pkl"), "wb")
    pickle.dump(image_embeddings_dict, output_file)