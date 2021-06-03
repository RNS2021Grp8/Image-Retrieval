import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from deploy.settings import BASE_DIR
import os
import gc
from .image_processing import generate_embeddings, get_image_paths
from .load_encoders import load_text_encoder


def find_matches(image_embeddings_dict, queries, k=9, normalize=True):
    image_paths = list(image_embeddings_dict.keys())
    image_embeddings = list(image_embeddings_dict.values())
    text_encoder = load_text_encoder()
    query_embedding = text_encoder(tf.convert_to_tensor(queries))
    if normalize:
        image_embeddings = tf.math.l2_normalize(image_embeddings, axis=1)
        query_embedding = tf.math.l2_normalize(query_embedding, axis=1)
    dot_similarity = tf.matmul(query_embedding, image_embeddings, transpose_b=True)
    results = tf.math.top_k(dot_similarity, k).indices.numpy()
    return [[image_paths[idx] for idx in indices] for indices in results]


def search(query):
    #image_embeddings = generate_embeddings()
    gc.collect()
    handle = open(os.path.join(BASE_DIR, "output.pkl"), "rb")
    image_embeddings_dict = pickle.load(handle)
    print("Embeddings generated")
    matches = find_matches(image_embeddings_dict, [query], normalize=True)[0]
    return matches
    """
    print("Matches found. Plotting...")
    plt.figure(figsize=(20, 20))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(mpimg.imread(matches[i]))
        plt.axis("off")
    plt.show()
    print("Plotting complete")
    """
