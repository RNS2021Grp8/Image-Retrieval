from collections import OrderedDict
import os
import gc
import pickle
import tensorflow as tf
from deploy.settings import BASE_DIR
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
    top_k = tf.math.top_k(dot_similarity, k)
    values = top_k.values.numpy().tolist()[0]
    indices = top_k.indices.numpy().tolist()[0]
    value_index = dict(zip(values, indices))
    value_index_sorted = OrderedDict(sorted(value_index.items(), reverse=True))
    results = list(value_index_sorted.values())
    return [[image_paths[idx] for idx in results]]

def search(query):
    gc.collect()
    handle = open(os.path.join(BASE_DIR, "output.pkl"), "rb")
    image_embeddings_dict = pickle.load(handle)
    print("Embeddings generated")
    matches = find_matches(image_embeddings_dict, [query], normalize=True)[0]
    return matches

