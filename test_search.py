import tensorflow as tf
from src.retrieval import search

tf.get_logger().setLevel("INFO")

if __name__ == "__main__":
    query = input("Enter Search Query: ")
    search(query)
