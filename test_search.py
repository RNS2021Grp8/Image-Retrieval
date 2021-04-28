import tensorflow as tf
from src.retrieval import search
import warnings
warnings.filterwarnings("ignore")

tf.get_logger().setLevel("INFO")

if __name__ == "__main__":
    query = input("Enter Search Query: ")
    search(query)
