import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow.keras import layers, models

def main():
    print("Staring MNIST classification")
    
    print("Loading dataset")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()