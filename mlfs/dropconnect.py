import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.utils import to_categorical


class DropConnect(keras.layers.Layer):
