import numpy as np
import sys
import os

sys.path.append(os.path.abspath("src"))
from layers.Bidirectional import Bidirectional
from layers.Conv2D import Conv2D
from layers.Dense import Dense
from layers.Embedding import Embedding
from layers.Flatten import Flatten
from layers.GlobalPooling import GlobalPooling, GlobalMaxPooling, GlobalAveragePooling
from layers.LSTM import LSTM
from layers.Pooling import Pooling, MaxPooling, AveragePooling
from layers.SimpleRNN import SimpleRNN

class Model:
    def __init__(self, layers=None):
        """
        Initialize the Model with an optional list of layers.

        Parameters:
        layers (list): List of layer instances to be added to the model.
        """
        self.layers = layers if layers is not None else []
        self.input_shape = None

    def add_layer(self, layer):
        """
        Add a layer to the model.

        Parameters:
        layer: An instance of a layer class.
        """
        if not hasattr(layer, 'set_input_shape'):
            raise ValueError("Layer must implement set_input_shape method.")
        self.layers.append(layer)

    def set_input_shape(self, input_shape):
        """
        Set the input shape for the model.

        Parameters:
        input_shape (tuple): Shape of the input data.
        """
        self.input_shape = input_shape