import numpy as np
import sys
import os
from tensorflow.keras.models import load_model

sys.path.append(os.path.abspath("src"))

from layers.Bidirectional import Bidirectional
from layers.Conv2D import Conv2D
from layers.Dense import Dense
from layers.Embedding import Embedding
from layers.Flatten import Flatten
from layers.GlobalPooling import GlobalMaxPooling, GlobalAveragePooling
from layers.LSTM import LSTM
from layers.Pooling import MaxPooling, AveragePooling
from layers.SimpleRNN import SimpleRNN

class Model:
    def __init__(self, layers=None, input_shape=None, **kwargs):
        """
        Initialize the Model with an optional list of layers.

        Parameters:
        layers (list): List of layer instances to be added to the model.
        """
        self.layers = layers if layers is not None else []
        self.input_shape = None
        self.kwargs = kwargs

    def add_layer(self, layer):
        """
        Add a layer to the model.

        Parameters:
        layer: An instance of a layer class.
        """
        if not hasattr(layer, 'set_input_shape'):
            raise ValueError("Layer must implement set_input_shape method.")
        self.layers.append(layer)

    def get_layers(self):
        """
        Get the list of layers in the model.

        Returns:
        list: List of layer instances.
        """
        return self.layers
    
    def get_layer(self, name=None, index=None):
        """
        Get a specific layer by name or index.

        Parameters:
        name (str): Name of the layer.
        index (int): Index of the layer.

        Returns:
        Layer instance: The requested layer.
        """
        if name is not None:
            for layer in self.layers:
                if layer.name == name:
                    return layer
            raise ValueError(f"Layer with name '{name}' not found.")
        elif index is not None:
            if 0 <= index < len(self.layers):
                return self.layers[index]
            else:
                raise IndexError("Layer index out of range.")
        else:
            raise ValueError("Either name or index must be provided.")

    def set_input_shape(self, input_shape):
        """
        Set the input shape for the model.

        Parameters:
        input_shape (tuple): Shape of the input data.
        """
        self.input_shape = input_shape
        if self.layers:
            self.layers[0].set_input_shape(input_shape)

            for i in range(1, len(self.layers)):
                try:
                    output_shape_prev = self.layers[i - 1].compute_output_shape(input_shape=self.layers[i - 1].input_shape)
                except Exception:
                    output_shape_prev = None

                self.layers[i].set_input_shape(output_shape_prev)

    def get_config(self):
        """
        Get the configuration of the model.

        Returns:
        dict: Configuration dictionary containing layer configurations.
        """
        config = {
            'input_shape': self.input_shape,
            'layers': [layer.get_config() for layer in self.layers],
            'kwargs': self.kwargs
        }
        return config
    
    def predict(self, x):
        """
        Make predictions using the model.

        Parameters:
        x (np.ndarray): Input data.

        Returns:
        np.ndarray: Output predictions.
        """
        i = 1
        for layer in self.layers:
            print(f"Processing layer {i}: {layer.__class__.__name__}")
            x = layer(x)
            i += 1
        return x
    
    def summary(self):
        """
        Print a summary of the model architecture.
        """
        print("Model Summary".center(80, "="))
        print(f"{'Layer (type)':<30} {'Output Shape':<25} {'Param #':>10}")
        print("=" * 80)

        total_params = 0
        input_shape = self.input_shape if self.input_shape is not None else None
        for i, layer in enumerate(self.layers):
            layer_name = layer.__class__.__name__

            # Output shape
            try:
                output_shape = layer.compute_output_shape(input_shape=input_shape)
            except Exception:
                output_shape = "?"

            # Parameter count
            param_count = 0
            for weight in getattr(layer, "trainable_weights", []):
                param_count += int(np.prod(weight.shape))

            total_params += param_count

            print(f"{layer_name:<30} {str(output_shape):<25} {param_count:>10,}")

            # Update input shape for the next layer
            input_shape = output_shape if isinstance(output_shape, tuple) else None

        print("=" * 80)
        print(f"{'Total params:':<56} {total_params:>10,}")
        print("=" * 80)

    def load(self, model_path):
        """
        Load a model from a file.

        Parameters:
        model_path (str): Path to the model file.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' does not exist.")
        
        loaded_model = load_model(model_path)
        self.layers = []
        
        for layer in loaded_model.layers:
            layer_class = layer.__class__.__name__
            layer_config = layer.get_config()
            layer_weights = layer.get_weights()

            if layer_class == 'Bidirectional':
                # get the forward and backward layer configurations
                forward_config = layer_config['layer']['config']
                merge_mode = layer_config['merge_mode']

                if 'backward_layer' in layer_config:
                    backward_config = layer_config['backward_layer']['config']
                else:
                    backward_config = forward_config.copy()
                
                # create the forward
                if layer_config['layer']['class_name'] == 'LSTM':
                    forward_layer = LSTM(**forward_config)
                else:
                    forward_layer = SimpleRNN(**forward_config)

                # create the backward layer
                if layer_config['backward_layer']['class_name'] == 'LSTM':
                    backward_layer = LSTM(**backward_config)
                else:
                    backward_layer = SimpleRNN(**backward_config)

                # create the Bidirectional layer
                my_layer = Bidirectional(
                    layer=forward_layer,
                    backward_layer=backward_layer,
                    merge_mode=merge_mode
                )
            elif layer_class == 'Conv2D':
                my_layer = Conv2D(**layer_config)
            elif layer_class == 'Dense':
                my_layer = Dense(**layer_config)
            elif layer_class == 'Embedding':
                my_layer = Embedding(**layer_config)
            elif layer_class == 'Flatten':
                my_layer = Flatten(**layer_config)
            elif layer_class == 'GlobalMaxPooling2D':
                my_layer = GlobalMaxPooling(**layer_config)
            elif layer_class == 'GlobalAveragePooling2D':
                my_layer = GlobalAveragePooling(**layer_config)
            elif layer_class == 'LSTM':
                my_layer = LSTM(**layer_config)
            elif layer_class == 'MaxPooling2D':
                my_layer = MaxPooling(**layer_config)
            elif layer_class == 'AveragePooling2D':
                my_layer = AveragePooling(**layer_config)
            elif layer_class == 'SimpleRNN':
                my_layer = SimpleRNN(**layer_config)
            else:
                continue
            
            my_layer.set_weights(layer_weights)
            self.layers.append(my_layer)

        if self.input_shape is not None:
            self.set_input_shape(self.input_shape)

if __name__ == "__main__":
    # Example usage
    model = Model()
    # model.set_input_shape((100,))  # Example input shape for an image
    model.set_input_shape((32, 32, 3))  # Example input shape for an image
    # model.load("cnn_model2.h5")
    model.summary()
