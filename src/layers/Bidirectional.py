import numpy as np

class Bidirectional():
    def __init__(
            self, 
            layer, 
            merge_mode="concat", 
            weights=None, 
            backward_layer=None,
            input_shape=None,
            **kwargs
        ):
        """
        Initialize the Bidirectional layer.

        Parameters:
            layer: The layer to be wrapped in a bidirectional layer.
            merge_mode (str): Mode for merging the outputs of the forward and backward layers. Options are 'concat', 'sum', 'ave', 'mul'.
            weights (list, optional): Weights for the forward and backward layers.
            backward_layer: Optional layer to be used for the backward pass.
            **kwargs: Additional keyword arguments.
        """
        self.layer = layer

        self.merge_mode = merge_mode.lower()
        if self.merge_mode not in ['concat', 'sum', 'ave', 'mul']:
            raise ValueError("merge_mode must be one of 'concat', 'sum', 'ave', or 'mul'.")

        self.backward_layer = backward_layer if backward_layer else layer
        self.set_input_shape(input_shape)
        self.set_weights(weights)
        self.kwargs = kwargs

    def get_config(self):
        """
        Get the configuration of the Bidirectional layer.

        Returns:
            dict: Configuration dictionary.
        """
        return {
            'layer': self.layer,
            'merge_mode': self.merge_mode,
            'backward_layer': self.backward_layer,
            'weights': self.weights,
            'input_shape': self.input_shape,
            'kwargs': self.kwargs
        }
    
    def set_weights(self, weights):
        """
        Sets the weights for the Bidirectional layer.

        Args:
            weights (list): Weights for the forward and backward layers.
        """
        if weights is None:
            return
        
        if not isinstance(weights, (list, tuple)) or len(weights) != 6:
            raise ValueError("weights must be a list or tuple of [forward_layer_kernel, forward_layer_recurrent_kernel, forward_layer_bias, backward_layer_kernel, backward_layer_recurrent_kernel, backward_layer_bias].")
        
        self.layer.set_weights(weights[:3])
        self.backward_layer.set_weights(weights[3:])

    def get_weights(self):
        """
        Get the weights of the Bidirectional layer.

        Returns:
            list: Weights of the forward and backward layers.
        """
        return self.layer.get_weights() + self.backward_layer.get_weights()
    
    def set_input_shape(self, input_shape):
        """
        Sets the input shape for the Bidirectional layer.
        """
        self.layer.set_input_shape(input_shape)
        self.backward_layer.set_input_shape(input_shape)
        self.input_shape = input_shape

    def compute_output_shape(self, input_shape=None):
        """
        Compute the output shape of the Bidirectional layer.

        Parameters:
            input_shape (tuple, optional): Input shape for the layer. If None, uses the layer's input shape.

        Returns:
            tuple: Output shape of the Bidirectional layer.
        """
        layer_output_shape = self.layer.compute_output_shape(input_shape)
        if self.merge_mode == 'concat':
            layer_output_shape = layer_output_shape[:-1] + (layer_output_shape[-1] * 2,)
        elif self.merge_mode in ['sum', 'ave', 'mul']:
            pass
        else:
            raise ValueError("Unsupported merge mode: {}".format(self.merge_mode))
        return layer_output_shape
    
    @property
    def trainable_weights(self):
        """
        Get the trainable weights of the Bidirectional layer.

        Returns:
            list: Trainable weights of the forward and backward layers.
        """
        return self.layer.trainable_weights + self.backward_layer.trainable_weights

    def __call__(self, inputs):
        """
        Call the Bidirectional layer with the given inputs.

        Parameters:
            inputs: Input data for the layer.

        Returns:
            Output of the Bidirectional layer after processing the inputs.
        """
        forward_output = self.layer(inputs)
        backward_output = self.backward_layer(np.flip(inputs, axis=1))

        if isinstance(backward_output, np.ndarray) and backward_output.ndim == 3:
            backward_output = np.flip(backward_output, axis=1)


        if self.merge_mode == 'concat':
            return np.concatenate([forward_output, backward_output], axis=-1)
        elif self.merge_mode == 'sum':
            return forward_output + backward_output
        elif self.merge_mode == 'ave':
            return (forward_output + backward_output) / 2
        elif self.merge_mode == 'mul':
            return forward_output * backward_output
        else:
            raise ValueError("Unsupported merge mode: {}".format(self.merge_mode))


if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.path.abspath("src"))
    from layers.SimpleRNN import SimpleRNN
    
    units = 64
    timesteps = 10
    features = 32
    batch_size = 5

    kernel_weights = np.random.rand(features, units)
    recurrent_kernel_weights = np.random.rand(units, units)
    bias_weights = np.random.rand(units)

    rnn_layer = SimpleRNN(units=units, input_shape=(timesteps, features), activation='tanh', return_sequences=False)    
    rnn_layer.set_weights([kernel_weights, recurrent_kernel_weights, bias_weights])
    
    input_data_batch = np.random.rand(batch_size, timesteps, features)
    input_data_single = np.random.rand(timesteps, features)

    bidirectional_layer = Bidirectional(rnn_layer, merge_mode='concat')
    
    print("--- Batch Input, Return Sequences ---")
    output_data_batch_seq = bidirectional_layer(input_data_batch)
    print("Input data shape:", input_data_batch.shape)
    print("Output data shape:", output_data_batch_seq.shape)
    print("Expected output shape (config):", bidirectional_layer.compute_output_shape())
    # print("Output data (first sample, first timestep):", output_data_batch_seq[0, 0, :5])

    print("\n--- Single Sample Input, Return Sequences ---")
    output_data_single_seq = bidirectional_layer(input_data_single)
    print("Input data shape:", input_data_single.shape)
    print("Output data shape:", output_data_single_seq.shape) # Should be (timesteps, units)
    print("Expected output shape (config):", bidirectional_layer.compute_output_shape())
    # print("Output data (first timestep):", output_data_single_seq[0, :5])
