import numpy as np

class Flatten():
    def __init__(self, input_shape=None, **kwargs):
        """
        Initialize the Flatten layer.

        Parameters:
        input_shape (tuple): Shape of the input data, should be in the form (height, width, channels).
        """
        self.set_input_shape(input_shape)
        self.kwargs = kwargs
    
    def get_config(self):
        """
        Get the configuration of the Flatten layer.

        Returns:
            dict: Configuration dictionary.
        """
        return {
            'input_shape': self.input_shape,
            'kwargs': self.kwargs
        }
    
    def set_weights(self, weights):
        """
        Sets the weights for the Flatten layer.

        Args:
            weights (list): Weights for the Flatten layer.
        """
        if weights:
            raise ValueError("Flatten layer does not have weights to set.")
    
    def get_weights(self):
        """
        Returns the weights of the Flatten layer.

        Returns:
            list: An empty list since Flatten does not have weights.
        """
        return []
    
    def set_input_shape(self, input_shape):
        """
        Sets the input shape for the Flatten layer.

        Args:
            input_shape (tuple): Shape of the input data. Should be in the form (height, width, channels).
        """
        if input_shape is not None:
            if not isinstance(input_shape, tuple) or len(input_shape) != 3:
                raise ValueError("input_shape must be a tuple of (height, width, channels).")
            if not all(isinstance(dim, int) and dim > 0 for dim in input_shape):
                raise ValueError("input_shape dimensions must be positive integers.")
        self.input_shape = input_shape

    def compute_output_shape(self, input_shape=None):
        """
        Computes the output shape of the Flatten layer.

        Parameters:
        input_shape (tuple): Shape of the input tensor. If None, uses the previously set input shape.

        Returns:
        tuple: Output shape of the flattened tensor.
        """
        if input_shape is None:
            if self.input_shape is None:
                raise ValueError("Input shape must be provided or set during initialization.")
            input_shape = self.input_shape
        
        if len(input_shape) == 4:
            batch, height, width, channels = input_shape
        elif len(input_shape) == 3:
            height, width, channels = input_shape
        else:
            raise ValueError("Input shape must be a 3D or 4D tensor.")
        
        return (height * width * channels,)

    @property
    def trainable_weights(self):
        """
        Returns the trainable weights of the Flatten layer.

        Returns:
            list: An empty list since Flatten does not have trainable weights.
        """
        return []

    def __flatten(self, inputs):
        """
        Flattens the input tensor.

        Args:
            inputs (numpy.ndarray): Input tensor of shape (batch_size, height, width, channels) or (height, width, channels).

        Returns:
            numpy.ndarray: Flattened tensor of shape (batch_size, height * width * channels) or (height * width * channels,).
        """
        if inputs.ndim == 4:
            batch_size = inputs.shape[0]
            return inputs.reshape(batch_size, -1)
        elif inputs.ndim == 3:
            return inputs.reshape(-1)
        else:
            raise ValueError("Input must be a 3D or 4D tensor.")

    def __call__(self, inputs):
        """
        Applies the Flatten layer to the input data.

        Args:
            inputs (numpy.ndarray): Input tensor of shape (batch_size, height, width, channels) or (height, width, channels).

        Returns:
            numpy.ndarray: Flattened tensor.
        """
        # Validate inputs
        if inputs is None:
            raise ValueError("Inputs cannot be None.")
        if not isinstance(inputs, np.ndarray):
            raise ValueError("Inputs must be a numpy ndarray.")
        
        # Ensure input shape is set if not provided
        if self.input_shape is None:
            if inputs.ndim == 4:
                self.set_input_shape(inputs.shape[1:])
            else:
                self.set_input_shape(inputs.shape)
            print("Input shape set to:", self.input_shape)
        else:
            if inputs.ndim == 4 and inputs.shape[1:] != self.input_shape:
                raise ValueError(f"Input shape {inputs.shape[1:]} does not match the set input shape {self.input_shape}.")
            elif inputs.ndim == 3 and inputs.shape != self.input_shape:
                raise ValueError(f"Input shape {inputs.shape} does not match the set input shape {self.input_shape}.")

        # Flatten the inputs
        return self.__flatten(inputs)
    
if __name__ == "__main__":
    # Example usage
    flatten_layer = Flatten(input_shape=(28, 29, 1))
    input_data = np.random.rand(10, 28, 29, 1)
    output_data = flatten_layer(input_data)
    print("Flattened output:", output_data)
    print("Input shape:", input_data.shape)
    print("Output shape:", output_data.shape)
    print("Output shape:", flatten_layer.compute_output_shape())
    print("Trainable parameters:", flatten_layer.trainable_weights)