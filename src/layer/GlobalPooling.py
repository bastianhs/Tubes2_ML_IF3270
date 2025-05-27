import numpy as np

class GlobalPooling():
    def __init__(self, mode='max', input_shape=None):
        """
        Initialize the GlobalPooling layer.

        Parameters:
        mode (str): The pooling mode, either 'max' or 'average'.
        input_shape (tuple): Shape of the input data, should be in the form (height, width, channels).
        """
        self.mode = mode.lower()
        if self.mode not in ['max', 'average']:
            raise ValueError("mode must be either 'max' or 'average'.")
        
        self.set_input_shape(input_shape)
    
    def get_config(self):
        """
        Get the configuration of the GlobalPooling layer.

        Returns:
        dict: Configuration dictionary containing the mode.
        """
        return {
            'mode': self.mode, 
            'input_shape': self.input_shape
        }
    
    def set_weights(self, weights):
        """
        Sets the weights for the MaxPooling layer.

        Args:
            weights (list): Weights for the MaxPooling layer. This layer does not use weights, so this method does nothing.
        """
        if weights:
            raise ValueError("MaxPooling layer does not have weights to set.")
        
    def get_weights(self):
        """
        Returns the weights of the MaxPooling layer.

        Returns:
            list: An empty list since MaxPooling does not have weights.
        """
        return []

    def set_input_shape(self, input_shape):
        """
        Sets the input shape for the MaxPooling layer.

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
        Computes the output shape of the GlobalPooling layer.

        Parameters:
        input_shape (tuple): Shape of the input tensor. If None, uses the previously set input shape.

        Returns:
        tuple: Output shape of the pooled tensor.
        """
        if input_shape is None:
            if self.input_shape is None:
                raise ValueError("Input shape must be provided or set during initialization.")
            input_shape = self.input_shape
            
        if len(input_shape) == 4: # Handle batch dimension
            _, height, width, channels = input_shape
        elif len(input_shape) == 3:
            height, width, channels = input_shape
        else:
            raise ValueError("Input shape must be (height, width, channels) or (batch, height, width, channels).")
        
        if channels is None or channels <= 0:
            raise ValueError("Number of input channels must be a positive integer.")
            
        return (channels,)

    @property
    def trainable_weights(self):
        """
        Returns the trainable weights of the GlobalPooling layer.

        Returns:
            list: An empty list since GlobalPooling does not have trainable weights.
        """
        return []

    def __pool(self, inputs):
        """
        Perform pooling operation on the inputs.

        Parameters:
        inputs (numpy.ndarray): Input tensor of shape (batch, height, width, channels) or (height, width, channels).

        Returns:
        numpy.ndarray: Pooled output tensor of shape (batch, channels) or (channels,).
        """
        # Handle input shape
        if inputs.ndim == 4:
            # (batch, height, width, channels) -> (batch, channels)
            if self.mode == 'max':
                return np.max(inputs, axis=(1, 2))
            elif self.mode == 'average':
                return np.mean(inputs, axis=(1, 2))
            else:
                raise ValueError("Invalid mode. Use 'max' or 'average'.")
        else:
            # (height, width, channels) -> (channels,)
            if self.mode == 'max':
                return np.max(inputs, axis=(0, 1))
            elif self.mode == 'average':
                return np.mean(inputs, axis=(0, 1))
            else:
                raise ValueError("Invalid mode. Use 'max' or 'average'.")

    def __call__(self, inputs):
        """
        Call the forward method to perform global pooling.

        Parameters:
        inputs (numpy.ndarray): Input tensor of shape (batch_size, channels, height, width).

        Returns:
        numpy.ndarray: Pooled output tensor of shape (batch_size, channels).
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
            
        # Do pooling
        if inputs.ndim in [3, 4]:
            return self.__pool(inputs)
        else:
            raise ValueError("Inputs must have shape (height, width, channels) or (batch, height, width, channels).")
    
class GlobalMaxPooling(GlobalPooling):
    def __init__(self, input_shape=None):
        """
        Initialize the GlobalMaxPooling layer.

        Parameters:
        input_shape (tuple): Shape of the input data, should be in the form (height, width, channels).
        """
        super().__init__(mode='max', input_shape=input_shape)

class GlobalAveragePooling(GlobalPooling):
    def __init__(self, input_shape=None):
        """
        Initialize the GlobalAveragePooling layer.

        Parameters:
        input_shape (tuple): Shape of the input data, should be in the form (height, width, channels).
        """
        super().__init__(mode='average', input_shape=input_shape)

if __name__ == "__main__":
    # Example usage
    inputs = np.random.rand(4, 100, 10)
    
    max_pooling_layer = GlobalMaxPooling()
    avg_output = max_pooling_layer(inputs)
    print("Max pooled output shape:", avg_output.shape)
    print("Max pooled output shape:", max_pooling_layer.compute_output_shape())
    print("Max pooled output:", avg_output)
    print("Trainable parameters:", max_pooling_layer.trainable_weights)
    print()

    avg_pooling_layer = GlobalAveragePooling()
    avg_output = avg_pooling_layer(inputs)
    print("Average pooled output shape:", avg_output.shape)
    print("Average pooled output shape:", avg_pooling_layer.compute_output_shape())
    print("Average pooled output:", avg_output)
    print("Trainable parameters:", avg_pooling_layer.trainable_weights)
    