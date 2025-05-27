import numpy as np

class Dense():
    def __init__(
            self,
            units,
            input_shape=None,
            kernel=None,
            bias=None,
            activation=None,
            **kwargs
        ):
        """
        Initialize the Dense layer.

        Parameters:
            units (int): Number of output units.
            input_shape (tuple, optional): Shape of the input data, should be in the form (input units,).
            kernel (np.ndarray, optional): Weights of the layer, should be of shape (input units, units).
            bias (np.ndarray, optional): Bias of the layer, should be of shape (units,).
            activation (callable, optional): Activation function to apply to the output.
            **kwargs: Additional keyword arguments.
        """
        if not isinstance(units, int) or units <= 0:
            raise ValueError("units must be a positive integer.")
        self.units = units

        self.set_input_shape(input_shape)
        self.set_weights([kernel, bias])

        if activation is not None and not callable(activation):
            raise ValueError("activation must be a callable function or None.")
        self.activation = activation
        
    def get_config(self):
        """
        Get the configuration of the Dense layer.
        Returns:
            dict: Configuration dictionary containing the units, input shape, kernel, bias, and activation.
        """
        return {
            'units': self.units,
            'input_shape': self.input_shape,
            'kernel': self.kernel,
            'bias': self.bias,
            'activation': self.activation
        }
    
    def set_weights(self, weights):
        """
        Sets the weights for the Dense layer.

        Args:
            weights (list): Weights for the Dense layer, should contain two elements: kernel and bias.
        """
        if not isinstance(weights, (list, tuple)) or len(weights) != 2:
            raise ValueError("weights must be a list or tuple of [kernel, bias].")
        
        kernel, bias = weights
        if kernel is not None:
            if not isinstance(kernel, np.ndarray):
                raise ValueError("kernel must be a numpy ndarray.")
            if kernel.ndim != 2:
                raise ValueError("kernel must be a 2D numpy ndarray.")
            if kernel.shape[1] != self.units or (self.input_shape and kernel.shape[0] != self.input_shape[0]):
                raise ValueError(f"kernel must be of shape ({self.input_shape[0] if self.input_shape else 'input units'}, {self.units}).")
        else:
            kernel = np.random.rand(self.input_shape[0], self.units) if self.input_shape else None
        self.kernel = kernel

        if bias is not None:
            if not isinstance(bias, np.ndarray):
                raise ValueError("bias must be a numpy ndarray.")
            if bias.ndim != 1:
                raise ValueError(f"bias shape must be ({self.units},).")
            if bias.shape[0] != self.units:
                raise ValueError(f"Bias shape {bias.shape} does not match units ({self.units},).")
        else:
            bias = np.random.rand(self.units) if self.units > 0 else None
        self.bias = bias        

    def get_weights(self):
        """
        Returns the weights of the Dense layer.

        Returns:
            list: A list containing the kernel and bias.
        """
        return [self.kernel, self.bias]
    
    def set_input_shape(self, input_shape):
        """
        Sets the input shape for the Dense layer.

        Args:
            input_shape (tuple): Shape of the input data, should be in the form (input units,).
        """
        if input_shape is not None:
            if not isinstance(input_shape, tuple) or len(input_shape) != 1:
                raise ValueError("input_shape must be a tuple of (input units,).")
            if not all(isinstance(dim, int) and dim > 0 for dim in input_shape):
                raise ValueError("input_shape dimensions must be positive integers.")
        self.input_shape = input_shape

    def compute_output_shape(self, input_shape=None):
        """
        Computes the output shape of the Dense layer.

        Parameters:
            input_shape (tuple): Shape of the input tensor. If None, uses the previously set input shape.

        Returns:
            tuple: Output shape of the Dense layer, which is (batch_size, units) jika batch, atau (units,) jika single sample.
        """
        if input_shape is None:
            if self.input_shape is None:
                raise ValueError("Input shape must be provided or set during initialization.")
            input_shape = self.input_shape

        if len(input_shape) == 2: # Handle batch dimension
            batch_size, input_units = input_shape
        elif len(input_shape) == 1:
            input_units = input_shape[0]
        else:
            raise ValueError("Input shape must be (height, width, channels) or (batch, height, width, channels).")
        
        if self.input_shape is not None and input_units != self.input_shape[0]:
            raise ValueError(f"Input shape must match the set input shape ({self.input_shape[0]}).")

        return (self.units,)
    
    @property
    def trainable_weights(self):
        """
        Returns the trainable weights of the Dense layer.

        Returns:
            list: A list containing the kernel and bias.
        """
        if self.kernel is None or self.bias is None:
            raise ValueError("Weights must be set before accessing trainable weights.")
        return [self.kernel, self.bias]
    
    def __forward(self, inputs):
        """
        Forward pass of the Dense layer.

        Args:
            inputs (np.ndarray): Input tensor of shape (batch_size, input units).

        Returns:
            np.ndarray: Output tensor of shape (batch_size, units).
        """
        # Validate Kernel and Bias
        if self.kernel.shape[0] != inputs.shape[1] or self.kernel.shape[1] != self.units:
            raise ValueError(f"Kernel shape {self.kernel.shape} does not match input shape {inputs.shape[1]}.")
        if self.bias.shape[0] != self.units:
            raise ValueError(f"Bias shape {self.bias.shape} does not match units ({self.units},).")
        
        # forward pass
        output = np.dot(inputs, self.kernel) + self.bias
        
        if self.activation is not None:
            output = self.activation(output)
        
        return output
    
    def __call__(self, inputs):
        """
        Calls the Dense layer with the given inputs.

        Args:
            inputs (np.ndarray): Input tensor of shape (batch_size, input units) atau (input units,).

        Returns:
            np.ndarray: Output tensor of shape (batch_size, units) atau (units,).
        """
        # Validate inputs
        if inputs is None:
            raise ValueError("Inputs cannot be None.")
        if not isinstance(inputs, np.ndarray):
            raise ValueError("Inputs must be a numpy ndarray.")
        
        # Ensure input shape is set if not provided
        if self.input_shape is None:
            if inputs.ndim == 2:
                self.set_input_shape(inputs.shape[1:])
            else:
                self.set_input_shape(inputs.shape)
            print("Input shape set to:", self.input_shape)

        # Validate weights
        if self.kernel is None or self.bias is None:
            raise ValueError("Kernel and bias must be set before calling the layer.")
        
        # Do forward pass
        if inputs.ndim == 2: # Batch input
            if self.input_shape is not None and inputs.shape[1] != self.input_shape[0]:
                raise ValueError(f"Input shape {inputs.shape[1]} does not match expected input shape {self.input_shape[0]}.")
            return self.__forward(inputs)
        elif inputs.ndim == 1: # Single input
            if self.input_shape is not None and inputs.shape[0] != self.input_shape[0]:
                raise ValueError(f"Input shape {inputs.shape[0]} does not match expected input shape {self.input_shape[0]}.")
            out = self.__forward(inputs[np.newaxis, :])
            return out[0]
        else:
            raise ValueError("Inputs must be of shape (batch_size, input units) or (input units,).")

if __name__ == "__main__":
    # Example usage
    kernel = np.random.rand(5, 10)
    bias = np.random.rand(10)
    input_data = np.random.rand(3, 5)
    # input_data = np.random.rand(5)

    dense_layer = Dense(units=10, activation=np.tanh, kernel=kernel, bias=bias)
    output_data = dense_layer(input_data)
    print("Output data:", output_data)
    print("Output shape:", output_data.shape)
    print("Output shape from compute_output_shape:", dense_layer.compute_output_shape())

    trainable_weights = dense_layer.trainable_weights
    print("Trainable Weights:")
    for weight in trainable_weights:
        print("Weight shape:", weight.shape)