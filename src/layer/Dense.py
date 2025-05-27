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

        if input_shape is not None:
            if not isinstance(input_shape, tuple) or len(input_shape) != 1:
                raise ValueError("input_shape must be a tuple of (input units,).")
        self.input_shape = input_shape
        
        if kernel is not None:
            if not isinstance(kernel, np.ndarray):
                raise ValueError("kernel must be a numpy array.")
            if kernel.ndim != 2 or kernel.shape[1] != units or (input_shape and kernel.shape[0] != input_shape[0]):
                raise ValueError(f"kernel must be of shape ({input_shape[0] if input_shape else 'input units'}, {units}).")
        self.kernel = kernel
            
        if bias is not None:
            if not isinstance(bias, np.ndarray):
                raise ValueError("bias must be a numpy array.")
            if bias.ndim != 1 or bias.shape[0] != units:
                raise ValueError(f"bias must be of shape ({units},).")
        self.bias = bias

        if activation is not None and not callable(activation):
            raise ValueError("activation must be a callable function.")
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
        if len(weights) != 2:
            raise ValueError("weights must be a list containing two elements: kernel and bias.")
        
        kernel, bias = weights
        
        if (not isinstance(kernel, np.ndarray)) or (kernel.ndim != 2) or (kernel.shape[1] != self.units) or (self.input_shape and kernel.shape[0] != self.input_shape[0]):
            raise ValueError(f"kernel must be a numpy array of shape ({self.input_shape[0] if self.input_shape else 'input units'}, {self.units}).")
        
        if (not isinstance(bias, np.ndarray)) or (bias.ndim != 1) or (bias.shape[0] != self.units):
            raise ValueError(f"bias must be a numpy array of shape ({self.units},).")
        
        self.kernel = kernel
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
        if not isinstance(input_shape, tuple) or len(input_shape) != 1:
            raise ValueError("input_shape must be a tuple of (input units,).")
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
            if self.input_shape is not None and input_units != self.input_shape[0]:
                raise ValueError(f"Input shape must match the set input shape ({self.input_shape[0]}).")
            return (batch_size, self.units)            
        elif len(input_shape) == 1:
            input_units = input_shape[0]
            if self.input_shape is not None and input_units != self.input_shape[0]:
                raise ValueError(f"Input shape must match the set input shape ({self.input_shape[0]}).")
            return (self.units,)
        else:
            raise ValueError("Input shape must be (height, width, channels) or (batch, height, width, channels).")
    
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
        if self.kernel is None or self.bias is None:
            raise ValueError("Weights must be set before performing forward pass.")
        
        if self.kernel.ndim != 2 or self.kernel.shape[1] != self.units or (self.input_shape and self.kernel.shape[0] != self.input_shape[0]):
            raise ValueError(f"Kernel must be of shape (input units, {self.units}.")
        
        if self.bias.ndim != 1 or self.bias.shape[0] != self.units:
            raise ValueError(f"Bias must be of shape ({self.units},).")
        
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
        # Ensure input shape is set if not provided
        if self.input_shape is None:
            if inputs.ndim == 2:
                self.set_input_shape(inputs.shape[1:])
                print("Input shape set to:", self.input_shape)
            elif inputs.ndim == 1:
                self.set_input_shape(inputs.shape)
            else:
                self.set_input_shape(inputs.shape)

        # Validate weights
        if self.kernel is None or self.bias is None:
            raise ValueError("Kernel and bias must be set before calling the layer.")
        
        # Validate inputs
        if inputs is None:
            raise ValueError("Inputs cannot be None.")
        if not isinstance(inputs, np.ndarray):
            raise ValueError("Inputs must be a numpy ndarray.")
        
        # Do forward pass
        if inputs.ndim == 2:
            # Batch input
            if self.input_shape is not None and inputs.shape[1] != self.input_shape[0]:
                raise ValueError(f"Inputs must be of shape (batch_size, {self.input_shape[0]}).")
            return self.__forward(inputs)
        elif inputs.ndim == 1:
            # Single input
            if self.input_shape is not None and inputs.shape[0] != self.input_shape[0]:
                raise ValueError(f"Input must be of shape ({self.input_shape[0]},).")
            out = self.__forward(inputs[np.newaxis, :])
            return out[0]
        else:
            raise ValueError("Inputs must be of shape (batch_size, input units) or (input units,).")

if __name__ == "__main__":
    # Example usage
    kernel = np.random.rand(5, 10)  # 5 input units, 10 output units
    bias = np.random.rand(10)  # 10 output units
    input_data = np.random.rand(3, 5)  # Batch of 3 samples with 5 features each
    # input_data = np.random.rand(5)     # single sample with 5 features each

    dense_layer = Dense(units=10, input_shape=(5,), activation=np.tanh, kernel=kernel, bias=bias)
    output_data = dense_layer(input_data)
    print("Output data:", output_data)
    print("Output shape:", output_data.shape)
    print("Output shape from compute_output_shape:", dense_layer.compute_output_shape())

    # Get the trainable weights
    trainable_weights = dense_layer.trainable_weights
    print("Trainable Weights:")
    for weight in trainable_weights:
        print(weight.shape)