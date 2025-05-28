import numpy as np
from ..utils.ActivationFunction import ActivationFunction

class SimpleRNN():
    def __init__(
            self,
            units,
            activation="tanh",
            kernel=None,
            recurrent_kernel=None,
            bias=None,
            input_shape=None,
            return_sequences=False,
            **kwargs
        ):
        """
        Initialize the SimpleRNN layer.

        Parameters:
            units (int): Number of output units (dimensionality of the hidden state).
            activation (str or callable): Activation function for the hidden state.
            kernel (np.ndarray, optional): Input-to-hidden weights, shape (input_features, units).
            recurrent_kernel (np.ndarray, optional): Hidden-to-hidden weights, shape (units, units).
            bias (np.ndarray, optional): Bias for the hidden state, shape (units,).
            input_shape (tuple, optional): Shape of the input data (timesteps, features).
            return_sequences (bool): Whether to return the last output or the full sequence.
            **kwargs: Additional keyword arguments.
        """
        if not isinstance(units, int) or units <= 0:
            raise ValueError("units must be a positive integer.")
        self.units = units

        # Store name, useful if config needs string
        # If activation was None or "", default to linear
        self.activation_name = activation 
        if not activation: self.activation_name = "linear"
        self.activation = ActivationFunction().activation(activation) if activation else ActivationFunction().activation("linear")
        self.set_input_shape(input_shape)
        self.set_weights([kernel, recurrent_kernel, bias])
        self.return_sequences = return_sequences

        self.kwargs = kwargs

    def get_config(self):
        """
        Get the configuration of the SimpleRNN layer.

        Returns:
            dict: Configuration dictionary.
        """

        return {
            'units': self.units,
            'activation': self.activation_name,
            'input_shape': self.input_shape,
            'return_sequences': self.return_sequences,
            'weights': [self.kernel, self.recurrent_kernel, self.bias]
        }
    
    def set_weights(self, weights):
        """
        Sets the weights for the SimpleRNN layer.

        Args:
            weights (list): A list containing [kernel, recurrent_kernel, bias].
        """
        if not isinstance(weights, (list, tuple)) or len(weights) != 3:
            raise ValueError("weights must be a list or tuple of [kernel, recurrent_kernel, bias].")
        
        kernel, recurrent_kernel, bias = weights

        # Input shape must be known to validate kernel shape
        if self.input_shape is None and kernel is not None:
            raise ValueError("input_shape must be set before setting weights if kernel is provided, "
                             "or kernel dimensions must not rely on input_shape for now.")

        if kernel is not None:
            if not isinstance(kernel, np.ndarray):
                raise ValueError("kernel must be a numpy array.")
            if self.input_shape and kernel.shape != (self.input_shape[1], self.units):
                raise ValueError(f"kernel must have shape ({self.input_shape[1]}, {self.units}). Got {kernel.shape}")
        self.kernel = kernel

        if recurrent_kernel is not None:
            if not isinstance(recurrent_kernel, np.ndarray):
                raise ValueError("recurrent_kernel must be a numpy array.")
            if recurrent_kernel.shape != (self.units, self.units):
                raise ValueError(f"recurrent_kernel must have shape ({self.units}, {self.units}). Got {recurrent_kernel.shape}")
        self.recurrent_kernel = recurrent_kernel

        if bias is not None:
            if not isinstance(bias, np.ndarray):
                raise ValueError("bias must be a numpy array.")
            if bias.shape != (self.units,):
                raise ValueError(f"bias must have shape ({self.units},). Got {bias.shape}")
        self.bias = bias

    def get_weights(self):
        """
        Returns the weights of the SimpleRNN layer.
        """
        return [self.kernel, self.recurrent_kernel, self.bias]
    
    def set_input_shape(self, input_shape):
        """
        Sets the input shape for the SimpleRNN layer.
        """
        if input_shape is not None:
            if not isinstance(input_shape, tuple) or len(input_shape) != 2:
                raise ValueError("input_shape must be a tuple of (timesteps, features).")
            if not (isinstance(input_shape[0], int) or input_shape[0] is None) or (isinstance(input_shape[0], int) and input_shape[0] <= 0) : # timesteps can be None
                raise ValueError("timesteps must be a positive integer or None.")
            if not isinstance(input_shape[1], int) or input_shape[1] <= 0:
                raise ValueError("features must be a positive integer.")
        self.input_shape = input_shape

    def compute_output_shape(self, input_shape=None):
        """
        Computes the output shape of the SimpleRNN layer.
        """
        if self.units is None:
            raise ValueError("Units must be set before computing output shape.")
        
        current_input_shape = input_shape if input_shape is not None else self.input_shape
        if current_input_shape is None:
            raise ValueError("Input shape must be provided or set during initialization.")

        if len(current_input_shape) == 3: # (batch_size, timesteps, features)
            _batch_size, timesteps, _features = current_input_shape
        elif len(current_input_shape) == 2: # (timesteps, features)
            timesteps, _features = current_input_shape
        else:
            raise ValueError("Input shape must be a 2D or 3D tensor (timesteps, features) or (batch_size, timesteps, features).")
        
        # If self.input_shape[0] (timesteps) is None, it means variable length, so we use timesteps from input_shape
        if self.input_shape and self.input_shape[0] is not None and self.input_shape[0] != timesteps:
             # Keras allows input_shape's timesteps to be None, then actual timesteps determined at call time.
            # If specific timesteps set at init, then they must match unless init timesteps was None.
            pass # Allow flexible timesteps if self.input_shape[0] was None


        if self.return_sequences:
            if len(current_input_shape) == 3: # batch mode
                return (current_input_shape[0], timesteps, self.units)
            return (timesteps, self.units)
        else:
            if len(current_input_shape) == 3: # batch mode
                return (current_input_shape[0], self.units)
            return (self.units,)
        
    @property
    def trainable_weights(self):
        """
        Returns the trainable weights of the SimpleRNN layer.
        """
        if self.kernel is None or self.recurrent_kernel is None or self.bias is None:
            # This check ensures weights are initialized, e.g. by build() or set_weights()
            # In a full framework, a build() method would create weights if not provided.
            raise ValueError("Weights must be set/initialized before accessing trainable weights.")
        return [self.kernel, self.recurrent_kernel, self.bias]

    def __rnn_forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass of the SimpleRNN layer.

        Args:
            inputs (np.ndarray): Input tensor of shape (batch_size, timesteps, features).

        Returns:
            np.ndarray: Output tensor. Shape is (batch_size, timesteps, units) if 
                        return_sequences is True, or (batch_size, units) otherwise.
        """
        batch_size, timesteps, _features = inputs.shape
        
        # Initialize hidden state (h_prev)
        h_prev = np.zeros((batch_size, self.units))
        
        if self.return_sequences:
            # Store all hidden states if return_sequences is True
            output_sequences = np.zeros((batch_size, timesteps, self.units))

        for t in range(timesteps):
            x_t = inputs[:, t, :]  # Current input: (batch_size, features)
            
            # SimpleRNN core calculation: h_t = activation(x_t @ W_xh + h_{t-1} @ W_hh + b_h)
            z = np.dot(x_t, self.kernel) + np.dot(h_prev, self.recurrent_kernel) + self.bias
            h_t = self.activation(z)  # Current hidden state: (batch_size, units)
            
            if self.return_sequences:
                output_sequences[:, t, :] = h_t
            
            h_prev = h_t  # Update hidden state for the next_lstm_forward timestep

        if self.return_sequences:
            return output_sequences
        else:
            # Return the last hidden state if return_sequences is False
            return h_t

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass of the SimpleRNN layer (callable interface).

        Args:
            inputs (np.ndarray): Input tensor. Can be 2D (timesteps, features) for a single
                                 sample or 3D (batch_size, timesteps, features) for a batch.

        Returns:
            np.ndarray: Output tensor from the RNN.
        """
        if inputs is None:
            raise ValueError("Inputs cannot be None.")
        if not isinstance(inputs, np.ndarray):
            raise ValueError("Inputs must be a numpy ndarray.")
        
        # Determine actual input shape at call time
        current_input_dim = inputs.ndim
        current_input_shape = inputs.shape

        # Set or validate input_shape
        if self.input_shape is None:
            if current_input_dim == 2: # (timesteps, features)
                self.set_input_shape(current_input_shape)
            elif current_input_dim == 3: # (batch_size, timesteps, features)
                self.set_input_shape(current_input_shape[1:]) # Store (timesteps, features)
            else:
                raise ValueError("Input must be 2D or 3D. Got {}D".format(current_input_dim))
            # print(f"Input shape dynamically set to: {self.input_shape}")

        # Validate weights (kernel, recurrent_kernel, bias)
        if self.kernel is None or self.recurrent_kernel is None or self.bias is None:
            # In a more complete framework, a "build" step would initialize weights here
            # if they are None, based on self.input_shape.
            # For now, we require them to be set.
            raise ValueError("Weights (kernel, recurrent_kernel, bias) must be set before calling the layer.")

        # Reshape inputs if 2D (single sample) to 3D (batch of 1)
        processed_inputs = inputs
        if current_input_dim == 2: # (timesteps, features)
            # Add batch dimension: (1, timesteps, features)
            processed_inputs = inputs[np.newaxis, :, :]
            # Update current_input_shape to reflect batch dimension for internal processing
            # current_input_shape = processed_inputs.shape 
        elif current_input_dim != 3: # Not 2D or 3D
             raise ValueError("Inputs must be a 2D or 3D tensor.")

        # Feature dimension check
        # current_input_shape[0] is batch_size, current_input_shape[1] is timesteps, current_input_shape[2] is features for 3D
        # current_input_shape[0] is timesteps, current_input_shape[1] is features for 2D
        num_features = processed_inputs.shape[2]
        if num_features != self.input_shape[1]:
            raise ValueError(f"Input features {num_features} does not match expected {self.input_shape[1]}.")

        # Perform forward pass
        output = self.__rnn_forward(processed_inputs)
        
        # If input was 2D, remove the batch dimension from output before returning
        if current_input_dim == 2 and not self.return_sequences:
            output = output[0] # (units,)
        elif current_input_dim == 2 and self.return_sequences:
            output = output[0, :, :] # (timesteps, units)
            
        return output

if __name__ == "__main__":
    # Example Usage for SimpleRNN

    # 1. Define ActivationFunction instance (if not using the global one from above)
    # af_instance = ActivationFunction() (not needed if class is defined globally)

    # 2. Layer Parameters
    units = 64
    timesteps = 10
    features = 32
    batch_size = 5

    # 3. Initialize SimpleRNN layer
    # Note: `input_shape` can be set here or inferred at first call if weights are built later.
    # For this example, we provide it for weight validation.
    rnn_layer = SimpleRNN(units=units, input_shape=(timesteps, features), activation='tanh', return_sequences=True)
    # rnn_layer_last_output = SimpleRNN(units=units, input_shape=(timesteps, features), activation='relu', return_sequences=False)

    # 4. Define Weights (manually for this example)
    # kernel: (features, units)
    kernel_weights = np.random.rand(features, units)
    # recurrent_kernel: (units, units)
    recurrent_kernel_weights = np.random.rand(units, units)
    # bias: (units,)
    bias_weights = np.random.rand(units)
    
    rnn_layer.set_weights([kernel_weights, recurrent_kernel_weights, bias_weights])
    # rnn_layer_last_output.set_weights([
    #     np.random.rand(features, units),
    #     np.random.rand(units, units),
    #     np.random.rand(units)
    # ])

    # 5. Prepare Input Data
    # Batch input: (batch_size, timesteps, features)
    input_data_batch = np.random.rand(batch_size, timesteps, features)
    # Single sample input: (timesteps, features)
    input_data_single = np.random.rand(timesteps, features)

    # 6. Forward Pass
    print("--- Batch Input, Return Sequences ---")
    output_data_batch_seq = rnn_layer(input_data_batch)
    print("Input data shape:", input_data_batch.shape)
    print("Output data shape:", output_data_batch_seq.shape)
    print("Expected output shape (config):", rnn_layer.compute_output_shape(input_data_batch.shape))
    print("Output data (first sample, first timestep):", output_data_batch_seq[0, 0, :5])

    print("\n--- Single Sample Input, Return Sequences ---")
    output_data_single_seq = rnn_layer(input_data_single)
    print("Input data shape:", input_data_single.shape)
    print("Output data shape:", output_data_single_seq.shape) # Should be (timesteps, units)
    print("Expected output shape (config):", rnn_layer.compute_output_shape(input_data_single.shape))
    print("Output data (first timestep):", output_data_single_seq[0, :5])

    print("Output data single: \n", output_data_single_seq)
    print("Output data batch: \n", output_data_batch_seq)


    # Example with return_sequences=False
    rnn_layer_last = SimpleRNN(units=units, input_shape=(timesteps, features), activation='tanh', return_sequences=False)
    rnn_layer_last.set_weights([kernel_weights, recurrent_kernel_weights, bias_weights])

    print("\n--- Batch Input, Return Last Output ---")
    output_data_batch_last = rnn_layer_last(input_data_batch)
    print("Input data shape:", input_data_batch.shape)
    print("Output data shape:", output_data_batch_last.shape) # Should be (batch_size, units)
    print("Expected output shape (config):", rnn_layer_last.compute_output_shape(input_data_batch.shape))
    print("Output data (first sample):", output_data_batch_last[0, :5])

    print("\n--- Single Sample Input, Return Last Output ---")
    output_data_single_last = rnn_layer_last(input_data_single)
    print("Input data shape:", input_data_single.shape)
    print("Output data shape:", output_data_single_last.shape) # Should be (units,)
    print("Expected output shape (config):", rnn_layer_last.compute_output_shape(input_data_single.shape))
    print("Output data:", output_data_single_last[:5])

    # 7. Get Layer Configuration
    # config = rnn_layer.get_config()
    # print("\nLayer Config:", config)
    # Note: If activation in config is a function object, it will print something like <function ActivationFunction.__tanh at 0x...>.
    # If it's self.activation_name, it will print "tanh". I've set it to return name.
