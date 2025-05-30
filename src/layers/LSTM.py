import numpy as np
import sys
import os

sys.path.append(os.path.abspath("src"))
from utils.ActivationFunction import ActivationFunction

class LSTM():
    def __init__(
            self,
            units,
            activation="tanh",
            recurrent_activation="sigmoid",
            kernel=None,
            recurrent_kernel=None,
            bias=None,
            input_shape=None,
            return_sequences=False,
            **kwargs
        ):
        """
        Initialize the LSTM layer.

        Parameters:
            units (int): Number of output units (dimensionality of the hidden state).
            activation (str): Activation function for the output.
            recurrent_activation (str): Activation function for the recurrent step.
            kernel (np.ndarray, optional): Weights of the layer, should be of shape (input units, units * 4).
            recurrent_kernel (np.ndarray, optional): Recurrent weights, should be of shape (units, units * 4).
            bias (np.ndarray, optional): Bias of the layer, should be of shape (units * 4,).
            input_shape (tuple, optional): Shape of the input data, should be in the form (timesteps, features). Timesteps is sequence length and features is embedding size.
            return_sequences (bool): Whether to return the last output in the output sequence, or the full sequence.
            **kwargs: Additional keyword arguments.

        Notes:
            From keras documentation, the weights are split into [input_gate | forget_gate | cell_gate | output_gate].
        """
        if not isinstance(units, int) or units <= 0:
            raise ValueError("units must be a positive integer.")
        self.units = units

        self.activation = ActivationFunction().activation(activation)
        self.recurrent_activation = ActivationFunction().activation(recurrent_activation)
        self.activation_name = activation
        self.recurrent_activation_name = recurrent_activation
        
        self.set_input_shape(input_shape)
        self.set_weights([kernel, recurrent_kernel, bias])
        self.return_sequences = return_sequences

        self.kwargs = kwargs

    def get_config(self):
        """
        Get the configuration of the LSTM layer.

        Returns:
            dict: Configuration dictionary.
        """
        return {
            'units': self.units,
            'activation': self.activation_name,
            'recurrent_activation': self.recurrent_activation_name,
            'input_shape': self.input_shape,
            'return_sequences': self.return_sequences,
            'kernel': self.kernel,
            'recurrent_kernel': self.recurrent_kernel,
            'bias': self.bias,
            'kwargs': self.kwargs,
        }
    
    def set_weights(self, weights):
        """
        Sets the weights for the LSTM layer.

        Args:
            weights (list): Weights for the LSTM layer, should contain three elements: kernel, recurrent_kernel, and bias.
        """
        if not isinstance(weights, (list, tuple)) or len(weights) != 3:
            raise ValueError("weights must be a list or tuple of [kernel, recurrent_kernel, bias].")
        
        kernel, recurrent_kernel, bias = weights
        if kernel is not None:
            if not isinstance(kernel, np.ndarray):
                raise ValueError("kernel must be a numpy array.")
            if self.input_shape and kernel.shape != (self.input_shape[1], self.units * 4):
                raise ValueError(f"kernel must have shape ({self.input_shape[1]}, {self.units * 4}). Got {kernel.shape}.")
        self.kernel = kernel

        if recurrent_kernel is not None:
            if not isinstance(recurrent_kernel, np.ndarray):
                raise ValueError("recurrent_kernel must be a numpy array.")
            if recurrent_kernel.shape != (self.units, self.units * 4):
                raise ValueError(f"recurrent_kernel must have shape ({self.units}, {self.units * 4}). Got {recurrent_kernel.shape}.")
        self.recurrent_kernel = recurrent_kernel

        if bias is not None:
            if not isinstance(bias, np.ndarray):
                raise ValueError("bias must be a numpy array.")
            if bias.shape != (self.units * 4,):
                raise ValueError(f"bias must have shape ({self.units * 4},). Got {bias.shape}.")
        self.bias = bias

    def get_weights(self):
        """
        Returns the weights of the LSTM layer.

        Returns:
            list: Weights of the LSTM layer.
        """
        return [self.kernel, self.recurrent_kernel, self.bias]
    
    def set_input_shape(self, input_shape):
        """
        Sets the input shape for the LSTM layer.

        Args:
            input_shape (tuple): Shape of the input data, should be in the form (timesteps, features).
        """
        if input_shape is not None:
            if not isinstance(input_shape, tuple) or len(input_shape) != 2:
                raise ValueError("input_shape must be a tuple of (timesteps, features).")
            if not isinstance(input_shape[0], int) or input_shape[0] <= 0:
                raise ValueError("timesteps must be a positive integer.")
            if not isinstance(input_shape[1], int) or input_shape[1] <= 0:
                raise ValueError("features must be a positive integer.")
        self.input_shape = input_shape

    def compute_output_shape(self, input_shape=None):
        """
        Computes the output shape of the LSTM layer.

        Args:
            input_shape (tuple, optional): Shape of the input data, should be in the form (timesteps, features).

        Returns:
            tuple: Output shape of the LSTM layer.
        """
        if self.units is None:
            raise ValueError("Units must be set before computing output shape.")
        
        if input_shape is None:
            if self.input_shape is None:
                raise ValueError("Input shape must be provided or set during initialization.")
            input_shape = self.input_shape

        if len(input_shape) == 3: # handle batch dimension
            batch_size, timesteps, features = input_shape
        elif len(input_shape) == 2: # handle single sample
            timesteps, features = input_shape
        else:
            raise ValueError("Input shape must be a 2D or 3D tensor (timesteps, features) or (batch_size, timesteps, features).")
        
        if self.input_shape is not None and self.input_shape[0] != timesteps:
            raise ValueError(f"Input shape mismatch: expected {self.input_shape[0]} timesteps, got {timesteps}.")
        
        if self.return_sequences:
            return (timesteps, self.units)
        else:
            return (self.units,)
        
    @property
    def trainable_weights(self):
        """
        Returns the trainable weights of the LSTM layer.

        Returns:
            list: A list containing the kernel, recurrent_kernel, and bias.
        """
        if self.kernel is None or self.recurrent_kernel is None or self.bias is None:
            raise ValueError("Weights must be set before accessing trainable weights.")
        return [self.kernel, self.recurrent_kernel, self.bias]

    def __lstm_forward(self, inputs):
        """
        Forward pass of the LSTM layer.

        Args:
            inputs (np.ndarray): Input tensor of shape (batch_size, timesteps, features).

        Returns:
            np.ndarray: Output tensor of shape (batch_size, timesteps, units) if return_sequences is True,
                        or (batch_size, units) if return_sequences is False.
        """
        batch_size, timesteps, features = inputs.shape
        output = np.zeros((batch_size, timesteps, self.units)) if self.return_sequences else np.zeros((batch_size, self.units))

        h_t = np.zeros((batch_size, self.units)) # Hidden state
        c_t = np.zeros((batch_size, self.units)) # Cell state

        for t in range(timesteps):
            # Get the input at time step t
            x_t = inputs[:, t, :]
            z = np.dot(x_t, self.kernel) + np.dot(h_t, self.recurrent_kernel) + self.bias

            # z will have shape = [input_gate | forget_gate | cell_gate | output_gate]
            # Split z into input gate, forget gate, cell gate, and output gate
            # Input Gate: first self.units elements
            i_t = self.recurrent_activation(z[:, :self.units])
            # Forget Gate: second self.units elements
            f_t = self.recurrent_activation(z[:, self.units:self.units * 2])
            # Cell Gate: third self.units elements
            c_tilde_t = self.activation(z[:, self.units * 2:self.units * 3])
            # Output Gate: fourth self.units elements
            o_t = self.recurrent_activation(z[:, self.units * 3:])
            
            c_t = f_t * c_t + i_t * c_tilde_t
            h_t = o_t * self.activation(c_t)
            
            if self.return_sequences:
                output[:, t, :] = h_t

            sys.stdout.write(f"\rdata processed: {t + 1}/{timesteps} timesteps")
            sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()

        
        if not self.return_sequences:
            output = h_t
        
        return output

    def __call__(self, inputs):
        """
        Forward pass of the LSTM layer.

        Args:
            inputs (np.ndarray): Input tensor of shape (batch_size, timesteps, features).

        Returns:
            np.ndarray: Output tensor of shape (batch_size, timesteps, units) if return_sequences is True,
                        or (batch_size, units) if return_sequences is False.
        """
        # Validate inputs
        if inputs is None:
            raise ValueError("Inputs cannot be None.")
        if not isinstance(inputs, np.ndarray):
            raise ValueError("Inputs must be a numpy ndarray.")
        
        # Ensure input shape is set if not provided
        if self.input_shape is None:
            if inputs.ndim == 3:
                self.set_input_shape(inputs.shape[1:])
            else:
                self.set_input_shape(inputs.shape)
            print("Input shape set to:", self.input_shape)

        # Validate weights
        if self.kernel is None or self.recurrent_kernel is None or self.bias is None:
            raise ValueError("Weights must be set before calling the LSTM layer.")
            
        # Do forward pass
        if inputs.ndim == 3: # Batch input
            batch_size, timesteps, features = inputs.shape
            if features != self.input_shape[1]:
                raise ValueError(f"Input features {features} does not match expected {self.input_shape[1]}.")
            output = self.__lstm_forward(inputs)
            return output
        elif inputs.ndim == 2: # Single sample input
            timesteps, features = inputs.shape
            if features != self.input_shape[1]:
                raise ValueError(f"Input features {features} does not match expected {self.input_shape[1]}.")
            output = self.__lstm_forward(inputs[np.newaxis, :, :])
            return output.squeeze(0)
        else:
            raise ValueError("Inputs must be a 2D or 3D tensor.")

        
if __name__ == "__main__":
    # Example usage
    # Layer Parameters
    units = 64
    timesteps = 10
    features = 32
    batch_size = 5

    # Define Weights
    kernel_weights = np.random.rand(features, units * 4)
    recurrent_kernel_weights = np.random.rand(units, units * 4)
    bias_weights = np.random.rand(units * 4)


    # Initialize SimpleRNN layer
    lstm_layer = LSTM(units=units, input_shape=(timesteps, features), return_sequences=True)    
    lstm_layer.set_weights([kernel_weights, recurrent_kernel_weights, bias_weights])
    
    # Prepare Input Data
    input_data_batch = np.random.rand(batch_size, timesteps, features)
    input_data_single = np.random.rand(timesteps, features)

    # Forward Pass
    print("--- Batch Input, Return Sequences ---")
    output_data_batch_seq = lstm_layer(input_data_batch)
    print("Input data shape:", input_data_batch.shape)
    print("Output data shape:", output_data_batch_seq.shape)
    print("Expected output shape (config):", lstm_layer.compute_output_shape())
    # print("Output data (first sample, first timestep):", output_data_batch_seq[0, 0, :5])

    print("\n--- Single Sample Input, Return Sequences ---")
    output_data_single_seq = lstm_layer(input_data_single)
    print("Input data shape:", input_data_single.shape)
    print("Output data shape:", output_data_single_seq.shape) # Should be (timesteps, units)
    print("Expected output shape (config):", lstm_layer.compute_output_shape())
    # print("Output data (first timestep):", output_data_single_seq[0, :5])

    # Example with return_sequences=False
    lstm_layer_last = LSTM(units=units, input_shape=(timesteps, features), return_sequences=False)
    lstm_layer_last.set_weights([kernel_weights, recurrent_kernel_weights, bias_weights])

    print("\n--- Batch Input, Return Last Output ---")
    output_data_batch_last = lstm_layer_last(input_data_batch)
    print("Input data shape:", input_data_batch.shape)
    print("Output data shape:", output_data_batch_last.shape) # Should be (batch_size, units)
    print("Expected output shape (config):", lstm_layer_last.compute_output_shape())
    # print("Output data (first sample):", output_data_batch_last[0, :5])

    print("\n--- Single Sample Input, Return Last Output ---")
    output_data_single_last = lstm_layer_last(input_data_single)
    print("Input data shape:", input_data_single.shape)
    print("Output data shape:", output_data_single_last.shape) # Should be (units,)
    print("Expected output shape (config):", lstm_layer_last.compute_output_shape())
    # print("Output data:", output_data_single_last[:5])