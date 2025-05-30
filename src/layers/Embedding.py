import numpy as np

class Embedding():
    def __init__(
            self,
            input_dim,
            output_dim,
            weights=None,
            input_shape=None,
            **kwargs
        ):
        """
        Initialize the Embedding layer.

        Parameters:
            input_dim (int): Size of the input vocabulary.
            output_dim (int): Dimension of the embedding vectors.
            weights (np.ndarray, optional): weights for the embedding layer, should be of shape (input_dim, output_dim).
            input_shape (tuple, optional): Shape of the input data, should be in the form (sequence_length,).
            **kwargs: Additional keyword arguments.
        """
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer.")
        if not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError("output_dim must be a positive integer.")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.set_input_shape(input_shape)
        self.set_weights(weights)

        self.kwargs = kwargs

    def get_config(self):
        """
        Get the configuration of the Embedding layer.

        Returns:
            dict: Configuration dictionary.
        """
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'input_shape': self.input_shape,
            'weights': self.weights,
            'kwargs': self.kwargs
        }

    def set_weights(self, weights):
        """
        Sets the weights for the Embedding layer.

        Args:
            weights (np.ndarray): Weights for the Embedding layer, should be of shape (input_dim, output_dim).
        """
        weights = np.array(weights) if weights is not None else None
        if weights is not None:
            if weights.ndim == 3:
                weights = weights[0]

            if not isinstance(weights, np.ndarray):
                raise ValueError("weights must be a numpy array.")
            if weights.shape != (self.input_dim, self.output_dim):
                raise ValueError(f"weights must have shape ({self.input_dim}, {self.output_dim}).")
        self.weights = weights

    def get_weights(self):
        """
        Returns the weights of the Embedding layer.

        Returns:
            np.ndarray: Weights of the Embedding layer.
        """
        return self.weights
    
    def set_input_shape(self, input_shape):
        """
        Sets the input shape for the Embedding layer.

        Args:
            input_shape (tuple): Shape of the input data. Should be in the form (sequence_length,).
        """
        if input_shape is not None:
            if not isinstance(input_shape, tuple) or len(input_shape) != 1:
                raise ValueError("input_shape must be a tuple of (sequence_length,).")
            if not isinstance(input_shape[0], int) or input_shape[0] <= 0:
                raise ValueError("sequence_length must be a positive integer.")
        self.input_shape = input_shape

    def compute_output_shape(self, input_shape=None):
        """
        Computes the output shape of the Embedding layer.

        Parameters:
            input_shape (tuple): Shape of the input tensor. If None, uses the previously set input shape.

        Returns:
            tuple: Output shape of the embedding tensor.
        """
        if input_shape is None:
            if self.input_shape is None:
                raise ValueError("Input shape must be set before computing output shape.")
            input_shape = self.input_shape

        if len(input_shape) == 2: # Handle batch dimension
            _, sequence_length = input_shape
        elif len(input_shape) == 1:
            sequence_length = input_shape[0]
        else:
            raise ValueError("Input shape must be a 1D or 2D tensor.")
        
        return (sequence_length, self.output_dim)        

    @property
    def trainable_weights(self):
        """
        Returns the trainable weights of the Embedding layer.

        Returns:
            list: A list containing the weights of the Embedding layer.
        """
        if self.weights is None:
            raise ValueError("Weights must be set before accessing trainable weights.")
        return [self.weights]

    def __embedding_lookup(self, inputs):
        """
        Perform embedding lookup for the input indices.
        Assumes inputs is always 2D: (batch_size, sequence_length).

        Parameters:
            inputs (np.ndarray): Input tensor of shape (batch_size, sequence_length).

        Returns:
            np.ndarray: Output tensor of shape (batch_size, sequence_length, output_dim).
        """
        return self.weights[inputs]

    def __call__(self, inputs):
        """
        Forward pass of the Embedding layer.

        Parameters:
            inputs (np.ndarray): Input tensor of shape (batch_size, sequence_length).

        Returns:
            np.ndarray: Output tensor of shape (batch_size, sequence_length, output_dim).
        """
        if inputs is None:
            raise ValueError("Inputs cannot be None.")
        if not isinstance(inputs, np.ndarray):
            raise ValueError("inputs must be a numpy array.")
        
        # Ensure input shape is set if not provided
        if self.input_shape is None:
            if inputs.ndim == 2:
                self.set_input_shape(inputs.shape[1:])
            else:
                self.set_input_shape(inputs.shape)
            print("Input shape set to:", self.input_shape)
        
        # Validate weights
        if self.weights is None:
            raise ValueError("Weights must be set before calling the Embedding layer.")
        
        # Do embedding lookup
        if inputs.ndim == 2:
            batch_size, sequence_length = inputs.shape
            if sequence_length != self.input_shape[0]:
                raise ValueError(f"Input sequence length {sequence_length} does not match expected {self.input_shape[0]}.")
            output = self.__embedding_lookup(inputs)
        elif inputs.ndim == 1:
            sequence_length = inputs.shape[0]
            if sequence_length != self.input_shape[0]:
                raise ValueError(f"Input sequence length {sequence_length} does not match expected {self.input_shape[0]}.")
            output = self.__embedding_lookup(inputs.reshape(1, -1))
            output = output.squeeze(0)
        else:
            raise ValueError("Input must be a 1D or 2D tensor.")
        
        return output

    
if __name__ == "__main__":
    max_tokens = 1000  # Size of the vocabulary
    embedding_dim = 64  # Dimension of the embedding vectors
    sequence_length = (10,)  # Length of the input sequences (analogous to number of words in a sentence)
    batch_size = 100  # Number of sequences in a batch

    # Example usage
    embedding_layer = Embedding(
        input_dim=max_tokens,
        output_dim=embedding_dim,
        input_shape=sequence_length
    )
    embedding_layer.set_weights(np.random.rand(max_tokens, embedding_dim))

    inputs = np.array(np.random.randint(0, max_tokens, (batch_size, sequence_length[0])))
    # inputs = np.array(np.random.randint(0, max_tokens, sequence_length[0]))

    outputs = embedding_layer(inputs)

    print("Output shape:", outputs.shape)
    print("Output shape:", embedding_layer.compute_output_shape())
    # print("Output:", outputs)