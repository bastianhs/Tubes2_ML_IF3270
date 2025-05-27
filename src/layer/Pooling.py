import numpy as np

class Pooling():
    def __init__(self, pool_size=(2, 2), mode='max', input_shape=None, strides=None, padding='valid', **kwargs):
        """
        Initializes a MaxPooling layer.

        Args:
            pool_size (tuple): Size of the pooling window (height, width).
            mode (str): Type of pooling operation, 'max' or 'average'. Defaults to 'max'.
            input_shape (tuple, optional): Shape of the input data. Should be in the form (height, width, channels). Defaults to None.
            strides (tuple, optional): Strides of the pooling operation. Defaults to None, which means it will be equal to pool_size.
            padding (str): Padding type, either 'valid' or 'same'. Defaults to 'valid'.
        """
        if not isinstance(pool_size, tuple) or len(pool_size) != 2:
            raise ValueError("pool_size must be a tuple of (height, width).")
        if not all(isinstance(dim, int) and dim > 0 for dim in pool_size):
            raise ValueError("pool_size dimensions must be positive integers.")
        self.pool_size = pool_size

        self.set_input_shape(input_shape)

        self.mode = mode.lower()
        if self.mode not in ['max', 'average']:
            raise ValueError("mode must be either 'max' or 'average'.")

        if strides is None:
            self.strides = pool_size
        elif isinstance(strides, int) and strides > 0:
            self.strides = (strides, strides)
        elif isinstance(strides, tuple) and len(strides) == 2 and all(isinstance(dim, int) and dim > 0 for dim in strides):
            self.strides = strides
        else:
            raise ValueError("strides must be a positive integer or a tuple of two positive integers.")

        self.padding = padding.lower()
        if self.padding not in ['valid', 'same']:
            raise ValueError("padding must be either 'valid' or 'same'.")
        
    def get_config(self):
        """
        Returns the configuration of the MaxPooling layer.

        Returns:
            dict: Configuration dictionary.
        """
        return {
            'pool_size': self.pool_size,
            'mode': self.mode,
            'input_shape': self.input_shape if self.input_shape is not None else (None, None, None),
            'strides': self.strides,
            'padding': self.padding
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
        Computes the output shape of the Pooling layer given the input shape.

        Args:
            input_shape (tuple, optional): Shape of the input data. Should be (height, width, channels) or (batch, height, width, channels).

        Returns:
            tuple: Output shape after applying the pooling operation.
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
        
        pool_height, pool_width = self.pool_size
        stride_height, stride_width = self.strides
        if self.padding == "valid":
            output_height = ((height - pool_height) // stride_height) + 1
            output_width = ((width - pool_width) // stride_width) + 1
        elif self.padding == "same":
            output_height = ((height - 1) // stride_height) + 1
            output_width = ((width - 1) // stride_width) + 1
        else:
            raise ValueError("Padding must be either 'valid' or 'same'.")
        
        if output_height <= 0 or output_width <= 0:
            raise ValueError("Output dimensions must be positive. Check input shape, pool size, and strides.")
        
        return (output_height, output_width, channels)

    @property
    def trainable_weights(self):
        """
        Returns the trainable weights of the Pooling layer.

        Returns:
            list: An empty list since Pooling does not have trainable weights.
        """
        return []

    def __add_padding(self, inputs):
        """
        Applies padding to the input data based on the specified padding type.

        Args:
            inputs (np.ndarray): Input data of shape (height, width, channels) or (batch, height, width, channels).

        Returns:
            np.ndarray: Padded input data.
        """
        if self.padding == "valid":
            return inputs
        elif self.padding == "same":
            if inputs.ndim == 4:
                batch, input_height, input_width, input_channels = inputs.shape
            elif inputs.ndim == 3:
                input_height, input_width, input_channels = inputs.shape
                batch = None
            else:
                raise ValueError("Inputs must be 3D or 4D array.")
            
            pool_height, pool_width = self.pool_size
            stride_height, stride_width = self.strides
            
            # Calculate padding
            out_height = ((input_height - 1) // stride_height) + 1
            out_width = ((input_width - 1) // stride_width) + 1
            pad_along_height = max((out_height - 1) * stride_height + pool_height - input_height, 0)
            pad_along_width = max((out_width - 1) * stride_width + pool_width - input_width, 0)
            pad_top = pad_along_height // 2
            pad_bottom = pad_along_height - pad_top
            pad_left = pad_along_width // 2
            pad_right = pad_along_width - pad_left
            pad_widths = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
            
            if batch is not None:
                padded = np.stack([np.pad(inputs[b], pad_widths, mode='constant') for b in range(batch)], axis=0)
            else:
                padded = np.pad(inputs, pad_widths, mode='constant')
            return padded
        else:
            raise ValueError("Padding must be either 'valid' or 'same'.")
    
    def __pool(self, inputs):
        """
        Applies the pooling operation to the input data.

        Args:
            inputs (np.ndarray): Input data of shape (batch_size, height, width, channels).

        Returns:
            np.ndarray: Pooled output data.
        """
        if self.mode == 'max':
            return self.__max_pool(inputs)
        elif self.mode == 'average':
            return self.__average_pool(inputs)
        else:
            raise ValueError("mode must be either 'max' or 'average'.")
        
    def __max_pool(self, inputs):
        """
        Applies the max pooling operation to the input data.

        Args:
            inputs (np.ndarray): Input data of shape (batch_size, height, width, channels).

        Returns:
            np.ndarray: Pooled output data.
        """
        batch_size, height, width, channels = inputs.shape
        pool_height, pool_width = self.pool_size
        stride_height, stride_width = self.strides

        output_height = (height - pool_height) // stride_height + 1
        output_width = (width - pool_width) // stride_width + 1

        output = np.zeros((batch_size, output_height, output_width, channels))

        for b in range(batch_size):
            for h in range(output_height):
                    h_start = h * stride_height
                    h_end = h_start + pool_height
                    for w in range(output_width):
                        w_start = w * stride_width
                        w_end = w_start + pool_width
                        output[b, h, w] = np.max(inputs[b, h_start:h_end, w_start:w_end], axis=(0, 1))

        return output
    
    def __average_pool(self, inputs):
        """
        Applies the average pooling operation to the input data.

        Args:
            inputs (np.ndarray): Input data of shape (batch_size, height, width, channels).

        Returns:
            np.ndarray: Pooled output data.
        """
        batch_size, height, width, channels = inputs.shape
        pool_height, pool_width = self.pool_size
        stride_height, stride_width = self.strides

        output_height = (height - pool_height) // stride_height + 1
        output_width = (width - pool_width) // stride_width + 1

        output = np.zeros((batch_size, output_height, output_width, channels))

        for b in range(batch_size):
            for h in range(output_height):
                h_start = h * stride_height
                h_end = h_start + pool_height
                for w in range(output_width):
                    w_start = w * stride_width
                    w_end = w_start + pool_width
                    output[b, h, w] = np.mean(inputs[b, h_start:h_end, w_start:w_end], axis=(0, 1))

        return output
    
    def __call__(self, inputs):
        """
        Applies the Pooling operation to the input data.

        Args:
            inputs (np.ndarray): Input data of shape (batch, height, width, channels) or (height, width, channels).

        Returns:
            np.ndarray: Pooled output data.
        """
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
        
        # Validate inputs
        if inputs is None:
            raise ValueError("Inputs cannot be None.")
        if not isinstance(inputs, np.ndarray):
            raise ValueError("Inputs must be a numpy ndarray.")
        
        # Do padding and pooling
        if inputs.ndim == 3:
            padded_inputs = self.__add_padding(inputs)
            padded_inputs = np.expand_dims(padded_inputs, axis=0)
            output = self.__pool(padded_inputs)
            return output[0]
        elif inputs.ndim == 4:
            padded_inputs = self.__add_padding(inputs)
            output = self.__pool(padded_inputs)
            return output
        else:
            raise ValueError("Inputs must be a 3D or 4D numpy array.")
    
class MaxPooling(Pooling):
    def __init__(self, pool_size=(2, 2), input_shape=None, strides=None, padding='valid', **kwargs):
        """
        Initializes a MaxPooling layer.

        Args:
            pool_size (tuple): Size of the pooling window (height, width).
            input_shape (tuple, optional): Shape of the input data. Should be in the form (height, width, channels). Defaults to None.
            strides (tuple, optional): Strides of the pooling operation. Defaults to None, which means it will be equal to pool_size.
            padding (str): Padding type, either 'valid' or 'same'. Defaults to 'valid'.
        """
        super().__init__(pool_size=pool_size, mode='max', input_shape=input_shape, strides=strides, padding=padding, **kwargs)

class AveragePooling(Pooling):
    def __init__(self, pool_size=(2, 2), input_shape=None, strides=None, padding='valid', **kwargs):
        """
        Initializes an AveragePooling layer.

        Args:
            pool_size (tuple): Size of the pooling window (height, width).
            input_shape (tuple, optional): Shape of the input data. Should be in the form (height, width, channels). Defaults to None.
            strides (tuple, optional): Strides of the pooling operation. Defaults to None, which means it will be equal to pool_size.
            padding (str): Padding type, either 'valid' or 'same'. Defaults to 'valid'.
        """
        super().__init__(pool_size=pool_size, mode='average', input_shape=input_shape, strides=strides, padding=padding, **kwargs)

if __name__ == "__main__":
    input_shape = (100, 50, 3)
    inputs = np.random.rand(10, input_shape[0], input_shape[1], input_shape[2])

    # Without padding
    pooling_layer = MaxPooling(pool_size=(2, 2), input_shape=input_shape, strides=100, padding='valid')
    output = pooling_layer(inputs)
    print("Max pooled output shape:", output.shape)
    print("Max pooled output shape:", pooling_layer.compute_output_shape())
    # print("Output data:\n", output)
    print("Trainable parameters:", pooling_layer.trainable_weights)

    pooling_layer = AveragePooling(pool_size=(2, 2), input_shape=input_shape, strides=5, padding='valid')
    output = pooling_layer(inputs)
    print("Average pooled output shape:", output.shape)
    print("Average pooled output shape:", pooling_layer.compute_output_shape())
    # print("Output data:\n", output)
    print("Trainable parameters:", pooling_layer.trainable_weights)

    # With padding
    pooling_layer = MaxPooling(pool_size=(2, 2), input_shape=input_shape, strides=10, padding='same')
    output = pooling_layer(inputs)
    print("Max pooled output shape with padding:", output.shape)
    print("Max pooled output shape with padding:", pooling_layer.compute_output_shape())
    # print("Output data with padding:\n", output)
    print("Trainable parameters with padding:", pooling_layer.trainable_weights)
    
    pooling_layer = AveragePooling(pool_size=(2, 2), input_shape=input_shape, strides=3, padding='same')
    output = pooling_layer(inputs)
    print("Average pooled output shape with padding:", output.shape)
    print("Average pooled output shape with padding:", pooling_layer.compute_output_shape())
    # print("Output data with padding:\n", output)
    print("Trainable parameters with padding:", pooling_layer.trainable_weights)
