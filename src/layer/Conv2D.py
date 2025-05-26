import numpy as np

class Conv2D():
    def __init__(
            self,
            filters,
            kernel_size,
            input_shape=None,
            kernel=None,
            bias=None,
            activation=None,
            strides=(1, 1),
            padding="valid",
            **kwargs
        ):
        """
        Initializes a Conv2D layer.

        Args:
            filters (int): Number of filters in the convolution.
            kernel_size (tuple): Size of the convolution kernel (height, width).
            input_shape (tuple, optional): Shape of the input data. Defaults to None. It should be in the form (height, width, channels).
            kernel (np.ndarray, optional): Weights for the convolution kernel. Defaults to None. Kernel should have shape (kernel_height, kernel_width, input_channels, filters).
            bias (np.ndarray, optional): Weights for the bias. Defaults to None. Bias should have shape (filters,).
            activation (callable, optional): Activation function to apply after convolution. Defaults to None.
            strides (tuple, optional): Strides of the convolution. Defaults to (1, 1).
            padding (str, optional): Padding type, either 'valid' or 'same'. Defaults to 'valid'.
            **kwargs: Additional keyword arguments.
        """
        self.filters = filters
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        self.kernel = kernel
        self.bias = bias
        self.activation = activation
        self.strides = strides
        self.padding = padding

    def set_weights(self, weights):
        """
        Sets the kernel and bias weights for the Conv2D layer.

        Args:
            weights (list): [kernel, bias] where kernel is np.ndarray and bias is np.ndarray.
        """
        if not isinstance(weights, (list, tuple)) or len(weights) != 2:
            raise ValueError("weights must be a list or tuple of [kernel, bias].")
        self.kernel, self.bias = weights

    def get_weights(self):
        """
        Returns the kernel and bias weights of the Conv2D layer.

        Returns:
            list: [kernel, bias]
        """
        return [self.kernel, self.bias]

    def get_config(self):
        """
        Returns the configuration of the Conv2D layer.

        Returns:
            dict: Configuration of the Conv2D layer.
        """
        return {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "input_shape": self.input_shape,
            "kernel": self.kernel,
            "bias": self.bias,
            "activation": self.activation,
            "strides": self.strides,
            "padding": self.padding
        }

    def compute_output_shape(self, input_shape=None):
        """
        Computes the output shape of the Conv2D layer given the input shape.

        Args:
            input_shape (tuple, optional): Shape of the input data. Defaults to None. It should be in the form (height, width, channels).

        Returns:
            tuple: Output shape after applying the convolution.
        """
        if input_shape is None:
            if self.input_shape is None:
                raise ValueError("Input shape must be provided or set during initialization.")
            input_shape = self.input_shape

        height, width, channels = input_shape
        kernel_height, kernel_width = self.kernel_size
        stride_height, stride_width = self.strides
        
        if self.padding == "valid":
            output_height = (height - kernel_height) // stride_height + 1
            output_width = (width - kernel_width) // stride_width + 1
        elif self.padding == "same":
            output_height = int(np.ceil(float(height) / float(stride_height)))
            output_width = int(np.ceil(float(width) / float(stride_width)))
        else:
            raise ValueError("Padding must be either 'valid' or 'same'.")
        
        if output_height <= 0 or output_width <= 0:
            raise ValueError("Output dimensions must be positive. Check input shape, kernel size, and strides.")
        
        return (output_height, output_width, self.filters)

    @property
    def trainable_weights(self):
        """
        Returns the trainable weights of the Conv2D layer (kernel and bias).
        Returns:
            list: [kernel, bias]
        """
        if self.kernel is None or self.bias is None:
            raise ValueError("Kernel and bias must be set before retrieving trainable weights.")
        return [self.kernel, self.bias]

    def __add_padding(self, inputs):
        """
        Adds padding to the input data based on the specified padding type.

        Args:
            inputs (np.ndarray): Input data to the Conv2D layer.

        Returns:
            np.ndarray: Padded input data.
        """
        if self.padding == "valid":
            return inputs
        elif self.padding == "same":
            input_height, input_width, input_channels = inputs.shape
            kernel_height, kernel_width = self.kernel_size
            stride_height, stride_width = self.strides

            # Calculate padding for 'same' convolution
            out_height = int(np.ceil(float(input_height) / float(stride_height)))
            out_width = int(np.ceil(float(input_width) / float(stride_width)))
            pad_along_height = max((out_height - 1) * stride_height + kernel_height - input_height, 0)
            pad_along_width = max((out_width - 1) * stride_width + kernel_width - input_width, 0)
            pad_top = pad_along_height // 2
            pad_bottom = pad_along_height - pad_top
            pad_left = pad_along_width // 2
            pad_right = pad_along_width - pad_left

            # Pad only height and width, not channels
            padded = np.pad(inputs, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')
            return padded
        else:
            raise ValueError("Padding must be either 'valid' or 'same'.")

    def __convolution(self, inputs):
        """
        Performs the convolution operation on the input data from scratch.
        This method applies the convolution operation using the kernel and padding specified during initialization.

        Args:
            inputs (np.ndarray): Input data to the Conv2D layer.

        Returns:
            np.ndarray: Output after applying the convolution.
        """
        if self.kernel is None:
            raise ValueError("Kernel must be set before performing convolution.")
        if self.padding not in ["valid", "same"]:
            raise ValueError("Padding must be either 'valid' or 'same'.")

        # Add padding if needed
        x = self.__add_padding(inputs)
        input_height, input_width, input_channels = x.shape
        kernel_height, kernel_width, kernel_in_channels, num_filters = self.kernel.shape
        stride_height, stride_width = self.strides
        if input_channels != kernel_in_channels:
            raise ValueError("Input channels must match kernel channels.")

        # Output shape
        output_height = (input_height - kernel_height) // stride_height + 1
        output_width = (input_width - kernel_width) // stride_width + 1
        output = np.zeros((output_height, output_width, num_filters))

        # Convolution operation
        for f in range(num_filters):
            for i in range(output_height):
                for j in range(output_width):
                    region = x[
                        i*stride_height:i*stride_height+kernel_height,
                        j*stride_width:j*stride_width+kernel_width,
                        :
                    ]  # shape: (kernel_height, kernel_width, input_channels)
                    output[i, j, f] = np.sum(region * self.kernel[:, :, :, f])
            # Add bias if available
            if self.bias is not None:
                output[:, :, f] += self.bias[f]
        
        # Apply activation if available
        if self.activation is not None:
            output = self.activation(output)
        
        return output
    
    def __call__(self, inputs):
        """
        Applies the Conv2D layer to the input data.

        Args:
            inputs (np.ndarray): Input data to the Conv2D layer.

        Returns:
            np.ndarray: Output after applying the convolution and activation function.
        """
        if self.kernel is None or self.bias is None:
            raise ValueError("Kernel and bias must be set before calling the layer.")

        return self.__convolution(inputs)
    
if __name__ == "__main__":
    # Example usage
    conv_layer = Conv2D(filters=32, kernel_size=(3, 3), input_shape=(10, 10, 3))
    conv_layer.set_weights([np.random.rand(3, 3, 3, 32), np.random.rand(32)])  # Random kernel and bias

    # Get the configuration of the layer
    config = conv_layer.get_config()
    print("Layer configuration:", config)

    # Create a random input tensor
    inputs = np.random.rand(10, 10, 3)

    # Apply the Conv2D layer
    output = conv_layer(inputs)
    print("Output:", output)
    print("Output shape:", output.shape)
    
    # Get the output shape
    output_shape = conv_layer.compute_output_shape(inputs.shape)
    print("Output shape:", output_shape)

    # Get trainable parameters
    trainable_params = conv_layer.trainable_weights
    print("Trainable parameters:", [param.shape for param in trainable_params])