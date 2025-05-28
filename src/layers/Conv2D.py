import numpy as np
from ..utils.ActivationFunction import ActivationFunction

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
        # Validate and set the parameters for the Conv2D layer
        if not isinstance(filters, int) or filters <= 0:
            raise ValueError("filters must be a positive integer.")
        self.filters = filters

        if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
            raise ValueError("kernel_size must be a tuple of (height, width).")
        if not all(isinstance(k, int) and k > 0 for k in kernel_size):
            raise ValueError("kernel_size values must be positive integers.")
        self.kernel_size = kernel_size

        self.set_input_shape(input_shape)
        self.set_weights([kernel, bias])
        self.activation = ActivationFunction().activation(activation)
        self.activation_name = activation

        if not isinstance(strides, tuple) or len(strides) != 2:
            raise ValueError("strides must be a tuple of (stride_height, stride_width).")
        if not all(isinstance(s, int) and s > 0 for s in strides):
            raise ValueError("strides values must be positive integers.")
        self.strides = strides

        self.padding = padding.lower()
        if self.padding not in ['valid', 'same']:
            raise ValueError("padding must be either 'valid' or 'same'.")

        self.kwargs = kwargs

    def get_config(self):
        """
        Returns the configuration of the Conv2D layer.

        Returns:
            dict: Configuration dictionary.
        """
        return {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "input_shape": self.input_shape,
            "kernel": self.kernel,
            "bias": self.bias,
            "activation": self.activation_name,
            "strides": self.strides,
            "padding": self.padding,
            "kwargs": self.kwargs
        }

    def set_weights(self, weights):
        """
        Sets the kernel and bias weights for the Conv2D layer.

        Args:
            weights (list): [kernel, bias] where kernel is np.ndarray and bias is np.ndarray.
        """
        if not isinstance(weights, (list, tuple)) or len(weights) != 2:
            raise ValueError("weights must be a list or tuple of [kernel, bias].")
        
        kernel, bias = weights
        if kernel is not None:
            if not isinstance(kernel, np.ndarray):
                raise ValueError("kernel must be a numpy ndarray.")
            if kernel.ndim != 4:
                raise ValueError("kernel shape must be (kernel_height, kernel_width, input_channels, filters).")
            if kernel.shape[0] != self.kernel_size[0] or kernel.shape[1] != self.kernel_size[1]:
                raise ValueError(f"kernel size must match kernel_size {self.kernel_size}.")
            if self.input_shape and kernel.shape[2] != self.input_shape[2]:
                raise ValueError(f"Input channels {self.input_shape[2]} do not match kernel channels {kernel.shape[2]}.")
            if kernel.shape[3] != self.filters:
                raise ValueError(f"Kernel filters {kernel.shape[3]} do not match specified filters {self.filters}.")
        else:
            kernel = np.random.rand(self.kernel_size[0], self.kernel_size[1], self.input_shape[2], self.filters) if self.input_shape else None
        self.kernel = kernel

        if bias is not None:
            if not isinstance(bias, np.ndarray):
                raise ValueError("bias must be a numpy ndarray.")
            if bias.ndim != 1:
                raise ValueError(f"bias shape must be ({self.filters},).")
            if bias.shape[0] != self.filters:
                raise ValueError(f"Bias shape {bias.shape} does not match filters {self.filters}.")
        else:
            bias = np.random.rand(self.filters) if self.filters > 0 else None
        self.bias = bias

    def get_weights(self):
        """
        Returns the kernel and bias weights of the Conv2D layer.

        Returns:
            list: [kernel, bias]
        """
        return [self.kernel, self.bias]

    def set_input_shape(self, input_shape):
        """
        Sets the input shape for the Conv2D layer.

        Args:
            input_shape (tuple): Shape of the input data. It should be in the form (height, width, channels).
        """
        if input_shape is not None:
            if not isinstance(input_shape, tuple) or len(input_shape) != 3:
                raise ValueError("input_shape must be a tuple of (height, width, channels).")
            if not all(isinstance(dim, int) and dim > 0 for dim in input_shape):
                raise ValueError("input_shape dimensions must be positive integers.")
        self.input_shape = input_shape

    def compute_output_shape(self, input_shape=None):
        """
        Computes the output shape of the Conv2D layer given the input shape.

        Args:
            input_shape (tuple, optional): Shape of the input data. Defaults to None. It should be in the form (height, width, channels) or (batch, height, width, channels).

        Returns:
            tuple: Output shape after applying the convolution.
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

        kernel_height, kernel_width = self.kernel_size
        stride_height, stride_width = self.strides
        if self.padding == "valid":
            output_height = ((height - kernel_height) // stride_height) + 1
            output_width = ((width - kernel_width) // stride_width) + 1
        elif self.padding == "same":
            output_height = ((height - 1) // stride_height) + 1
            output_width = ((width - 1) // stride_width) + 1
        else:
            raise ValueError("Padding must be either 'valid' or 'same'.")
        
        if output_height <= 0 or output_width <= 0:
            raise ValueError("Output dimensions must be positive. Check input shape, kernel size, or strides.")

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

            # Calculate padding
            out_height =  ((input_height - 1) // stride_height) + 1
            out_width = ((input_width - 1) // stride_width) + 1
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
        # Add padding
        x = self.__add_padding(inputs)
        
        # input, kernel, and stride dimensions
        input_height, input_width, input_channels = x.shape
        kernel_height, kernel_width, kernel_in_channels, num_filters = self.kernel.shape
        stride_height, stride_width = self.strides

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
        output = self.activation(output)
        
        return output
    
    def __call__(self, inputs):
        """
        Applies the Conv2D layer to the input data.

        Args:
            inputs (np.ndarray): Input data to the Conv2D layer. Shape: (height, width, channels) or (batch, height, width, channels)

        Returns:
            np.ndarray: Output after applying the convolution and activation function.
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

        # Validate weights
        if self.kernel is None or self.bias is None:
            raise ValueError("Kernel and bias must be set before calling the layer.")
                
        # Do the convolution
        if inputs.ndim == 3:
            # Single sample, shape (height, width, channels)
            if self.input_shape is not None and inputs.shape != self.input_shape:
                raise ValueError(f"Input shape {inputs.shape} does not match expected input shape {self.input_shape}.")
            if inputs.shape[2] != self.kernel.shape[2]:
                raise ValueError(f"Input channels {inputs.shape[2]} do not match kernel channels {self.kernel.shape[2]}.")
            return self.__convolution(inputs)
        elif inputs.ndim == 4:
            # Batch input, shape (batch, height, width, channels)
            batch_size = inputs.shape[0]
            results = []
            for i in range(batch_size):
                sample = inputs[i]
                if self.input_shape is not None and sample.shape != self.input_shape:
                    raise ValueError(f"Input shape {sample.shape} does not match expected input shape {self.input_shape}.")
                if sample.shape[2] != self.kernel.shape[2]:
                    raise ValueError(f"Input channels {sample.shape[2]} do not match kernel channels {self.kernel.shape[2]}.")
                results.append(self.__convolution(sample))
            return np.stack(results, axis=0)
        else:
            raise ValueError("Inputs must have shape (height, width, channels) or (batch, height, width, channels).")
    
if __name__ == "__main__":
    batch_size = 1
    channels = 10
    input_height = 32
    input_width = 24
    kernel_height = 3
    kernel_width = 5
    filters = 32
    strides = (1, 1)
    padding = "valid"
    activation = "relu"

    # Example usage
    kernel = np.random.rand(kernel_height, kernel_width, channels, filters)
    bias = np.random.rand(filters)

    conv_layer = Conv2D(
        filters=filters,
        kernel_size=(kernel_height, kernel_width),
        input_shape=(input_height, input_width, channels),
        activation=activation,
        strides=strides,
        padding=padding,
        # kernel=kernel,
        # bias=bias
    )
    conv_layer.set_weights([kernel, bias])

    # input tensor
    # inputs = np.random.rand(batch_size, input_height, input_width, channels)
    inputs = np.random.rand(input_height, input_width, channels)

    # Apply the Conv2D layer
    output = conv_layer(inputs)
    print("Output shape                            :", output.shape)
    print("Output shape from compute_output_shape():", conv_layer.compute_output_shape())

    # # Get the configuration of the Conv2D layer
    # config = conv_layer.get_config()
    # print("Conv2D Layer Configuration:")
    # for key, value in config.items():
    #     print(f"{key}: {value}")

    # Get the trainable weights
    trainable_weights = conv_layer.trainable_weights
    print("Trainable Weights:")
    for weight in trainable_weights:
        if weight is not None:
            print(weight.shape)
        else:
            print("None")