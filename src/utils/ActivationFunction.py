"""
This module contains the activation functions used in model. 

List of activation functions
- Linear
- ReLU
- Sigmoid
- Hyperbolic Tangent (tanh)
- Softmax
"""

import numpy as np

class ActivationFunction:
    """
    Base class for activation functions.
    """

    def __linear(self, x: np.ndarray) -> np.ndarray:
        """
        Linear activation function.

        Parameters:
        x (np.ndarray): Input array.

        Returns:
        np.ndarray: Output array after applying the linear activation function.
        """
        return x
    
    def __relu(self, x: np.ndarray) -> np.ndarray:
        """
        ReLU activation function.

        Parameters:
        x (np.ndarray): Input array.

        Returns:
        np.ndarray: Output array after applying the ReLU activation function.
        """
        return np.maximum(0, x)
    
    def __sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function.

        Parameters:
        x (np.ndarray): Input array.

        Returns:
        np.ndarray: Output array after applying the sigmoid activation function.
        """
        return 1 / (1 + np.exp(-x))
    
    def __tanh(self, x: np.ndarray) -> np.ndarray:

        """
        Hyperbolic Tangent (tanh) activation function.

        Parameters:
        x (np.ndarray): Input array.

        Returns:
        np.ndarray: Output array after applying the tanh activation function.
        """
        return np.tanh(x)
    
    def __softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Softmax activation function.

        Parameters:
        x (np.ndarray): Input array.

        Returns:
        np.ndarray: Output array after applying the softmax activation function.
        """
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def activation(self, activation_type: str) -> np.ndarray:
        """
        Apply the specified activation function to the input.

        Parameters:
        x (np.ndarray): Input array.
        activation_type (str): Name of the activation function.
        kwargs (dict): Additional parameters for the activation function.

        Returns:
        np.ndarray: Output array after applying the activation function.
        """
        if activation_type == 'linear':
            return self.__linear
        elif activation_type == 'relu':
            return self.__relu
        elif activation_type == 'sigmoid':
            return self.__sigmoid
        elif activation_type == 'tanh':
            return self.__tanh
        elif activation_type == 'softmax':
            return self.__softmax
        else:
            raise ValueError(f"Unknown activation function: {activation_type}. Use 'linear', 'relu', 'sigmoid', 'tanh', or 'softmax'.")