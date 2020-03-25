from .utils import tanh, d_tanh

import numpy as np
from scipy.stats import truncnorm

def truncated_normal(mean=0, sd=1, low=0, upp=10):
    """Compute Truncated Norm value for weight matrix initialization"""
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

class ff_network():
    """Fully Connected Neural Network"""
    def __init__(self,
            input_dim,
            hidden_dim,
            output_dim,
            learning_rate=1e-3,
            momentum=0.9,
            bias=None,
            batch_size=1
            ):
        """Intialize weights
        Args:
            input_dim: Input layer dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output layer dimension
            learning_rate (Default=1e-5): Learning rate
            momentum (Default=0.9): Momentum factor
            bias (Defualt=None): Use bais value or not
            batch_size (Default=1): Batch Size
        """
        super().__init__()

        # Parameters
        self.inputSize = input_dim
        self.outputSize = output_dim
        self.hiddenSize = hidden_dim
        self.lr = learning_rate
        self.momentum = momentum
        self.bias = bias
        self.batch_size = batch_size

        # Initialize weight matrices
        self.create_weight_matrices()

    def create_weight_matrices(self):
        """ A method to initialize the weight matrices
        of the neural network"""

        bias_node = 1 if self.bias else 0

        rad = 1 / np.sqrt(self.inputSize + bias_node)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.w1 = X.rvs((self.hiddenSize,
            self.inputSize + bias_node))

        rad = 1 / np.sqrt(self.hiddenSize + bias_node)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.w2 = X.rvs((self.hiddenSize,
            self.hiddenSize + bias_node))

        rad = 1 / np.sqrt(self.hiddenSize + bias_node)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.w3 = X.rvs((self.outputSize,
            self.hiddenSize + bias_node))

        # Auxilary matrices for storing momentum calculation
        self.w1_delta = np.zeros((self.hiddenSize,
            self.inputSize + bias_node))
        self.w2_delta = np.zeros((self.hiddenSize,
            self.hiddenSize + bias_node))
        self.w3_delta = np.zeros((self.outputSize,
            self.hiddenSize + bias_node))

    def forward(self, x):
        """Forward pass on Neural Network
        Args:
            x: Input matrix
        Returns:
            output: Output matrix
        """
        # If bias is present, concatenate vector of 1's
        if self.bias:
            x = np.concatenate((x, \
                np.ones((x.shape[0],1))), axis=1)

        x = np.array(x, ndmin=2).T

        # Forward pass first layer
        self.v1 = np.dot(self.w1, x)
        self.o1 = tanh(self.v1)

        # If bias is present, concatenate vector of 1's
        if self.bias:
            self.o1 = np.concatenate((self.o1, \
                    np.ones((1, self.o1.shape[1]))),axis=0)

        # Forward pass second layer
        self.v2 = np.dot(self.w2, self.o1)
        self.o2 = tanh(self.v2)

        # If bias is present, concatenate vector of 1's
        if self.bias:
            self.o2 = np.concatenate((self.o2, \
                    np.ones((1, self.o2.shape[1]))),axis=0)

        # Forward pass third layer
        self.v3 = np.dot(self.w3, self.o2)
        self.o3 = tanh(self.v3)
        return self.o3

    def backwards(self, inp, expected):
        """Forward pass on Neural Network
        Args:
            inp: Input matrix
            expected: Expected value matrix
        """
        # If bias is present, concatenate vector of 1's
        if self.bias:
            inp = np.concatenate((inp, \
                    np.ones((inp.shape[0],1))), axis=1)


        inp = np.array(inp, ndmin=2).T
        expected = expected.T
        # Compute error at last layer
        self.o_error = np.mean(expected - self.o3, axis = 1, keepdims = True)
        # Note: Refer backpropogation slides for full details on
        #       backprop equation.
        # TODO: Backpropogation through multiple layers in a loop.

        # Pass output loss through derivative of activation function.
        # In this case, Tanh.
        self.l3_error = self.o_error*d_tanh(np.mean(self.v3,axis = 1 , keepdims = True))

        # Compute delta of weights in layer. Scale delta by a factor of
        # learning rate.
        delta_w3 = self.lr*np.dot(self.l3_error, np.mean(self.o2, axis = 1, keepdims = True).T)
        # Add delta and momentum factor for improved stability to
        # weight matrix.
        self.w3 += (1 - self.momentum)*delta_w3 + self.momentum * self.w3_delta

        # Update momentum factor
        self.w3_delta = delta_w3

        # Note: The above steps are repeated for the two other weight
        #       matrices. In the next revision of this code, will use
        #       loops to optimize and streamline backprop computation.

        # Backprop for weight matrix 2
        if self.bias:
            self.l2_error = np.dot(self.w3[:,:-1].T, self.l3_error)*d_tanh(np.mean(self.v2,axis = 1, keepdims = True))
        else :
            self.l2_error = np.dot(self.w3.T, self.l3_error)*d_tanh(np.mean(self.v2,axis = 1, keepdims = True))
        delta_w2 = self.lr*np.dot(self.l2_error, np.mean(self.o1, axis = 1, keepdims= True).T)

        self.w2 += (1 - self.momentum)*delta_w2 + self.momentum * self.w2_delta
        self.w2_delta = delta_w2


        # Backprop for weight matrix 1
        if self.bias:
            self.l1_error = np.dot(self.w2[:,:-1].T, self.l2_error)*d_tanh(np.mean(self.v1,axis = 1, keepdims = True))
        else :
            self.l1_error = np.dot(self.w2.T, self.l2_error)*d_tanh(np.mean(self.v1,axis = 1, keepdims = True))
        delta_w1 = self.lr*np.dot(self.l1_error, np.mean(inp, axis = 1, keepdims= True).T)

        self.w1 += (1 - self.momentum)*delta_w1 + self.momentum * self.w1_delta
        self.w1_delta = delta_w1

    def eval(self, x):
        """Evaluate the Neural Network
        Args:
            x: Input value (ie. No batch sized matrix)
        Returns:
            output: Output value
        """
        # If bias is present, concatenate vector of 1's
        if self.bias:
            x = np.concatenate((x, [1]))

        x = np.array(x, ndmin=2).T

        o1 = tanh(np.dot(self.w1, x))

        # If bias is present, concatenate vector of 1's
        if self.bias:
            o1 = np.concatenate((o1, \
                    np.ones((1, o1.shape[1]))), axis=0)

        o2 = tanh(np.dot(self.w2, o1))

        # If bias is present, concatenate vector of 1's
        if self.bias:
            o2 = np.concatenate((o2, \
                    np.ones((1, o2.shape[1]))), axis=0)

        o3 = tanh(np.dot(self.w3, o2))
        return o3