from .utils import tanh, d_tanh

import numpy as np
import keras

from keras.models import Sequential
from keras import regularizers
from keras.layers import Dense, Dropout
from keras.initializers import lecun_normal, Constant

class ff_network(Sequential):
    """Fully Connected Neural Network"""
    def __init__(self,
            learning_rate=1e-3,
            momentum=0.9,
            bias=None,
            batch_size=1
            ):
        """Intialize weights

        Args:
            learning_rate (Default=1e-5): Learning rate
            momentum (Default=0.9): Momentum factor
            bias (Defualt=None): Use bais value or not
            batch_size (Default=1): Batch Size
        """
        super().__init__()

        # Parameters
        self.lr = learning_rate
        self.momentum = momentum
        self.bias = bias
        self.batch_size = batch_size
        self.num_layers = 0

        # Auxilary matrix for storing momentum calculation
        self.w_delta = []
        self.b_delta = []

    def add_linear(self,
            input_dim,
            output_dim,
            INPUT = False
            ):
        """Add layer to the network

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
        """
        if INPUT == True:
            self.add(Dense(output_dim = output_dim,
                        use_bias = self.bias,
                        kernel_initializer = lecun_normal(seed=None),
                        bias_initializer = Constant(0.1),
                        activation = 'tanh',
                        input_dim = input_dim)
                    )
        else:
            self.add(Dense(output_dim = output_dim,
                        use_bias = self.bias,
                        kernel_initializer = lecun_normal(seed=None),
                        bias_initializer = Constant(0.1),
                        activation = 'tanh')
                    )

        # Weight delta for momentum calculation
        w_d = np.zeros([output_dim, input_dim])
        if self.bias:
            b_d = np.zeros(self.layers[self.num_layers].get_weights()[1].shape)

        self.w_delta.append(w_d)
        self.b_delta.append(b_d)
        self.num_layers += 1

    def forward(self, x):
        """Forward pass on Neural Network

        Args:
            x: Input matrix

        Returns:
            output: Output matrix
        """
        x = np.array(x, ndmin=2).T

        self.v = []
        self.o = []

        # Iterate over every layer and forward pass, store v and o
        # values.
        for layer_idx in range(self.num_layers):
            weight = self.layers[layer_idx].get_weights()[0]
            v = np.dot(weight.T, x)
            if self.bias:
                bias = self.layers[layer_idx].get_weights()[1]
                bias = np.expand_dims(bias, axis=1)
                v += bias
            x = tanh(v)

            self.v.append(v)
            self.o.append(x)

        return x

    def backwards(self, inp, expected):
        """Forward pass on Neural Network

        Args:
            inp: Input matrix
            expected: Expected value matrix
        """
        inp = np.array(inp, ndmin=2).T

        # Compute error at last layer
        self.o_error = np.mean(expected.T - self.o[self.num_layers-1],
                axis=1, keepdims=True)

        # Note: Refer backpropogation slides for full details on
        #       backprop equation.
        # TODO: Backpropogation through multiple layers in a loop.

        # Pass output loss through derivative of activation function.
        # In this case, Tanh.
        self.llayer_error = self.o_error * \
                np.mean(d_tanh(self.v[self.num_layers-1]),
                        axis=1, keepdims=True)

        # Compute delta of weights in layer. Scale delta by a factor of
        # learning rate.
        delta_llast_layer = self.lr * \
                np.dot(self.llayer_error,
                    np.mean(self.o[self.num_layers-2],
                        axis=1, keepdims=True).T)

        # Add delta and momentum factor for improved stability to
        # weight matrix.
        weight = self.layers[self.num_layers-1].get_weights()[0]
        if self.bias :
            bias = self.layers[self.num_layers-1].get_weights()[1]
            bias_delta = self.lr * self.llayer_error.reshape(-1)
            updatedWeights = [weight + \
                delta_llast_layer.T + \
                self.momentum * self.w_delta[self.num_layers-1].T,
                bias + bias_delta + self.momentum*self.b_delta[self.num_layers-1]]
        else:
            updatedWeights = [weight + \
                (1 - self.momentum)*delta_llast_layer.T + \
                self.momentum * self.w_delta[self.num_layers-1].T]

        self.layers[self.num_layers-1].set_weights(updatedWeights)
        # Update momentum factor
        self.w_delta[self.num_layers-1] = delta_llast_layer
        self.b_delta[self.num_layers-1] = bias_delta

        # Note: The above steps are repeated for the two other weight
        # matrices.
        for layer_idx in range(self.num_layers-2, -1, -1):
            if self.bias:
                self.llayer_error = np.dot(self.layers[layer_idx+1].get_weights()[0],
                    self.llayer_error) * np.mean(d_tanh(self.v[layer_idx]), axis = 1, keepdims = True)

            if layer_idx == 0:
                delta_llast_layer = self.lr * \
                        np.dot(self.llayer_error, np.mean(inp, axis = 1, keepdims = True).T)
            else:
                delta_llast_layer = self.lr * \
                        np.dot(self.llayer_error, np.mean(self.o[layer_idx-1], axis = 1, keepdims= True).T)

            weight = self.layers[layer_idx].get_weights()[0]
            if self.bias:
                bias = self.layers[layer_idx].get_weights()[1]
                bias_delta = self.lr * self.llayer_error.reshape(-1)
                updatedWeights = [weight + delta_llast_layer.T + \
                            self.momentum * self.w_delta[layer_idx].T,
                        bias + bias_delta + self.momentum*self.b_delta[layer_idx]]
            else:
                updatedWeights = [weight + (1 - self.momentum) * delta_llast_layer.T + \
                            self.momentum * self.w_delta[layer_idx].T]

            self.layers[layer_idx].set_weights(updatedWeights)
            self.w_delta[layer_idx] = delta_llast_layer
            self.b_delta[layer_idx] = bias_delta

        del self.o
        del self.v

import torch
from torch import nn

def init_weights_agent(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        try:
            nn.init.xavier_uniform_(m.bias)
        except:
            a = 0

class pytorch_network(nn.Module):
    """Fully Connected Neural Network"""
    def __init__(self,
            learning_rate=1e-3,
            momentum=0.9,
            bias=None,
            batch_size=1
            ):
        """Intialize weights

        Args:
            learning_rate (Default=1e-5): Learning rate
            momentum (Default=0.9): Momentum factor
            bias (Defualt=None): Use bais value or not
            batch_size (Default=1): Batch Size
        """
        super().__init__()

        # Parameters
        self.lr = learning_rate
        self.momentum = momentum
        self.bias = bias
        self.batch_size = batch_size
        self.num_layers = 0

        self.layers = nn.Sequential()

        # Auxilary matrix for storing momentum calculation
        self.w_delta = []
        self.b_delta = []

    def add_linear(self,
            input_dim,
            output_dim,
            INPUT = False
            ):
        """Add layer to the network

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
        """
        self.layers.add_module(str(self.num_layers),
            nn.Linear(input_dim, output_dim, bias=self.bias))
        self.layers.apply(init_weights_agent)

        # Weight delta for momentum calculation
        w_d = torch.zeros([output_dim, input_dim])
        if self.bias:
            b_d = torch.zeros(self.layers[self.num_layers].bias.shape)

        self.w_delta.append(w_d)
        self.b_delta.append(b_d)
        self.num_layers += 1

    def forward(self, x):
        """Forward pass on Neural Network

        Args:
            x: Input matrix

        Returns:
            output: Output matrix
        """
        x = torch.Tensor(x)

        self.v = []
        self.o = []

        with torch.no_grad():
            # Iterate over every layer and forward pass, store v and o
            # values.
            for layer_idx in range(self.num_layers):
                weight = self.layers[layer_idx].weight
                v = torch.matmul(x, weight.T)
                if self.bias:
                    bias = self.layers[layer_idx].bias
                    v += bias.T
                x = torch.tanh(v)

                self.v.append(v)
                self.o.append(x)

        return x

    def backwards(self, inp, expected):
        """Forward pass on Neural Network

        Args:
            inp: Input matrix
            expected: Expected value matrix
        """
        inp = torch.Tensor(inp)
        expected = torch.Tensor(expected)

        with torch.no_grad():

            # Compute error at last layer
            self.o_error = torch.mean(expected - self.o[self.num_layers-1],
                    dim=0, keepdims=True)

            # Note: Refer backpropogation slides for full details on
            #       backprop equation.
            # TODO: Backpropogation through multiple layers in a loop.

            # Pass output loss through derivative of activation function.
            # In this case, Tanh.
            self.llayer_error = self.o_error * \
                    torch.mean(d_tanh(self.v[self.num_layers-1]),
                            dim=0, keepdims=True)

            # Compute delta of weights in layer. Scale delta by a factor of
            # learning rate.
            delta_llast_layer = self.lr * \
                    torch.matmul(self.llayer_error,
                        torch.mean(self.o[self.num_layers-2],
                            dim=0, keepdims=True))

            # Add delta and momentum factor for improved stability to
            # weight matrix.
            self.layers[self.num_layers-1].weight += \
                    1 * delta_llast_layer + \
                    self.momentum * self.w_delta[self.num_layers-1]
            if self.bias :
                self.layers[self.num_layers-1].bias += \
                        self.lr * self.llayer_error.reshape(-1) + \
                        self.momentum * self.b_delta[self.num_layers-1]
            # Update momentum factor
            self.w_delta[self.num_layers-1] = delta_llast_layer
            self.b_delta[self.num_layers-1] = self.lr * self.llayer_error.reshape(-1)

            # Note: The above steps are repeated for the two other weight
            # matrices.
            for layer_idx in range(self.num_layers-2, -1, -1):
                if self.bias:
                    self.llayer_error = torch.matmul(self.llayer_error,
                                        self.layers[layer_idx+1].weight) * \
                                        torch.mean(d_tanh(self.v[layer_idx]),
                                                        dim=0, keepdims=True)

                if layer_idx == 0:
                    delta_llast_layer = self.lr * \
                            torch.matmul(self.llayer_error.T,
                                torch.mean(inp, dim=0, keepdims=True))
                else:
                    delta_llast_layer = self.lr * \
                            torch.matmul(self.llayer_error.T,
                                torch.mean(self.o[layer_idx-1], dim=0, keepdims=True))

                self.layers[layer_idx].weight += 1 * delta_llast_layer + \
                                                    self.momentum * self.w_delta[layer_idx]
                if self.bias:
                    self.layers[layer_idx].bias += self.lr * self.llayer_error.reshape(-1) + \
                                self.momentum * self.b_delta[layer_idx]

                self.w_delta[layer_idx] = delta_llast_layer
                self.b_delta[layer_idx] = self.lr * self.llayer_error.reshape(-1)

            del self.o
            del self.v

    def predict(self, x):
        """Forward pass on Neural Network for prediction

        Args:
            x: Input matrix

        Returns:
            output: Output matrix
        """
        x = torch.Tensor(x)

        with torch.no_grad():
            # Iterate over every layer and forward pass, store v and o
            # values.
            for layer_idx in range(self.num_layers):
                weight = self.layers[layer_idx].weight
                v = torch.matmul(x, weight.T)
                if self.bias:
                    bias = self.layers[layer_idx].bias
                    v += bias.T
                x = torch.tanh(v)

        return x
