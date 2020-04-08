import torch
from torch import nn
from .utils import tanh, d_tanh

import numpy as np

from keras.models import Sequential
from keras import regularizers
from keras.layers import Dense, Dropout
from keras.initializers import lecun_normal, Constant

torch.set_printoptions(precision=4)
torch.set_default_dtype(torch.float32)
torch.Generator().manual_seed(2147483647)
print("Initial Seed: ", torch.Generator().initial_seed())

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
                   INPUT=False
                   ):
        """Add layer to the network

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
        """
        if INPUT:
            self.add(Dense(output_dim=output_dim,
                           use_bias=self.bias,
                           kernel_initializer=lecun_normal(seed=None),
                           bias_initializer=Constant(0.1),
                           activation='tanh',
                           input_dim=input_dim)
                     )
        else:
            self.add(Dense(output_dim=output_dim,
                           use_bias=self.bias,
                           kernel_initializer=lecun_normal(seed=None),
                           bias_initializer=Constant(0.1),
                           activation='tanh')
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
        self.o_error = np.mean(expected.T - self.o[self.num_layers - 1],
                               axis=1, keepdims=True)

        # Note: Refer backpropogation slides for full details on
        #       backprop equation.
        # TODO: Backpropogation through multiple layers in a loop.

        # Pass output loss through derivative of activation function.
        # In this case, Tanh.
        self.llayer_error = self.o_error * \
            np.mean(d_tanh(self.v[self.num_layers - 1]),
                    axis=1, keepdims=True)

        # Compute delta of weights in layer. Scale delta by a factor of
        # learning rate.
        delta_llast_layer = self.lr * \
            np.dot(self.llayer_error,
                   np.mean(self.o[self.num_layers - 2],
                           axis=1, keepdims=True).T)

        # Add delta and momentum factor for improved stability to
        # weight matrix.
        weight = self.layers[self.num_layers - 1].get_weights()[0]
        if self.bias:
            bias = self.layers[self.num_layers - 1].get_weights()[1]
            bias_delta = self.lr * self.llayer_error.reshape(-1)
            updatedWeights = [weight +
                              delta_llast_layer.T +
                              self.momentum *
                              self.w_delta[self.num_layers -
                                           1].T, bias +
                              bias_delta +
                              self.momentum *
                              self.b_delta[self.num_layers -
                                           1]]
        else:
            updatedWeights = [weight +
                              (1 - self.momentum) * delta_llast_layer.T +
                              self.momentum * self.w_delta[self.num_layers - 1].T]

        self.layers[self.num_layers - 1].set_weights(updatedWeights)
        # Update momentum factor
        self.w_delta[self.num_layers - 1] = delta_llast_layer
        self.b_delta[self.num_layers - 1] = bias_delta

        # Note: The above steps are repeated for the two other weight
        # matrices.
        for layer_idx in range(self.num_layers - 2, -1, -1):
            if self.bias:
                self.llayer_error = np.dot(self.layers[layer_idx + 1].get_weights()[0],
                                           self.llayer_error) * np.mean(d_tanh(self.v[layer_idx]), axis=1, keepdims=True)

            if layer_idx == 0:
                delta_llast_layer = self.lr * \
                    np.dot(self.llayer_error, np.mean(
                        inp, axis=1, keepdims=True).T)
            else:
                delta_llast_layer = self.lr * \
                    np.dot(self.llayer_error, np.mean(
                        self.o[layer_idx - 1], axis=1, keepdims=True).T)

            weight = self.layers[layer_idx].get_weights()[0]
            if self.bias:
                bias = self.layers[layer_idx].get_weights()[1]
                bias_delta = self.lr * self.llayer_error.reshape(-1)
                updatedWeights = [
                    weight +
                    delta_llast_layer.T +
                    self.momentum *
                    self.w_delta[layer_idx].T,
                    bias +
                    bias_delta +
                    self.momentum *
                    self.b_delta[layer_idx]]
            else:
                updatedWeights = [weight +
                                  (1 -
                                   self.momentum) *
                                  delta_llast_layer.T +
                                  self.momentum *
                                  self.w_delta[layer_idx].T]

            self.layers[layer_idx].set_weights(updatedWeights)
            self.w_delta[layer_idx] = delta_llast_layer
            self.b_delta[layer_idx] = bias_delta

        del self.o
        del self.v


def init_weights_agent(m):
    """Initialize the weights and bias values of network layer

    Args:
        m: network layer
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        try:
            nn.init.xavier_uniform_(m.bias)
        except BaseException:
            a = 0


@torch.jit.script
def _objective(output, expected):
    """Compute MSE between output and expected value

    Args:
        output: Output from model
        expected: Expected output

    Returns:
        MSE error as tensor
    """
    return 0.5 * torch.pow(expected - output, 2)


class pytorch_network(nn.Module):
    """Fully Connected Neural Network"""

    def __init__(self,
                 learning_rate=1e-3,
                 momentum=0.9,
                 bias=None,
                 batch_size=1,
                 pop_size=None,
                 k_val=None,
                 cross_prob=None
                 ):
        """Intialize weights

        Args:
            learning_rate (Default=1e-5): Learning rate
            momentum (Default=0.9): Momentum factor
            bias (Defualt=None): Use bais value or not
            batch_size (Default=1): Batch Size
            pop_size (Default=None): Population Size for DE
            k_val (Default=None): K value for DE
            cross_prob (Default=None): Crossover probability for DE
        """
        super().__init__()

        # Parameters
        self.lr = learning_rate
        self.momentum = momentum
        self.bias = bias
        self.batch_size = batch_size
        self.num_layers = 0

        # DE parameters
        self.pop_size = pop_size
        self.k_val = k_val
        self.cross_prob = cross_prob

        # DE candidate values
        self.population = None          # Tensor of candidates
        self.pop_init = True            # flag for initializing candidates
        self.candidate_loss = torch.zeros(
            (pop_size, 1))   # track candidate loss

        # Network layers
        self.layers = nn.Sequential()

        # Auxilary matrix for storing momentum calculation
        self.w_delta = []
        self.b_delta = []

    def add_linear(self,
                   input_dim,
                   output_dim,
                   INPUT=False
                   ):
        """Add layer to the network

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
        """
        self.layers.add_module(str(self.num_layers), nn.Linear(
            input_dim, output_dim, bias=self.bias))
        self.layers.apply(init_weights_agent)

        # Weight delta for momentum calculation
        w_d = torch.zeros([output_dim, input_dim])
        if self.bias:
            b_d = torch.zeros(self.layers[self.num_layers].bias.shape)
            b_shape = self.layers[self.num_layers].bias.numel()

        # Initialize candidates
        if self.pop_init:
            self.population = nn.init.xavier_uniform_(torch.empty(
                self.pop_size, input_dim * output_dim + b_shape))
            self.pop_init = False
        else:
            self.population = torch.cat(
                (self.population,
                 nn.init.xavier_uniform_(
                     torch.empty(
                         self.pop_size,
                         input_dim *
                         output_dim +
                         b_shape))),
                dim=1)

        self.w_delta.append(w_d)
        self.b_delta.append(b_d)
        self.num_layers += 1

    def _set_weights_to_layers(self, candidate):
        """Set model weights and bias value as candidates value

        Args:
            candidate: Candidate tensor
        """
        last_idx = 0

        # Iterate over every layer
        for layer_idx in range(0, self.num_layers, 1):
            # Get layer dimensions
            w_shape = torch.tensor(self.layers[layer_idx].weight.shape)
            w_numel = torch.prod(w_shape).item()
            b_shape = torch.tensor(self.layers[layer_idx].bias.shape)
            b_numel = torch.prod(b_shape).item()

            # Decode the candidate and get weight, bias matrices
            weight = candidate[last_idx:last_idx +
                               w_numel].reshape(tuple(w_shape))
            last_idx += w_numel
            bias = candidate[last_idx:last_idx + b_numel].reshape(b_shape)
            last_idx += b_numel

            # Set layer weight, bias
            self.layers[layer_idx].weight = torch.nn.Parameter(weight)
            self.layers[layer_idx].bias = torch.nn.Parameter(bias)

    def _mutant(self, idx, F):
        """Generate Mutant vector and perform Crossover

        Args:
            idx: Index of candidate
            F: F value hyperparameter

        Returns:
            mutant: Generated mutant vector
        """
        # Generate random indices
        r = torch.randint(0, self.pop_size, (3,))
        # Re-generate if it contains candidate index
        while idx in r:
            r = torch.randint(0, self.pop_size, (3,))

        # Compute mutant
        mutant = torch.zeros(self.population[idx].shape)
        mutant = self.population[idx] + \
            self.k_val * (self.population[r[0]] - self.population[idx]) + \
            F * (self.population[r[1]] - self.population[r[2]])

        # Crossover
        probs = torch.randn(mutant.shape)
        return torch.where(
            probs >= self.cross_prob,
            mutant,
            self.population[idx])

    def forward_de(self, x):
        """Forward pass on Neural Network candidates

        Args:
            x: Input matrix

        Returns:
            output: Output matrix,
                    resulting shape (pop_size, batch_size, output_size)
        """
        x = torch.tensor(x)
        y = []

        # Iterate over candidates
        for idx, candidate in enumerate(self.population):
            self._set_weights_to_layers(candidate)
            xx = self.forward_bp(x)
            y.append(xx.numpy())
        self.y = torch.tensor(y)

        return y

    def backwards_de(self, input_, expected):
        """Backwards pass on Neural Network using Differential Evolution

        Args:
            input_: Input matrix
            expected: Expected value matrix
        """
        input_ = torch.tensor(input_)
        expected = torch.tensor(expected)

        with torch.no_grad():
            F = torch.FloatTensor(1).uniform_(-2, 2)
            gen_loss = torch.zeros((self.pop_size, 1))

            # Iterate over candidates
            for idx, _ in enumerate(self.population):
                # Compute loss of candidate
                loss = torch.mean(_objective(self.y[idx], expected))

                # Perform DE, if the generated weights perform worse, recompute
                # till it performs better than existing candidate
                re = 0
                while re == 0:
                    trial = self._mutant(idx, F)

                    self._set_weights_to_layers(trial)
                    trial_output = self.predict(input_)
                    trial_loss = torch.mean(_objective(trial_output, expected))
                    #print(trial_loss, loss)

                    if trial_loss <= loss:
                        self.population[idx] = trial
                        gen_loss[idx][0] = trial_loss
                        re = 1

        # Set the model weight to the best performing candidate
        min_in_gen, min_weight_idx = torch.min(gen_loss, dim=0)
        self._set_weights_to_layers(self.population[min_weight_idx.item()])

        # Save the candidates loss history
        self.candidate_loss = torch.cat((self.candidate_loss, gen_loss), dim=1)

    def forward_bp(self, x):
        """Forward pass on Neural Network
        Args:
            x: Input matrix
        Returns:
            output: Output matrix
        """
        x = torch.tensor(x)

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

    def backwards_bp(self, inp, expected):
        """Backwards pass on Neural Network using Backpropogation

        Args:
            inp: Input matrix
            expected: Expected value matrix
        """
        inp = torch.tensor(inp)
        expected = torch.tensor(expected)

        with torch.no_grad():

            # Compute error at last layer
            self.o_error = torch.mean(expected - self.o[self.num_layers - 1],
                                      dim=0, keepdims=True)

            # Note: Refer backpropogation slides for full details on
            #       backprop equation.
            # TODO: Backpropogation through multiple layers in a loop.

            # Pass output loss through derivative of activation function.
            # In this case, Tanh.
            self.llayer_error = self.o_error * \
                torch.mean(d_tanh(self.v[self.num_layers - 1]),
                           dim=0, keepdims=True)

            # Compute delta of weights in layer. Scale delta by a factor of
            # learning rate.
            delta_llast_layer = self.lr * \
                torch.matmul(self.llayer_error,
                             torch.mean(self.o[self.num_layers - 2],
                                        dim=0, keepdims=True))

            # Add delta and momentum factor for improved stability to
            # weight matrix.
            self.layers[self.num_layers - 1].weight += \
                1 * delta_llast_layer + \
                self.momentum * self.w_delta[self.num_layers - 1]
            if self.bias:
                self.layers[self.num_layers - 1].bias += \
                    self.lr * self.llayer_error.reshape(-1) + \
                    self.momentum * self.b_delta[self.num_layers - 1]
            # Update momentum factor
            self.w_delta[self.num_layers - 1] = delta_llast_layer
            self.b_delta[self.num_layers - 1] = self.lr * \
                self.llayer_error.reshape(-1)

            # Note: The above steps are repeated for the two other weight
            # matrices.
            for layer_idx in range(self.num_layers - 2, -1, -1):
                if self.bias:
                    self.llayer_error = torch.matmul(self.llayer_error,
                                                     self.layers[layer_idx + 1].weight) * \
                        torch.mean(d_tanh(self.v[layer_idx]),
                                   dim=0, keepdims=True)

                if layer_idx == 0:
                    delta_llast_layer = self.lr * \
                        torch.matmul(self.llayer_error.T,
                                     torch.mean(inp, dim=0, keepdims=True))
                else:
                    delta_llast_layer = self.lr * \
                        torch.matmul(self.llayer_error.T,
                                     torch.mean(self.o[layer_idx - 1], dim=0, keepdims=True))

                self.layers[layer_idx].weight += 1 * delta_llast_layer + \
                    self.momentum * self.w_delta[layer_idx]
                if self.bias:
                    self.layers[layer_idx].bias += self.lr * self.llayer_error.reshape(-1) + \
                        self.momentum * self.b_delta[layer_idx]

                self.w_delta[layer_idx] = delta_llast_layer
                self.b_delta[layer_idx] = self.lr * \
                    self.llayer_error.reshape(-1)

            del self.o
            del self.v

    def predict(self, x):
        """Forward pass on Neural Network for prediction

        Args:
            x: Input matrix

        Returns:
            output: Output matrix
        """
        x = torch.tensor(x)

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
