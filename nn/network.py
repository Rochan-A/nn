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

		# Weights
		self.weights = []
		# Auxilary matrix for storing momentum calculation
		self.w_delta = []

	def add_linear(self,
			input_dim,
			output_dim
			):
		"""Add layer to the network

		Args:
			input_dim: Input dimension
			output_dim: Output dimension
		"""
		bias_node = 1 if self.bias else 0

		# Initialize weight matrices
		rad = 1 / np.sqrt(input_dim + bias_node)
		X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
		w = X.rvs((output_dim,
			input_dim + bias_node))

		# Weight delta for momentum calculation
		w_d = np.zeros((output_dim,
			input_dim + bias_node))

		self.weights.append(w)
		self.w_delta.append(w_d)
		self.num_layers += 1

	def forward(self, x):
		"""Forward pass on Neural Network

		Args:
			x: Input matrix

		Returns:
			output: Output matrix
		"""
		# If bias is present, concatenate vector of 1's
		if self.bias:
			x = np.concatenate((x,
				np.ones((x.shape[0],1))), axis=1)

		x = np.array(x, ndmin=2).T

		self.v = []
		self.o = []

		# Iterate over every layer and forward pass, store v and o
		# values.
		for layer_idx in range(self.num_layers):
			v = np.dot(self.weights[layer_idx], x)
			x = tanh(v)

			if self.bias and layer_idx != self.num_layers - 1:
				x = np.concatenate((x,
						np.ones((1, x.shape[1]))), axis=0)

			self.v.append(v)
			self.o.append(x)

		return x

	def backwards(self, inp, expected):
		"""Forward pass on Neural Network

		Args:
			inp: Input matrix
			expected: Expected value matrix
		"""
		# If bias is present, concatenate vector of 1's
		if self.bias:
			inp = np.concatenate((inp,
					np.ones((inp.shape[0],1))), axis=1)

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
				d_tanh(np.mean(self.v[self.num_layers-1],
						axis=1, keepdims=True))

		# Compute delta of weights in layer. Scale delta by a factor of
		# learning rate.
		delta_llast_layer = self.lr * \
				np.dot(self.llayer_error,
					np.mean(self.o[self.num_layers-2],
						axis=1, keepdims=True).T)

		# Add delta and momentum factor for improved stability to
		# weight matrix.
		self.weights[self.num_layers-1] += \
				(1 - self.momentum)*delta_llast_layer + \
				self.momentum * self.w_delta[self.num_layers-1]

		# Update momentum factor
		self.w_delta[self.num_layers-1] = delta_llast_layer

		# Note: The above steps are repeated for the two other weight
		# matrices.
		for layer_idx in range(self.num_layers-2, -1, -1):
			if self.bias:
				self.llayer_error = np.dot(self.weights[layer_idx+1][:,:-1].T, self.llayer_error) * \
							d_tanh(np.mean(self.v[layer_idx], axis = 1, keepdims = True))
			else :
				self.llayer_error = np.dot(self.weights[layer_idx+1].T, self.llayer_error) * \
							d_tanh(np.mean(self.v[layer_idx], axis = 1, keepdims = True))

			if layer_idx == 0:
				delta_llast_layer = self.lr * \
						np.dot(self.llayer_error, np.mean(inp, axis = 1, keepdims = True).T)
			else:
				delta_llast_layer = self.lr * \
						np.dot(self.llayer_error, np.mean(self.o[layer_idx-1], axis = 1, keepdims= True).T)

			self.weights[layer_idx] += (1 - self.momentum) * delta_llast_layer + \
							self.momentum * self.w_delta[layer_idx]
			self.w_delta[layer_idx] = delta_llast_layer
			
		del self.o
		del self.v

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

		for layer_idx in range(self.num_layers):
			v = np.dot(self.weights[layer_idx], x)
			x = tanh(v)

			if self.bias and layer_idx != self.num_layers - 1:
				x = np.concatenate((x,
					np.ones((1, x.shape[1]))), axis=0)

		return x
