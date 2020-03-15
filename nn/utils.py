import numpy as np

def tanh(x):
	"""Tanh Activation Function"""
	return np.tanh(x)

def d_tanh(x):
	"""Derivative of Tanh Activation Function"""
	return 1.0 - np.tanh(x)**2

def sigmoid(s):
	"""Sigmoid Activation Function"""
	return 1/(1+np.exp(-s))

def d_sigmoid(s):
	"""Derivative of Sigmoid Activation Function"""
	return s * (1.0 - s)