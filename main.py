from nn.network import ff_network

import numpy as np
import copy
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

def scale_linear_bycolumn(rawpoints,
			high=1.0,
			low=-1.0
			):
	"""Normalize the dataset.

	Args:
		rawpoints: Dataset
		high: Upper bound on normalized value
		low: Lower bound on normalized value

	Returns:
		dataset: Normalized dataset
		rng: Vector consisting of ranges for each column
		mins: Vector consisting of least values for each column
		maxs: Vector consisting of larges values for each column
	"""
	mins = np.min(rawpoints, axis=0)
	maxs = np.max(rawpoints, axis=0)
	rng = maxs - mins
	return high - (((high - low)*(maxs - rawpoints))/ rng), rng, mins, maxs

def de_normalize(output,
		maxs,
		rng
		):
	"""De-Normalize the value

	Args:
		output: Output from the model
		maxs: Vector consisting of largest values for each column
		rng: Vector consisting of ranges for each column

	Returns:
		value: De-normalized value
	"""
	return maxs[-1] - (((1.0 - output[0][0]) * rng[-1])/2)

def training_data(file_path):
	"""Read and Normalize training data.

	Args:
		file_path: path to dataset file

	Returns:
		dataset: Normalized dataset
		rng: Vector consisting of ranges for each column
		mins: Vector consisting of least values for each column
		maxs: Vector consisting of largest values for each column
		original_dataset: Original dataset
	"""
	my_data = np.genfromtxt(file_path, delimiter='\t')
	dataset, rng, mins, maxs = scale_linear_bycolumn(my_data)
	return dataset, rng, mins, maxs, my_data

def split(dataset, test_split=0.2):
	"""Split the dataset into training, validation and test.

	Args:
		dataset: Dataset
		test_split: Ratio of test

	Returns:
		tr_idx: Array of indices for Training
		test_idx: Array of indices for Testing
	"""
	t_idx = np.random.choice(len(dataset), len(dataset))
	test_idx = t_idx[:int(len(dataset)*test_split)]
	tr_idx = t_idx[int(len(dataset)*test_split):]
	return tr_idx, test_idx

def train_network(dataset,
			train_idx,
			test_idx,
			model,
			input_size,
			epoch=100,
			batch_size=1
			):
	"""Training loop.

	Args:
		dataset: Dataset
		train_idx: Array of indices for Training
		test_idx: Array of indices for Validation(?)
		model: Neural Network object
		input_size: Input dimension
		epoch (Default=100): Number of epochs
		batch_size (Default=1): Batch Size

	Returns:
		loss_history: Array of training loss
	"""
	loss_history = []
	for i in range(EPOCH):
		for k in range(0, len(train_idx), batch_size):
			end = min(k+batch_size, len(train_idx))
			output = model.forward(dataset[k:end, :INPUT])
			model.backwards(dataset[k:end, :INPUT], \
					dataset[k:end, INPUT])

		loss = 0
		np.random.shuffle(train_idx)

		for _, val in enumerate(test_idx):
			output = model.eval(dataset[val, :INPUT])
			loss += np.square(dataset[val, INPUT] - output)*0.5
		loss /= len(train_idx)
		loss_history.append(loss[0])

		print("Epoch: ", i," Validation Loss: ", loss[0])

	np.savetxt('loss_history.csv', loss_history, delimiter=',')
	return loss_history

def scores(data, model, test_idx, input_size):
	"""Compute MAPE and RSQ.

	Args:
		data: Original Dataset
		model: Neural Network object
		test_idx: Array of Test indices
		input_size: Input dimension
	"""
	norm_data = copy.deepcopy(data)
	norm_data, rng, _, maxs = scale_linear_bycolumn(norm_data)

	val_out = np.zeros((len(test_idx), 1), dtype=np.float32)

	mape = 0
	cnt = 0
	save_this = []
	for i, j in enumerate(test_idx):
		output = nn.eval(norm_data[j, :INPUT])
		val_out[i, 0] = output

		output = de_normalize(output, maxs, rng)
		save_this.append([data[j, INPUT], output])

		if abs(dataset[j, INPUT]) > 0.01:
			mape += abs(output - data[j,INPUT])/ \
					abs(data[j,INPUT])
			cnt += 1

	save_this = np.asarray(save_this)
	rsq = np.corrcoef(save_this[:, 0], save_this[:, 1])
	print("RSQ: ", rsq[0, 1]*rsq[0, 1])

	np.savetxt('output_vals.csv', save_this, delimiter=',')
	print("MAPE: ", (mape*100)/cnt)

if __name__ == '__main__':
	# Neural Network Parameters
	INPUT = 13
	OUTPUT = 1

	# Training Parameters
	EPOCH = 400
	BATCH_SIZE = 1

	# Read dataset and normalize
	dataset, rng, mins, maxs, data = \
		training_data('../dataset/mpp_dataset_v1_13-inputs.txt')

	# Initialize Neural Network
	nn = ff_network(bias=True, batch_size=BATCH_SIZE)

	# Add layers
	nn.add_linear(INPUT, 50)
	nn.add_linear(50, 50)
	nn.add_linear(50, OUTPUT)

	# Generate test and train indices arrays
	train_idx, test_idx = split(dataset)

	# Train
	loss_history = train_network(dataset, train_idx, test_idx, nn, \
				INPUT, epoch=EPOCH, batch_size=BATCH_SIZE)

	# Compute MAPE and RSQ value
	scores(data, nn, test_idx, INPUT)

	# Plot loss history
	plt.scatter(np.arange(len(loss_history)), loss_history, alpha=0.5)
	plt.show()