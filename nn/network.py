import torch
from torch import nn

torch.set_printoptions(precision=4)
torch.set_default_dtype(torch.float32)
print("Initial Seed: ", torch.Generator().initial_seed())


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
            None


class pytorch_network(nn.Module):
    """Fully Connected Neural Network"""

    def __init__(self,
                bias=None,
                ):
        """Intialize weights

        Args:
            bias (Defualt=None): Use bais value or not
        """
        super().__init__()

        # Parameters
        self.bias = bias
        self.num_layers = 0

        # Network layers
        self.layers = nn.Sequential()

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
            if self.bias:
                b_shape = torch.tensor(self.layers[layer_idx].bias.shape)
                b_numel = torch.prod(b_shape).item()

            # Decode the candidate and get weight, bias matrices
            weight = candidate[last_idx:last_idx +
                               w_numel].reshape(tuple(w_shape))
            last_idx += w_numel
            if self.bias:
                bias = candidate[last_idx:last_idx + b_numel].reshape(b_shape)
                last_idx += b_numel

            # Set layer weight, bias
            self.layers[layer_idx].weight = torch.nn.Parameter(weight)
            if self.bias:
                self.layers[layer_idx].bias = torch.nn.Parameter(bias)


    def predict(self, x):
        """Forward pass on Neural Network for prediction

        Args:
            x: Input matrix

        Returns:
            output: Output matrix
        """
        x = torch.tensor(x)

        with torch.no_grad():
            # Iterate over every layer and forward pass
            for layer_idx in range(self.num_layers):
                weight = self.layers[layer_idx].weight
                v = torch.matmul(x, weight.T)
                if self.bias:
                    bias = self.layers[layer_idx].bias
                    v += bias.T
                x = torch.tanh(v)

        return x
