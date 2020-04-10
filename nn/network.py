import torch
from torch import nn

torch.set_printoptions(precision=4)
torch.set_default_dtype(torch.float32)
torch.Generator().manual_seed(2147483647)
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
                 bias=None,
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

        self.bias = bias
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
        if self.bias:
            b_shape = self.layers[self.num_layers].bias.numel()
        else :
            b_shape = 0

        # Initialize candidates
        if INPUT :
            self.population = nn.init.xavier_uniform_(torch.empty(
                self.pop_size, input_dim * output_dim + b_shape))
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
            if self.bias :
                b_shape = torch.tensor(self.layers[layer_idx].bias.shape)
                b_numel = torch.prod(b_shape).item()

            # Decode the candidate and get weight, bias matrices
            weight = candidate[last_idx:last_idx +
                               w_numel].reshape(tuple(w_shape))
            last_idx += w_numel
            if self.bias :
                bias = candidate[last_idx:last_idx + b_numel].reshape(b_shape)
                last_idx += b_numel

            # Set layer weight, bias
            self.layers[layer_idx].weight = torch.nn.Parameter(weight)
            if self.bias :
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
            probs >= self.cross_prob,self.population[idx],
            mutant)


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
                vec_output = self.predict(input_)
                loss = torch.mean(_objective(vec_output, expected))

                # Perform DE, if the generated weights perform worse, recompute
                # till it performs better than existing candidate
                re = 0
                while re == 0:
                    trial = self._mutant(idx, F)

                    self._set_weights_to_layers(trial)
                    vec_output = self.predict(input_)
                    trial_loss = torch.mean(_objective(vec_output, expected))
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
