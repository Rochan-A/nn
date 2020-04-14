from nn.network import pytorch_network

import torch
from torch import nn
import torch.multiprocessing as mp

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

class de_evo():
    """Differential Evolution Class. Fully GPU parallizable"""
    def __init__(self,
                pop_size=None,
                k=None,
                cross_prob=None,
                bias=None
                ):
        """Initialize Differential Evolution parameters

        Args:
            pop_size (Default=None): Population Size for DE
            k_val (Default=None): K value for DE
            cross_prob (Default=None): Crossover probability for DE
            bias (Defualt=None): Use bais value or not
        """
        super().__init__()

        # DE parameters
        self.pop_size = pop_size
        self.k_val = k
        self.cross_prob = cross_prob

        # Parameters
        self.bias = bias
        self.device = None
        self.d_count = None

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.d_count = torch.cuda.device_count()
        else:
            self.device = torch.device('cpu')

        # DE candidate values
        self.candidate_loss = torch.zeros(
            (pop_size, 1), device=self.device)

        # Store networks in array, assign to device
        self.candidateVec = []
        if torch.cuda.is_available():
            for gpu_idx in range(self.d_count):
                with torch.cuda.device(gpu_idx):
                    for idx in range(gpu_idx, self.pop_size, self.d_count):
                        device_idx = 'cuda:' + str(gpu_idx)
                        self.candidateVec.append(pytorch_network(bias=self.bias,
                                                                 device=device_idx,
                                                                 idx=idx
                                                                 ).to(device=device_idx))
        else:
            self.candidateVec = [
                pytorch_network(bias=self.bias).to(device=self.device)
                for i in range(self.pop_size)
            ]


    def add_linear(self,
                   input_dim,
                   output_dim,
                   inp_layer=False
                   ):
        """Add layer to the candidate networks

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            inp_layer: flag for input layer
        """

        for i in range(self.pop_size):
            self.candidateVec[i].add_linear(input_dim, output_dim, inp_layer)

    def forward(self, x):
        """Forward pass on candidates in parallel

        Args:
            x: Input tensor
        """

        # Convert input to tensor, broadcast to all available GPU's
        x = torch.tensor(x)
        x = torch.cuda.comm.broadcast(x, [i for i in range(self.d_count)])

        # Use torch.multiprocessing support for CUDA, Queue to pass outputs
        jobs = []
        ctx = mp.get_context('spawn')
        queue = ctx.Queue()

        # Start a process for each candidate
        for idx in range(self.pop_size):
            p = ctx.Process(target=self.candidateVec[idx].forward, args=(x, queue))
            jobs.append(p)
            p.start()

        # Read outputs from Queue
        res = []
        for idx in range(self.pop_size):
            res.append(queue.get())
        res = res.sort(key = lambda x: x[0])

        # Concatenate into single tensor
        out_tensor = res[0][1]
        for idx in range(self.pop_size - 1):
            out_tensor = torch.cat([out_tensor, res[idx][0]], dim=0)

        # Wait for processes to end
        for proc in jobs:
            proc.join()

        print(out_tensor.numpy().shape)

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
            self.population[idx],
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
