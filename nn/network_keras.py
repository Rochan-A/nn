import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np




def _objective(output, expected):
    """Compute MSE between output and expected value
    Args:
        output: Output from model
        expected: Expected output
    Returns:
        MSE error as tensor
    """
    return 0.5 * np.power(expected - output, 2)



class keras_network(Sequential):
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
        self.candidate_loss = np.zeros(
            (pop_size, 1))   # track candidate loss


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
            self.add(Dense(output_dim = output_dim ,\
                             use_bias= self.bias,\
                             kernel_initializer= keras.initializers.lecun_normal(seed=None),\
                             bias_initializer= keras.initializers.Constant(0.1),\
                             activation = 'tanh', input_dim = input_dim))
        else :
            self.add(Dense(output_dim = output_dim,use_bias= self.bias,\
                                 kernel_initializer= keras.initializers.lecun_normal(seed=None)\
                                 ,bias_initializer= keras.initializers.Constant(0.1), activation = 'tanh'))

        
        if self.bias:
            b_shape = np.shape(self.layers[self.num_layers].get_weights()[1])[0]
        else :
            b_shape = 0

        
        # Initialize candidates
        if INPUT:
            self.population = np.random.rand(
                self.pop_size, input_dim * output_dim + b_shape)-0.5
        else:
            self.population = np.concatenate(
                (self.population,np.random.rand(
                self.pop_size, input_dim * output_dim + b_shape)-0.5),
                axis =1)
                
       # updating number of layers
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
            w_shape = np.shape(self.layers[layer_idx].get_weights()[0])
            w_numel = np.prod(w_shape)
            if self.bias :
                b_shape = np.shape(self.layers[layer_idx].get_weights()[1])
                b_numel = np.prod(b_shape)

            # Decode the candidate and get weight, bias matrices
            weight = candidate[last_idx:last_idx +
                               w_numel].reshape(w_shape)
            last_idx += w_numel
            if self.bias :
                bias = candidate[last_idx:last_idx + b_numel].reshape(b_shape)
                last_idx += b_numel
            else :
                bias = []

            # Set layer weight, bias
            if self.bias :
                self.layers[layer_idx].set_weights([weight,bias])
            else :
                self.layers[layer_idx].set_weights([weight])
            
            
    def _mutant(self, idx, F):
        """Generate Mutant vector and perform Crossover
        Args:
            idx: Index of candidate
            F: F value hyperparameter
        Returns:
            mutant: Generated mutant vector
        """
        # Generate random indices
        r = np.random.randint(0, self.pop_size, (3,))
        # Re-generate if it contains candidate index
        while idx in r:
            r = np.random.randint(0, self.pop_size, (3,))

        # Compute mutant
        mutant = np.zeros(self.population[idx].shape)
        mutant = self.population[idx] + \
            self.k_val * (self.population[r[0]] - self.population[idx]) + \
            F * (self.population[r[1]] - self.population[r[2]])

        # Crossover
        probs = np.random.rand(mutant.shape[0])
        return np.where(
            probs >= self.cross_prob,
            mutant,
            self.population[idx])
        

    def backwards_de(self, input_, expected):
        """Backwards pass on Neural Network using Differential Evolution
        Args:
            input_: Input matrix
            expected: Expected value matrix
        """

        
        F = np.random.uniform(-2,2)
        gen_loss = np.zeros((self.pop_size, 1))

        # Iterate over candidates
        for idx, _ in enumerate(self.population):
            # Compute loss of candidate
            self._set_weights_to_layers(self.population[idx])
            vec_output = self.predict(input_)
            loss = np.mean(_objective(vec_output, expected))

            # Perform DE, if the generated weights perform worse, recompute
            # till it performs better than existing candidate
            #re = 0
            #while re == 0:
            trial = self._mutant(idx, F)

            self._set_weights_to_layers(trial)
            vec_output = self.predict(input_)
            trial_loss = np.mean(_objective(vec_output, expected))
                #print(trial_loss, loss)
                
            if trial_loss <= loss:
                self.population[idx] = trial
                gen_loss[idx][0] = trial_loss
            #        re = 1
            else :
                gen_loss[idx][0] = loss
                        

        # Set the model weight to the best performing candidate
        min_weight_idx = np.argmin(gen_loss)
        self._set_weights_to_layers(self.population[min_weight_idx])

        # Save the candidates loss history
        self.candidate_loss = np.concatenate((self.candidate_loss, gen_loss), axis = 1)












