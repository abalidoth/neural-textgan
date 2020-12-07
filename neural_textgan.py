import numpy as np
from tqdm import tqdm
from random import choices
from copy import deepcopy

def clamped_relu(x):
    return np.clip(x,0,6)

def mutate_array(i, prob, strength):
    i += np.random.normal(0,strength,i.shape)*(np.random.random(i.shape)<prob)

class GeneratorAgent:
    def __init__(
        self,
        alphabet_size,
        hidden_layer_sizes,
        activation = np.arctan
    ):
        self.alphabet_size=alphabet_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.recurrent_layer = np.random.random(hidden_layer_sizes[-1])
        self.hidden_layers = []
        self.biases = []
        self.activation = activation
        all_sizes = [self.hidden_layer_sizes[-1]] + self.hidden_layer_sizes + [self.alphabet_size]
        for i, j in zip(all_sizes[:-1], all_sizes[1:]):
            self.hidden_layers.append(np.random.normal(0,1,[i,j]))
            self.biases.append(np.random.normal(0,1,[j]))
            
    def generate(self, length, ):
        out_vec = []
        out_pred =  []
        self.recurrent_layer = np.random.random(self.recurrent_layer.shape)
        for i in range(length):
            work = self.recurrent_layer
            for i,j in zip(self.hidden_layers, self.biases):
                old_work, work = work, self.activation(i@work + j)
            self.recurrent_layer = old_work.copy()
            out_vec.append(work)
            out_pred.append(np.argmax(work))
        return out_vec, out_pred

    def mutate(self, prob = 0.1, strength = 1):
        out = deepcopy(self)
        for i in out.hidden_layers:
            mutate_array(i, prob, strength)
        for i in out.biases:
            mutate_array(i, prob, strength)
        return out


class DiscriminatorAgent:
    def __init__(
        self,
        alphabet_size,
        hidden_layer_sizes,
        activation = clamped_relu
    ):
        self.alphabet_size=alphabet_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.hidden_layers = []
        self.biases = []
        self.activation = activation
        all_sizes = [self.hidden_layer_sizes[-1]+self.alphabet_size] + self.hidden_layer_sizes
        for i, j in zip(all_sizes[:-1], all_sizes[1:]):
            self.hidden_layers.append(np.random.normal(0,1,[i,j]))
            self.biases.append(np.random.normal(0,1,[j]))
        self.determiner = np.random.normal(0,1,[self.hidden_layer_sizes[-1]])
        self.determiner_bias = np.random.normal(0,1,[1])
            
    def bulk_discriminate(self, text_matrix):
        """text_matrix is integers in shape [num_samples, length]"""
        text_blocks = \
            np.arange(self.alphabet_size)[:, np.newaxis, np.newaxis]==\
            text_matrix[np.newaxis, :, :]
        
        recurrent_layer = np.zeros([self.hidden_layer_sizes[-1], text_blocks.shape[1]])

        for i in range(text_blocks.shape[2]):
            work = np.concatenate(
                [text_blocks[:,:,i],recurrent_layer],
                axis=0
            )
            for hid, bias in zip(self.hidden_layers, self.biases):
                temp = np.tensordot(
                    hid,
                    work,
                    axes=((0),(0))
                )
                work = self.activation(temp + bias[:,np.newaxis])
            recurrent_layer = work
        return np.tensordot(work, self.determiner, ((0),(0))) + self.determiner_bias

    def mutate(self, prob = 0.1, strength = 1):
        out = deepcopy(self)
        for i in out.hidden_layers:
            mutate_array(i, prob, strength)
        for i in out.biases:
            mutate_array(i, prob, strength)
        mutate_array(out.determiner, prob, strength)
        mutate_array(out.determiner_bias, prob, strength)
        return out