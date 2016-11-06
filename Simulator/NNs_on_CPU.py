import os, sys, time

import ReCAM, Simulator
from  NeuralNetwork import NeuralNetwork
from NumberFormats import FixedPoint
import random, copy, math


'''lib_path = os.path.abspath(os.path.join('swalign-0.3.3'))
sys.path.append(lib_path)
import swalign'''


''' Define a NN structure and the appropriate column to store.
Every layer has number of neurons - the net will be presented as a list.
First (last) layer is input (output) layer.
'''


################################################
####        AUX functions & definitions
################################################
max_operation_string = "max"

def check_if_all_are_zero(list):
    if all(v == 0.0 for v in list):
        return True
    return False

def check_if_any_is_zero(list):
    if any(v == 0.0 for v in list):
        return True
    return False

def convert_if_needed(result, number_format=None):
    if number_format:
        return number_format.convert(result)
    return result

def convert_to_non_zero_if_needed(result, number_format=None):
    if number_format:
        return number_format.convert_to_non_zero(result)
    return result

def listWithListOperation(list_A, list_B, res_list, operation, number_format=None):
    if operation=='-':
        for i in range(len(list_A)):
            res_list[i] = convert_if_needed(list_A[i] - list_B[i], number_format)
    elif operation == '+':
        for i in range(len(list_A)):
            res_list[i] = convert_if_needed(list_A[i] + list_B[i], number_format)
    elif operation == '*':
        for i in range(len(list_A)):
            res_list[i] = convert_if_needed(list_A[i] * list_B[i], number_format)


def listWithScalarOperation(scalar, list_A, res_list, operation, number_format=None):
    if scalar == 0:
        converted_scalar = convert_if_needed(scalar, number_format)
    else:
        converted_scalar = convert_to_non_zero_if_needed(scalar, number_format)

    if operation == '*':
        for i in range(len(list_A)):
            res_list[i] = convert_if_needed(converted_scalar * list_A[i], number_format)
    elif operation == '+':
        for i in range(len(list_A)):
            res_list[i] = convert_if_needed(converted_scalar + list_A[i], number_format)

################################################
####        ReLU Function
################################################

def ReLU_activation(input):
    return max(0, input)

def ReLU_activation_on_list(input, output):
    for i in range(len(input)):
        output[i] = max(0, input[i])

def ReLU_derivative(activation, numbers_format):
    result = 1 if activation >= 0 else 0
    return convert_if_needed(result, numbers_format)

def ReLU_derivative_on_list(activation, derivative, numbers_format):
    for i in range(len(activation)):
        derivative[i] = 0 if activation[i] == 0 else derivative[i]


################################################
####        softmax Function
################################################
def softmax_derivative(activation, target, numbers_format):
    return convert_if_needed(activation - target ,numbers_format)

def softmax_derivative_on_list(activation, target, derivative, numbers_format):
    for i in range(len(activation)):
        derivative[i] = convert_if_needed(activation[i] - target[i] ,numbers_format)


################################################
####        NN & List Operations
################################################

def initialize_NN_on_CPU(numbers_format=None):
    NN_on_CPU = CPU_NN_Manager(numbers_format)

    return NN_on_CPU

class CPU_NN_Manager:
    def __init__(self, numbers_format=None):
        self.numbersFormat = numbers_format
        self.SGD_mini_batch_size = 0
        self.learning_rate = 0.01
        self.SGD_weights = []
        self.samples_trained = 0
        self.epochs = 0
        self.samples_in_dataset = 0

    def set_learning_rate(self, new_learning_rate):
        self.learning_rate = new_learning_rate

    def set_SGD_parameters(self, nn, mini_batch_size, new_learning_rate):
        self.SGD_mini_batch_size = mini_batch_size
        self.learning_rate = new_learning_rate
        self.SGD_weights = copy.deepcopy(nn.weightsMatrices)

    ############################################################
    ######  Feedforward an input through the net
    ############################################################
    def feedforward(self, nn, input):
        input.append(1)

        num_of_net_layers = len(nn.layers)
        activations = []
        activations.append(copy.deepcopy(input))

        for layer_index in range(1, num_of_net_layers):
            activations.append([])
            neurons_in_layer = len(nn.weightsMatrices[layer_index])
            weights_per_neuron = len(nn.weightsMatrices[layer_index][0])
            layer_type = nn.layers[layer_index][0]

            for neuron in range(neurons_in_layer):
                weighted_sum = 0
                for weight in range(weights_per_neuron):
                    mul_result = convert_if_needed(activations[layer_index-1][weight] * nn.weightsMatrices[layer_index][neuron][weight], nn.numbersFormat)
                    weighted_sum = convert_if_needed(weighted_sum + mul_result, nn.numbersFormat)
                    #print("Working on layer {}, neuron {}, weight {}".format(layer_index, neuron, weight))

                if layer_type == "FC":
                    weighted_sum = max(0, weighted_sum)
                activations[layer_index].append(weighted_sum)

            if layer_index!=num_of_net_layers-1:
                activations[layer_index].append(1) #bias exists only in input+hidden layers

        if nn.layers[num_of_net_layers-1][0] == "softmax":
            last_layer_neurons_num = nn.layers[num_of_net_layers - 1][1]
            max_output = max(activations[num_of_net_layers - 1])

            # 1. Calculate sum of exponents with (activation - max) as power
            # 2. For each activation, calculate its softmax = exp(activation-max)/exponents_sum
            sum_of_exponents = 0
            for output_index in range(last_layer_neurons_num):
                sum_of_exponents += math.exp(activations[num_of_net_layers-1][output_index] - max_output)

            for output_index in range(last_layer_neurons_num):
                biased_neuron_activation = activations[num_of_net_layers-1][output_index] - max_output
                activations[num_of_net_layers-1][output_index] = convert_if_needed(math.exp(biased_neuron_activation) / sum_of_exponents, nn.numbersFormat)

        #print(net_output)
        input.pop()
        return activations

    ############################################################
    ######  Backward propagation of an output through the net
    ############################################################
    def backPropagation(self, nn, activations, target):
        num_of_net_layers = len(nn.layers)
        partial_derivatives = [None]
        deltas = [[] for x in range(len(nn.layers))] #DEBUG
        deltas[0] = None #DEBUG

        curr_delta = None
        prev_delta = None
        for layer_index in range(num_of_net_layers-1):
            partial_derivatives.append([])

        for layer_index in range(num_of_net_layers-1, 0, -1):
            layer_type = nn.layers[layer_index][0]
            neurons_in_layer = len(nn.weightsMatrices[layer_index])
            weights_per_neuron = len(nn.weightsMatrices[layer_index][0])

            # Deltas of output layer
            if layer_index == num_of_net_layers-1:
                if layer_type == "softmax":
                    curr_delta = [0] * len(activations[num_of_net_layers - 1])
                    softmax_derivative_on_list(activations[num_of_net_layers - 1], target, curr_delta, nn.numbersFormat)
                    deltas[layer_index] = curr_delta  # DEBUG.
                elif layer_type == "FC":
                    curr_delta = [0] * len(activations[num_of_net_layers - 1])
                    listWithListOperation(activations[num_of_net_layers - 1], target, curr_delta, '-', nn.numbersFormat)
                    ReLU_derivative_on_list(activations[num_of_net_layers - 1], curr_delta, nn.numbersFormat)
                    deltas[layer_index] = curr_delta  # DEBUG.

            # Deltas of hidden layers
            else:
                if layer_type == "FC":
                    neurons_in_prev_bp_layer = len(nn.weightsMatrices[layer_index+1])
                    weights_in_prev_bp_layer_neuron = len(nn.weightsMatrices[layer_index+1][0])

                    curr_delta = [0] * weights_in_prev_bp_layer_neuron
                    for neuron_in_prev_bp_index in reversed(range(neurons_in_prev_bp_layer)):
                        temp_delta = [0] * weights_in_prev_bp_layer_neuron
                        listWithScalarOperation(prev_delta[neuron_in_prev_bp_index], nn.weightsMatrices[layer_index+1][neuron_in_prev_bp_index], temp_delta, '*', nn.numbersFormat)
                        listWithListOperation(temp_delta, curr_delta, curr_delta, '+', nn.numbersFormat)

                    ReLU_derivative_on_list(activations[layer_index], curr_delta, nn.numbersFormat)
                    deltas[layer_index] = curr_delta[:-1]  # DEBUG

            for neuron_index in range(neurons_in_layer):
                neuron_pds = [0] * weights_per_neuron
                listWithScalarOperation(curr_delta[neuron_index], activations[layer_index-1], neuron_pds, '*', nn.numbersFormat)
                partial_derivatives[layer_index].append(neuron_pds)

            prev_delta = curr_delta

        ##return partial_derivatives
        return (partial_derivatives, deltas) #DEBUG

    ############################################################
    ######  Backward propagation of an output through the net
    ############################################################
    def SGD_train(self, nn, NN_input, target_output):
        activations = self.feedforward(nn, NN_input)
        ##partial_derivatives = self.backPropagation(nn, activations, target_output)
        (partial_derivatives, deltas)= self.backPropagation(nn, activations, target_output) #DEBUG
        partial_derivatives_to_return  = copy.deepcopy(partial_derivatives) #DEBUG
        num_of_net_layers = len(nn.layers)
        formatted_learning_rate = convert_to_non_zero_if_needed(self.learning_rate, nn.numbersFormat)

        # Simple Learning Algorithm Steps
        # 1. PDs * learning_rate -> Learning_values_list
        # 2. Update net weights with Learning_values_list
        # 3. LATER: modify to implement SGD (define a mini-batch size and accumulate)

        for layer_index in range(num_of_net_layers-1, 0, -1):
            neurons_in_layer = len(nn.weightsMatrices[layer_index])
            weights_per_neuron = len(nn.weightsMatrices[layer_index][0])

            for neuron_index in range(neurons_in_layer):
                # PDs * learning_rate
                if self.samples_trained % self.SGD_mini_batch_size == 0: # First sample in mini-batch
                    listWithScalarOperation(formatted_learning_rate, partial_derivatives[layer_index][neuron_index],
                                            self.SGD_weights[layer_index][neuron_index], '*', nn.numbersFormat)
                else:   # NOT First sample in mini-batch, should accumulate gradients
                    listWithScalarOperation(formatted_learning_rate, partial_derivatives[layer_index][neuron_index],
                                            partial_derivatives[layer_index][neuron_index], '*', nn.numbersFormat)
                    listWithListOperation(self.SGD_weights[layer_index][neuron_index], partial_derivatives[layer_index][neuron_index],
                                          self.SGD_weights[layer_index][neuron_index], '+', nn.numbersFormat)
        self.samples_trained += 1

        if self.samples_trained % self.SGD_mini_batch_size == 0:  # A complete mini-batch was accumulated -> update NN weights
            for layer_index in range(num_of_net_layers - 1, 0, -1):
                neurons_in_layer = len(nn.weightsMatrices[layer_index])
                one_over_mini_batch_size = convert_to_non_zero_if_needed(1 / self.SGD_mini_batch_size, nn.numbersFormat)
                for neuron_index in range(neurons_in_layer):
                    listWithScalarOperation(one_over_mini_batch_size, self.SGD_weights[layer_index][neuron_index],
                                            self.SGD_weights[layer_index][neuron_index], '*', nn.numbersFormat)
                    listWithListOperation(nn.weightsMatrices[layer_index][neuron_index], self.SGD_weights[layer_index][neuron_index],
                                          nn.weightsMatrices[layer_index][neuron_index], '-', nn.numbersFormat)

        return (partial_derivatives_to_return, activations, deltas) #DEBUG

############################################################
######  Test function
############################################################
def test():
    nn_input_size = 3 # actual input length will be +1 due to bias
    fixed_point_10bit = FixedPoint.FixedPointFormat(6,10)

    # 1. Read Input

    # 2. FP

    # 3. BP

############################################################
######  Execute
############################################################
#test()