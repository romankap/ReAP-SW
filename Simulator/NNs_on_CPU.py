import os, sys, time

import ReCAM, Simulator
from  NeuralNetwork import NeuralNetwork
from NumberFormats import FixedPoint
import random


'''lib_path = os.path.abspath(os.path.join('swalign-0.3.3'))
sys.path.append(lib_path)
import swalign'''


''' Define a NN structure and the appropriate column to store.
Every layer has number of neurons - the net will be presented as a list.
First (last) layer is input (output) layer.
'''


def createFullyConnectNN(weights_format, input_size):
    nn = NeuralNetwork(weights_format, input_size)
    print("input layer size =", nn.layers[0])

    nn.addLayer("FC", 3)
    print("Added FC layer, size =", nn.layers[1])

    nn.addLayer("output", 2)
    print("Added output layer, size =", nn.layers[2])

    return nn


def listWithListOperation(list_A, list_B, res_list, operation, number_format=None):
    if operation=='-':
        for i in range(len(list_A)):
            res_list[i] = list_A[i] - list_B[i]
    elif operation == '+':
        for i in range(len(list_A)):
            res_list[i] = list_A[i] + list_B[i]
    elif operation == '*':
        for i in range(len(list_A)):
            res_list[i] = number_format.convert(list_A[i] * list_B[i])


def listWithScalarOperation(scalar, list_A, res_list, operation, number_format):
    if operation == '*':
        for i in range(len(list_A)):
            res_list[i] = number_format.convert(scalar * list_A[i])
    elif operation == '+':
        for i in range(len(list_A)):
            res_list[i] = number_format.convert(scalar + list_A[i])


############################################################
######  Load Input to ReCAM (read from file / generate)
############################################################
def loadInput(storage, input_format, input_size, column_index, start_row, generate_random_input=True):
    input_vector = []

    if generate_random_input:
        for i in range(input_size):
            input_vector.append(input_format.convert(random.uniform(0.0001, 1)))
        #bias
        input_vector.append(1)
    else:
        print("Loading input from HD isn't supported yet")
        #Bias should be added manually (either here or in FP function)

    storage.loadData(input_vector, start_row, input_format.total_bits, column_index)


############################################################
######  Forward propagate an input through the net
############################################################
def ReLUactivation(input):
    return max(0, input)

def forwardPropagation(nn, input, number_format):
    print("FP in NN")

    num_of_net_layers = len(nn.layers)
    activations = []
    activations.append(input)

    for layer_index in range(1, num_of_net_layers):
        activations.append([])
        neurons_in_layer = len(nn.weightsMatrices[layer_index])
        weights_per_neuron = len(nn.weightsMatrices[layer_index][0])

        for neuron in range(neurons_in_layer):
            sum = 0
            for weight in range(weights_per_neuron):
                sum += number_format.convert(activations[layer_index-1][weight] * nn.weightsMatrices[layer_index][neuron][weight])
            activations[layer_index].append(sum)

        if layer_index!=num_of_net_layers-1:
            activations[layer_index].append(1) #bias exists only in input+hidden layers

    #print(net_output)
    return activations


############################################################
######  Backward propagation of an output through the net
############################################################
def backPropagation(nn, activations, target, number_format):
    num_of_net_layers = len(nn.layers)

    partial_derivatives = []
    partial_derivatives.append(None)

    curr_delta = None
    prev_delta = None
    for layer_index in range(num_of_net_layers-1):
        partial_derivatives.append([])

    for layer_index in range(num_of_net_layers-1, 0, -1):
        neurons_in_layer = len(nn.weightsMatrices[layer_index])
        weights_per_neuron = len(nn.weightsMatrices[layer_index][0])

        # Deltas of output layer
        if layer_index==num_of_net_layers-1:
            curr_delta = [0] * len(activations[num_of_net_layers - 1])
            listWithListOperation(activations[num_of_net_layers - 1], target, curr_delta, '-')
        # Deltas of hidden layers
        else:
            neurons_in_prev_bp_layer = len(nn.weightsMatrices[layer_index+1])
            weights_per_neuron_in_prev_bp_layer = len(nn.weightsMatrices[layer_index+1][0])

            curr_delta = [0] * weights_per_neuron
            for neuron_in_prev_bp_index in range(neurons_in_prev_bp_layer):
                temp_delta = [0] * weights_per_neuron
                listWithScalarOperation(prev_delta[neuron_in_prev_bp_index], nn.weightsMatrices[layer_index+1][neuron_in_prev_bp_index], temp_delta, '*', number_format)
                listWithListOperation(temp_delta, curr_delta, curr_delta, '+')

        for neuron_index in range(neurons_in_layer):
            neuron_pds = [0] * weights_per_neuron
            listWithScalarOperation(curr_delta[neuron_index], activations[layer_index-1], neuron_pds, '*', number_format)
            partial_derivatives[layer_index].append(neuron_pds)

        prev_delta = curr_delta

    print("Finished BP in NN on CPU")
    return partial_derivatives


############################################################
######  Test function
############################################################
def test():
    nn_input_size = 3 # actual input length will be +1 due to bias
    fixed_point_10bit = FixedPoint.FixedPointFormat(6,10)
    nn = createFullyConnectNN(fixed_point_10bit, nn_input_size)

    # 1. Read Input

    # 2. FP

    # 3. BP

############################################################
######  Execute
############################################################
test()