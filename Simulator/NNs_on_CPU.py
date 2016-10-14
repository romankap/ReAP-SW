import os, sys, time

import ReCAM, Simulator
from  NeuralNetwork import NeuralNetwork
from NumberFormats import FixedPoint
import random, copy


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

def convert_if_needed(result, number_format=None):
    if number_format:
        return number_format.convert(result)
    return result

################################################
####        NN & List Operations
################################################

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
            res_list[i] = convert_if_needed(list_A[i] - list_B[i], number_format)
    elif operation == '+':
        for i in range(len(list_A)):
            res_list[i] = convert_if_needed(list_A[i] + list_B[i], number_format)
    elif operation == '*':
        for i in range(len(list_A)):
            res_list[i] = convert_if_needed(list_A[i] * list_B[i], number_format)


def listWithScalarOperation(scalar, list_A, res_list, operation, number_format=None):
    converted_scalar = convert_if_needed(scalar, number_format)

    if operation == '*':
        for i in range(len(list_A)):
            res_list[i] = convert_if_needed(converted_scalar * list_A[i], number_format)
    elif operation == '+':
        for i in range(len(list_A)):
            res_list[i] = convert_if_needed(converted_scalar + list_A[i], number_format)


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
######  Feedforward an input through the net
############################################################
def ReLUactivation(input):
    return max(0, input)

def feedforward(nn, input):
    print("FP in NN")

    num_of_net_layers = len(nn.layers)
    activations = []
    activations.append(input)

    for layer_index in range(1, num_of_net_layers):
        activations.append([])
        neurons_in_layer = len(nn.weightsMatrices[layer_index])
        weights_per_neuron = len(nn.weightsMatrices[layer_index][0])

        for neuron in range(neurons_in_layer):
            weighted_sum = 0
            for weight in range(weights_per_neuron):
                weighted_sum = nn.numbersFormat.convert(weighted_sum +
                                                        nn.numbersFormat.convert(activations[layer_index-1][weight] * nn.weightsMatrices[layer_index][neuron][weight]))
            activations[layer_index].append(weighted_sum)

        if layer_index!=num_of_net_layers-1:
            activations[layer_index].append(1) #bias exists only in input+hidden layers

    #print(net_output)
    return activations


############################################################
######  Backward propagation of an output through the net
############################################################
def backPropagation(nn, activations, target):
    num_of_net_layers = len(nn.layers)
    partial_derivatives = [None]

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
            listWithListOperation(activations[num_of_net_layers - 1], target, curr_delta, '-', nn.numbersFormat)
        # Deltas of hidden layers
        else:
            neurons_in_prev_bp_layer = len(nn.weightsMatrices[layer_index+1])
            weights_in_prev_bp_layer_neuron = len(nn.weightsMatrices[layer_index+1][0])

            curr_delta = [0] * weights_in_prev_bp_layer_neuron
            for neuron_in_prev_bp_index in range(neurons_in_prev_bp_layer):
                temp_delta = [0] * weights_in_prev_bp_layer_neuron
                listWithScalarOperation(prev_delta[neuron_in_prev_bp_index], nn.weightsMatrices[layer_index+1][neuron_in_prev_bp_index], temp_delta, '*', nn.numbersFormat)
                listWithListOperation(temp_delta, curr_delta, curr_delta, '+', nn.numbersFormat)

        for neuron_index in range(neurons_in_layer):
            neuron_pds = [0] * weights_per_neuron
            listWithScalarOperation(curr_delta[neuron_index], activations[layer_index-1], neuron_pds, '*', nn.numbersFormat)
            partial_derivatives[layer_index].append(neuron_pds)

        prev_delta = curr_delta

    print("Finished BP in NN on CPU")
    return partial_derivatives


############################################################
######  Backward propagation of an output through the net
############################################################
def update_weights(nn, SGD_weights, partial_derivatives, learning_rate = 0.05):
    num_of_net_layers = len(nn.layers)
    learning_values_list = copy.deepcopy(partial_derivatives)
    formatted_learning_rate = nn.numbersFormat.convert(learning_rate)

    if SGD_weights==[None]:
        SGD_weights = copy.deepcopy(partial_derivatives)

    # Simple Learning Algorithm Steps
    # 1. PDs * learning_rate -> Learning_values_list
    # 2. Update net weights with Learning_values_list
    # 3. LATER: modify to implement SGD (define a mini-batch size and accumulate)

    for layer_index in range(num_of_net_layers-1, 0, -1):
        neurons_in_layer = len(nn.weightsMatrices[layer_index])
        weights_per_neuron = len(nn.weightsMatrices[layer_index][0])

        for neuron_index in range(neurons_in_layer):
            # PDs * learning_rate
            listWithScalarOperation(formatted_learning_rate, partial_derivatives[layer_index][neuron_index],
                                    learning_values_list[layer_index][neuron_index],'*', nn.numbersFormat)

    for layer_index in range(num_of_net_layers - 1, 0, -1):
        neurons_in_layer = len(nn.weightsMatrices[layer_index])
        for neuron_index in range(neurons_in_layer):
            listWithListOperation(nn.weightsMatrices[layer_index][neuron_index], learning_values_list[layer_index][neuron_index],
                                  nn.weightsMatrices[layer_index][neuron_index], '-', nn.numbersFormat)



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
#test()