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


############################################################
######  Load NN to ReCAM
############################################################
def loadNNtoStorage(storage, nn, column_index, nn_start_row_in_ReCAM):
    nn_row_in_ReCAM = nn_start_row_in_ReCAM

    for layer_index in range(1, len(nn.layers)):
        neurons_in_layer = len(nn.weightsMatrices[layer_index])
        weights_in_layer = len(nn.weightsMatrices[layer_index][0])

        for neuron_index in range(neurons_in_layer):
            storage.loadData(nn.weightsMatrices[layer_index][neuron_index], nn_row_in_ReCAM, nn.numbersFormat.total_bits, column_index)
            nn_row_in_ReCAM += weights_in_layer

    storage.printArray()


############################################################
######  Load Input to ReCAM (read from file / generate)
############################################################
def loadInputToStorage(storage, input_format, input_size, column_index, start_row, generate_random_input=True):
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


#####################################################################
######      Broadcast a single element to multiple ReCAM rows
#####################################################################
def loadTargetOutputToStorage(storage, target_output, start_row, number_format, column_index):
    storage.loadData(target_output, start_row, number_format.total_bits, column_index)

#####################################################################
######      Broadcast a single element to multiple ReCAM rows
#####################################################################
def broadcastData(storage, data_col_index, data_start_row_index, data_length,
                  destination_start_row, destination_col_index, destination_delta, total_destination_rows):

    weight_row = destination_start_row
    for data_row in range(data_start_row_index, data_start_row_index + data_length):
        storage.broadcastDataElement(data_col_index, data_row,  weight_row,
                                     destination_col_index, destination_delta, total_destination_rows)
        weight_row += 1

        # cycle count is set by broadcastDataElement()


#####################################################################
######      Accumulate multiple sums in parallel, row-by-row
#####################################################################
def parallelAccumulate(storage, col_A, col_B, res_col, start_row, rows_delta,
                       num_of_parallel_sums, num_of_accumulations_per_sum, number_format):
    #first iteration - only accumulate
    tagged_rows = range(start_row, start_row + num_of_parallel_sums*rows_delta, rows_delta)
    storage.taggedRowWiseOperation(col_A, col_B, res_col, tagged_rows, '+', number_format)

    start_row_to_accumulate = start_row
    for i in range(1, num_of_accumulations_per_sum):
        storage.tagRows(res_col)
        tagged_rows = range(start_row_to_accumulate, start_row_to_accumulate + num_of_parallel_sums*rows_delta, rows_delta)
        storage.shiftColumnOnTaggedRows(res_col, tagged_rows, direction_of_shift=1) #shift to higher rows

        start_row_to_accumulate = start_row + i
        storage.tagRows(res_col)
        tagged_rows = range(start_row_to_accumulate, start_row_to_accumulate + num_of_parallel_sums*rows_delta, rows_delta)
        storage.taggedRowWiseOperation(col_A, col_B, res_col, tagged_rows, '+', number_format)

    for i in range(1, num_of_parallel_sums+1):
        storage.tagRows(res_col)
        accumulation_result_row = start_row + i*num_of_accumulations_per_sum - 1
        final_output_destination_row = start_row + num_of_parallel_sums*num_of_accumulations_per_sum + i-1
        storage.broadcastDataElement(res_col, accumulation_result_row, final_output_destination_row, res_col, 0, 1)


############################################################
######  Forward propagate an input through the net
############################################################
def forwardPropagation(nn, storage, nn_weights_column, nn_start_row, input_column, input_start_row):
    bias = [1]
    number_of_nn_layers = len(nn.layers)
    start_row = nn_start_row
    activations_col = input_column
    MUL_result_col = 2
    ACC_result_col = 3

    for layer_index in range(1, number_of_nn_layers):
        neurons_in_layer = len(nn.weightsMatrices[layer_index])
        weights_per_neuron = len(nn.weightsMatrices[layer_index][0])
        layer_total_weights = neurons_in_layer * weights_per_neuron
        zero_vector = [0] * layer_total_weights

        storage.printArray(msg=("beginning of forwardPropagation iteration, layer " + str(layer_index)))
        # 1) Broadcast
        #Load bias
        activations_from_prev_layer = nn.layers[layer_index - 1][1]
        storage.loadData(bias, start_row + activations_from_prev_layer, nn.numbersFormat.total_bits, activations_col)

        broadcast_start_row = start_row + weights_per_neuron # first instance of input is aligned with first neuron weights
        broadcastData(storage, activations_col, start_row, weights_per_neuron,
                      broadcast_start_row, activations_col, weights_per_neuron, neurons_in_layer-1) #first appearance of input is already aligned with first neuron weights

        storage.printArray(msg="after broadcast")

        # 2) MUL
        hidden_layer_start_row = start_row

        storage.loadData(zero_vector, start_row, nn.numbersFormat.total_bits, MUL_result_col)
        storage.MULConsecutiveRows(start_row, start_row + layer_total_weights-1, MUL_result_col, nn_weights_column, activations_col, nn.numbersFormat)

        storage.printArray(msg="after MUL")

        # 3) Accumulate

        storage.loadData(zero_vector, start_row, nn.numbersFormat.total_bits, ACC_result_col)

        parallelAccumulate(storage, MUL_result_col, ACC_result_col, ACC_result_col,
                           start_row, weights_per_neuron,
                           neurons_in_layer, weights_per_neuron, nn.numbersFormat)

        start_row += layer_total_weights
        activations_col, ACC_result_col = ACC_result_col, activations_col


        storage.printArray(msg="after Accumulate")

    output_col = activations_col #after each iteration, activations_col holds layer output
    net_output = []
    for i in range(nn.layers[number_of_nn_layers-1][1]):
        net_output.append(storage.crossbarArray[start_row+i][output_col])
    print("")
    print("=== NN output is: ", net_output)

############################################################
######  Backward propagation of an output through the net
############################################################
def backPropagation(input):
    print("BP in NN")

    # 1) Calc deltas: (out-target) for iteration 1, delta(N+1)*W(N+1) for the rest
    # For non-first iteration, matrix multiplication will be

    # 2) Calc partial derivatives: activation*delta for weights, only 'delta' for bias


############################################################
######  Test function
############################################################
def test():
    storage = ReCAM.ReCAM(1024)
    storage.setVerbose(True)

    table_header_row = ["NN", "input", "MUL", "ACC"]
    storage.setPrintHeader(table_header_row)

    nn_input_size = 3 # actual input length will be +1 due to bias
    fixed_point_10bit = FixedPoint.FixedPointFormat(6,10)
    nn = createFullyConnectNN(fixed_point_10bit, nn_input_size)

    nn_weights_column = 0
    nn_start_row = 0
    loadNNtoStorage(storage, nn, nn_weights_column, nn_start_row)

    input_column = 1
    input_start_row = 0
    loadInputToStorage(storage, fixed_point_10bit, nn_input_size, input_column, input_start_row)

    target_output = [1,2]
    loadTargetOutputToStorage(storage, target_output, nn_start_row + nn.totalNumOfNetWeights, fixed_point_10bit, nn_weights_column)

    forwardPropagation(nn, storage, nn_weights_column, nn_start_row, input_column, input_start_row)


############################################################
######  Execute
############################################################
test()