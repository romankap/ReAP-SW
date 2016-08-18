import os, sys, time

import ReCAM, Simulator
import NeuralNetwork
from NumberFormats import FixedPoint
import random


'''lib_path = os.path.abspath(os.path.join('swalign-0.3.3'))
sys.path.append(lib_path)
import swalign'''


''' Define a NN structure and the appropriate column to store.
Every layer has number of neurons - the net will be presented as a list.
First (last) layer is input (output) layer.
'''


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
def loadInputToStorage(storage, input_format, input_size, column_index, start_row, input_vector=None):
    if not input_vector:
        random_input_vector = []
        for i in range(input_size):
            random_input_vector.append(input_format.convert(random.uniform(0.0001, 1)))
        #bias
            random_input_vector.append(1)
        storage.loadData(random_input_vector, start_row, input_format.total_bits, column_index)

    else:
        print("Loading input from input_vector")
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
                  destination_start_row, destination_row_hops, destination_col_index, destination_delta, destination_rows_per_element):

    destination_row = destination_start_row
    for data_row in range(data_start_row_index, data_start_row_index + data_length):
        storage.broadcastDataElement(data_col_index, data_row,  destination_row,
                                     destination_col_index, destination_delta, destination_rows_per_element)
        destination_row += destination_row_hops

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
        storage.shiftColumnOnTaggedRows(res_col, tagged_rows, distance_to_shift=1) #shift to higher-indexed rows

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
def forwardPropagation(nn, storage, nn_weights_column, nn_start_row, input_column, MUL_column, accumulation_column):
    bias = [1]
    number_of_nn_layers = len(nn.layers)
    start_row = nn_start_row
    activations_col = input_column
    MUL_result_col = MUL_column
    ACC_result_col = accumulation_column

    table_header_row = ["NN", "input", "MUL", "ACC"]
    storage.setPrintHeader(table_header_row)

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
                      broadcast_start_row, 1, activations_col, weights_per_neuron, neurons_in_layer-1) #first appearance of input is already aligned with first neuron weights

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

    return (net_output, output_col)


############################################################
######  Backward propagation of an output through the net
############################################################
def backPropagation(nn, storage, nn_start_row, nn_weights_column, output_col, partial_derivatives_col,
                    activations_col, deltas_col, next_deltas_col):
    print("BP in NN")

    table_header_row = ["NN", "Out/Activ", "dErr/dW", "Activ/Out", "deltas", "next layer"]
    storage.setPrintHeader(table_header_row)

    zero_vector = [0] * nn.totalNumOfNetWeights
    storage.loadData(zero_vector, nn_start_row, nn.numbersFormat.total_bits, deltas_col)
    storage.loadData(zero_vector, nn_start_row, nn.numbersFormat.total_bits, next_deltas_col)

    output_start_row = nn_start_row + nn.totalNumOfNetWeights
    storage.rowWiseOperation(output_col, nn_weights_column, next_deltas_col,
                             output_start_row, output_start_row+nn.layers[-1][1]-1, '-')

    for layer_index in range(len(nn.layers)-1, 0, -1):

        # Layer structure
        neurons_in_layer = len(nn.weightsMatrices[layer_index])
        weights_per_neuron = len(nn.weightsMatrices[layer_index][0])
        total_layer_weights = neurons_in_layer * weights_per_neuron
        layer_start_row = output_start_row - total_layer_weights

        # Get layer deltas to 'deltas_col'
        broadcastData(storage, next_deltas_col, output_start_row, neurons_in_layer,
                      layer_start_row, weights_per_neuron, deltas_col, 1, weights_per_neuron)

        # Calculate partial derivatives for each weight
        storage.rowWiseOperation(deltas_col, activations_col, partial_derivatives_col,
                                 layer_start_row, layer_start_row+total_layer_weights-1, '*', nn.numbersFormat)


        # Calculating delta(prev_layer) = delta(curr_layer)*W(curr_layer)
        if layer_index!=1:     # No need for next_layer_delta in first hidden layer
            storage.rowWiseOperation(deltas_col, nn_weights_column, deltas_col,
                                     layer_start_row, layer_start_row + total_layer_weights - 1, '*', nn.numbersFormat)

            next_deltas_sum_start_row = output_start_row - weights_per_neuron
            storage.rowWiseOperation(deltas_col, deltas_col, next_deltas_col,
                                     next_deltas_sum_start_row, next_deltas_sum_start_row + weights_per_neuron-1, 'max', nn.numbersFormat)

            for neuron in range(neurons_in_layer-1):
                rows_to_shift = range(next_deltas_sum_start_row, next_deltas_sum_start_row + weights_per_neuron)
                storage.shiftColumnOnTaggedRows(next_deltas_col, rows_to_shift, -weights_per_neuron)

                next_deltas_sum_start_row -= weights_per_neuron
                storage.rowWiseOperation(deltas_col, next_deltas_col, next_deltas_col,
                                        next_deltas_sum_start_row, next_deltas_sum_start_row+weights_per_neuron-1, '+', nn.numbersFormat)

        output_start_row -= total_layer_weights
        output_col, activations_col = activations_col, output_col

    print("Finished BP in NN")


############################################################
######  Test function
############################################################
def NN_on_ReCAM_test():
    storage = ReCAM.ReCAM(1024)
    storage.setVerbose(True)

    nn_input_size = 3 # actual input length will be +1 due to bias
    fixed_point_10bit = FixedPoint.FixedPointFormat(6,10)
    nn = NeuralNetwork.createDemoFullyConnectNN(fixed_point_10bit, nn_input_size)

    nn_weights_column = 0
    nn_start_row = 0
    loadNNtoStorage(storage, nn, nn_weights_column, nn_start_row)

    input_column = 1
    FP_MUL_column = 2
    FP_accumulation_column = 3
    input_start_row = nn_start_row
    loadInputToStorage(storage, fixed_point_10bit, nn_input_size, input_column, input_start_row)

    target_output = [1,2]
    loadTargetOutputToStorage(storage, target_output, nn_start_row + nn.totalNumOfNetWeights, fixed_point_10bit, nn_weights_column)

    FP_output_column = forwardPropagation(nn, storage, nn_weights_column, nn_start_row, input_column, FP_MUL_column, FP_accumulation_column)

    BP_output_column = FP_output_column
    BP_partial_derivatives_column = FP_MUL_column
    activations_column = 1 if FP_output_column==3 else 3
    BP_deltas_column = 4
    BP_next_deltas_column = 5

    backPropagation(nn, storage, nn_start_row, nn_weights_column, BP_output_column, BP_partial_derivatives_column,
                    activations_column, BP_deltas_column, BP_next_deltas_column)


############################################################
######  Execute
############################################################
#NN_on_ReCAM_test()