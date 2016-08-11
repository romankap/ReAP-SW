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
            input_vector.append(input_format.convert(random.uniform(0.0001, input_format.max)))
        #bias
        input_vector.append(1)
    else:
        print("Loading input from HD isn't supported yet")
        #Bias should be added manually (either here or in FP function)

    storage.loadData(input_vector, start_row, input_format.total_bits, column_index)


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

############################################################
######  Forward propagate an input through the net
############################################################
def forwardPropagation(nn, storage, nn_weights_column, nn_start_row, input_column, input_start_row):
    number_of_layers = len(nn.layers)

    storage.printArray()
    # 1) Broadcast
    input_layer_size = nn.layers[0][1]

    #Load bias
    storage.loadData([1], input_start_row + input_layer_size, nn.numbersFormat.total_bits, input_column)
    input_layer_size +=1

    hidden_layer_size = nn.layers[1][1]
    hidden_layer_start_row = nn_start_row + input_layer_size
    broadcastData(storage, input_column, input_start_row, input_layer_size,
                  hidden_layer_start_row, input_column, input_layer_size, hidden_layer_size-1) #first appearance of input is already aligned to appropriate weights

    storage.printArray()

    # 2) MUL
    print("MUL all layer weights with input")

    # 3) ACC


############################################################
######  Backward propagation of an output through the net
############################################################
def backPropagation(input):
    print("FW + BW in NN")


############################################################
######  Test function
############################################################
def test():
    storage = ReCAM.ReCAM(1024)
    storage.setVerbose(True)

    table_header_row = ["NN", "input"]
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

    forwardPropagation(nn, storage, nn_weights_column, nn_start_row, input_column, input_start_row)


############################################################
######  Execute
############################################################
test()