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


def createFullyConnectNN(input_size=5):
    fixed_point_8bit = FixedPoint.FixedPointFormat(6,10)
    nn = NeuralNetwork(fixed_point_8bit, input_size)
    print("input layer size =", nn.layers[0])

    nn.addLayer("FC", 3)
    print("Added FC layer, size =", nn.layers[1])

    nn.addLayer("output", 2)
    print("Added output layer, size =", nn.layers[2])

    return nn


def loadNNtoStorage(storage, nn, column_index):

    nn_row_in_ReCAM = 0

    for layer_index in range(1, len(nn.layers)):
        neurons_in_layer = len(nn.weightsMatrices[layer_index])
        weights_in_layer = len(nn.weightsMatrices[layer_index][0])

        for neuron_index in range(neurons_in_layer):
            storage.loadData(nn.weightsMatrices[layer_index][neuron_index], nn_row_in_ReCAM, nn.numbersFormat.total_bits, column_index)
            nn_row_in_ReCAM += weights_in_layer

    storage.printArray()


def loadInputToStorage(storage, input_size, column_index):
    input_vector = []
    input_format = FixedPoint.FixedPointFormat(8,8)

    for i in range(input_size):
        input_vector.append(input_format.convert(random.uniform(0.0001, input_format.max)))
    #bias
        input_vector.append(1)

    storage.loadData(input_vector, 0, input_format.total_bits, column_index)


def feedFrowardNN(input):
    print("Feed forward")


def trainOnSingleInput(input):
    print("FW + BW in NN")


def test():
    storage = ReCAM.ReCAM(1024)
    verbose_prints = True
    storage.setVerbose(verbose_prints)

    input_size = 5
    nn = createFullyConnectNN(input_size)
    loadNNtoStorage(storage, nn, 0)

    #create temp input
    loadInputToStorage(storage, input_size, 1)


test()