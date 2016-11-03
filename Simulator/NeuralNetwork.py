import os, sys, time
import random
from NumberFormats import FixedPoint
import aux_functions

'''lib_path = os.path.abspath(os.path.join('swalign-0.3.3'))
sys.path.append(lib_path)
import swalign'''


''' Define a NN structure and the appropriate column to store.
Every layer has number of neurons - the net will be presented as a list.
First (last) layer is input (output) layer.
'''

def generateRandomInput(input_size, input_format):
    input_vector = []
    for i in range(input_size):
        input_vector.append(input_format.convert(random.gauss(0, 0.01)))
    #bias
    input_vector.append(1)
    return input_vector


def createDebugNN(weights_format, input_size):
    nn = NeuralNetwork(weights_format, input_size)
    print("input layer size =", nn.layers[0])

    nn.addLayer("FC", 3)
    print("Added FC layer, size =", nn.layers[1])

    nn.addLayer("FC", 2)
    print("Added FC layer, size =", nn.layers[2])

    # for layer_index in reversed(range(1, len(nn.layers))):
    #     for neuron_index in range(len(nn.layers[layer_index])):
    #         weights_per_neuron = len(nn.layers[layer_index])
    #
    #         for weight_index in range(weights_per_neuron):
    #             nn.weightsMatrices[layer_index][neuron_index][weight_index] = 0.0
    nn.weightsMatrices[1][1][1] = 0.0

    return nn


def createDemoFullyConnectNN(weights_format, input_size):
    nn = NeuralNetwork(weights_format, input_size)
    print("input layer size =", nn.layers[0])

    nn.addLayer("FC", 3)
    print("Added FC layer, size =", nn.layers[1])

    nn.addLayer("FC", 5)
    print("Added FC layer, size =", nn.layers[2])

    nn.addLayer("FC", 4)
    print("Added FC layer, size =", nn.layers[3])

    nn.addLayer("FC", 4)
    print("Added FC layer, size =", nn.layers[4])

    #nn.addLayer("softmax", 2)
    #print("Added softmax layer, size =", nn.layers[4])

    nn.addLayer("FC", 3)
    print("Added Fc layer, size =", nn.layers[5])

    return nn

def createMNISTFullyConnectNN(weights_format, input_size):
    nn = NeuralNetwork(weights_format, input_size)
    print("input layer size =", nn.layers[0])

    nn.addLayer("FC", 15)
    print("Added FC layer, size =", nn.layers[1])

    nn.addLayer("output", 10)
    print("Added output layer, size =", nn.layers[2])

    return nn


class NeuralNetwork:
    def __init__(self, number_format, inputs = 10):
        random.seed()
        self.numbersFormat = number_format
        self.weightsMax = number_format.max
        self.layers = []
        self.layers.append(("input", inputs)) # +1 due to bias

        self.weightsMatrices = []
        self.weightsMatrices.append(None)
        self.totalNumOfNetWeights = 0


    def addLayer(self, type, new_layer_neurons):
        weights_per_neuron = self.layers[len(self.layers)-1][1] + 1 # +1 due to bias

        self.weightsMatrices.append([[] for x in range(new_layer_neurons)]) #append new layer weights
        self.layers.append((type, new_layer_neurons))

        if type == "FC" or type == "output" or type == "softmax":
            self.addFCLayer(weights_per_neuron, new_layer_neurons, len(self.layers)-1)
            self.totalNumOfNetWeights += weights_per_neuron*new_layer_neurons
        else:
            print("Unknown layer type")

    ###############################################################
    ####    Initialize all weights for a FC layer               ###
    ###############################################################
    def addFCLayer(self, prev_later_neurons, new_layer_neurons, new_layer_index):
        max_random_weight = 1

        for i in range(new_layer_neurons):
            for j in range(prev_later_neurons):
                self.weightsMatrices[new_layer_index][i].append(aux_functions.convert_number_to_non_zero_if_needed(getRandomWeight(max_random_weight), self.numbersFormat))


    def convert_all_results_to_format(self, layer_index):
        num_of_neurons = len(self.weightsMatrices[layer_index])
        weights_per_neuron = len(self.weightsMatrices[layer_index][0])

        for i in range(num_of_neurons):
            for j in range(weights_per_neuron):
                self.weightsMatrices[layer_index][i][j] = aux_functions.convert_number_to_non_zero_if_needed(self.weightsMatrices[layer_index][i][j], self.numbersFormat)



def getRandomWeight(max):
    return random.gauss(0, 0.01)


def test():
    fixed_point_8bit = FixedPoint.FixedPointFormat(8,8)
    nn = NeuralNetwork(fixed_point_8bit, 5)
    print("input layer size =", nn.layers[0])

    nn.addLayer("FC", 3)
    print("Added FC layer, size =", nn.layers[1])

    nn.addLayer("output", 2)
    print("output layer size =", nn.layers[2])


#test()