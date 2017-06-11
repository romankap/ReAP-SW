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

    nn.addLayer("FC", 10)
    print("Added FC layer, size =", nn.layers[1])

    nn.addLayer("softmax", 10)
    print("Added softmax layer, size =", nn.layers[2])

    return nn


def createDemoFullyConnectNN(weights_format=None, input_size=10):
    nn = NeuralNetwork(weights_format, input_size)
    print("input layer size =", nn.layers[0])

    nn.addLayer("FC", 3)
    print("Added FC layer, size =", nn.layers[1])


    #nn.addLayer("FC", 3)
    #print("Added FC layer, size =", nn.layers[3])

    #nn.addLayer("FC", 6)
    #print("Added FC layer, size =", nn.layers[4])

    #nn.addLayer("softmax", 2)
    #print("Added softmax layer, size =", nn.layers[4])

    nn.addLayer("softmax", 10)
    print("Added Fc layer, size =", nn.layers[2])

    return nn

def createMNISTConvNet(weights_format=None, input_size=10):
    nn = NeuralNetwork(weights_format, input_size)
    i = 0
    print("input layer size =", nn.layers[i])
    i += 1

    # nn.addLayer("FC", 2500)
    # print("Added FC layer, size =", nn.layers[i])
    # i += 1

    nn.addLayer("conv", 32, 5, 5)
    print("Added FC layer, size =", nn.layers[i])
    i += 1

    nn.addLayer("softmax", 10)
    print("Added output layer, size =", nn.layers[i])

    return nn


def createMNISTFullyConnectNN(weights_format=None, input_size=10):
    nn = NeuralNetwork(weights_format, input_size)
    i=0
    print("input layer size =", nn.layers[i])
    i+=1

    # nn.addLayer("FC", 2500)
    # print("Added FC layer, size =", nn.layers[i])
    # i += 1

    nn.addLayer("FC", 2000)
    print("Added FC layer, size =", nn.layers[i])
    i += 1

    nn.addLayer("FC", 1500)
    print("Added FC layer, size =", nn.layers[i])
    i += 1

    nn.addLayer("FC", 1000)
    print("Added FC layer, size =", nn.layers[i])
    i += 1

    nn.addLayer("FC", 500)
    print("Added FC layer, size =", nn.layers[i])
    i += 1

    nn.addLayer("softmax", 10)
    print("Added output layer, size =", nn.layers[i])

    return nn

def createMNISTWeightExtractionNet(weights_format=None, hidden_layer_size=50, input_size=10):
    nn = NeuralNetwork(weights_format, input_size)
    print("input layer size =", nn.layers[0])

    nn.addLayer("FC", hidden_layer_size)
    print("Added FC layer, size =", nn.layers[1])

    nn.addLayer("softmax", 10)
    print("Added output layer, size =", nn.layers[2])

    return nn


class NeuralNetwork:
    def __init__(self, number_format=None, inputs=10):
        random.seed()
        self.numbersFormat = number_format
        if self.numbersFormat:
            self.weightsMax = number_format.max
        self.layers = []
        self.layers.append(("input", inputs)) # +1 due to bias

        self.weightsMatrices = []
        self.weightsMatrices.append(None)
        self.totalNumOfNetWeights = 0


    def addLayer(self, type, new_layer_neurons, conv_y_dim_size=0, conv_x_dim_size=0):
        weights_per_neuron = self.layers[len(self.layers)-1][1] + 1 # +1 due to bias
        layer_index = len(self.layers)-1

        if type == "conv":
            num_of_curr_layer_feature_maps = new_layer_neurons
            num_of_prev_layer_feature_maps = len(self.weightsMatrices[layer_index-1]) if layer_index!=0 else 1
            self.weightsMatrices.append([[[[] for y in range(conv_y_dim_size)] for output_feature_map in range(num_of_curr_layer_feature_maps)] for input_feature_map in range(num_of_prev_layer_feature_maps) ])  # append new layer weights
            #self.weightsMatrices[-1].append(0)
            self.layers.append((type, num_of_curr_layer_feature_maps))
            self.addConvLayer(num_of_prev_layer_feature_maps, layer_index, num_of_curr_layer_feature_maps, conv_y_dim_size, conv_x_dim_size)

        if type == "FC" or type == "output" or type == "softmax":
            self.weightsMatrices.append([[] for x in range(new_layer_neurons)])  # append new layer weights
            self.layers.append((type, new_layer_neurons))

            self.addFCLayer(weights_per_neuron, new_layer_neurons, layer_index)
            self.totalNumOfNetWeights += weights_per_neuron*new_layer_neurons
        else:
            print("Unknown layer type")

    ###############################################################
    ####    Initialize all weights for a FC layer               ###
    ###############################################################
    def addConvLayer(self, num_of_prev_layer_feature_maps, new_layer_index, num_of_curr_layer_feature_maps, y_dim_size, x_dim_size):
        for input_feature_map_index in range(num_of_prev_layer_feature_maps):
            for output_feature_map_index in range(num_of_curr_layer_feature_maps):
                for y_index in range(y_dim_size):
                    for x_index in range(x_dim_size):
                        self.weightsMatrices[new_layer_index][input_feature_map_index][output_feature_map_index][y_index].append(aux_functions.convert_number_to_non_zero_if_needed(getRandomWeight(num_of_curr_layer_feature_maps), self.numbersFormat))
            self.weightsMatrices[input_feature_map_index][-1].append(aux_functions.convert_number_to_non_zero_if_needed(getRandomWeight(num_of_curr_layer_feature_maps), self.numbersFormat))

    def addFCLayer(self, prev_layer_neurons, new_layer_neurons, new_layer_index):
        for i in range(new_layer_neurons):
            for j in range(prev_layer_neurons):
                self.weightsMatrices[new_layer_index][i].append(aux_functions.convert_number_to_non_zero_if_needed(getRandomWeight(prev_layer_neurons), self.numbersFormat))


    def convert_all_results_to_format(self, layer_index):
        num_of_neurons = len(self.weightsMatrices[layer_index])
        weights_per_neuron = len(self.weightsMatrices[layer_index][0])

        for i in range(num_of_neurons):
            for j in range(weights_per_neuron):
                self.weightsMatrices[layer_index][i][j] = aux_functions.convert_number_to_non_zero_if_needed(self.weightsMatrices[layer_index][i][j], self.numbersFormat)



def getRandomWeight(fan_in=100):
    return random.gauss(0, 1/fan_in)


def test():
    fixed_point_8bit = FixedPoint.FixedPointFormat(8,8)
    nn = NeuralNetwork(fixed_point_8bit, 5)
    print("input layer size =", nn.layers[0])

    nn.addLayer("FC", 3)
    print("Added FC layer, size =", nn.layers[1])

    nn.addLayer("output", 2)
    print("output layer size =", nn.layers[2])


#test()