import os, sys, time

import ReCAM, Simulator
import NumberFormats

'''lib_path = os.path.abspath(os.path.join('swalign-0.3.3'))
sys.path.append(lib_path)
import swalign'''


''' Define a NN structure and the appropriate column to store.
Every layer has number of neurons - the net will be presented as a list.
First (last) layer is input (output) layer.
'''

class NeuralNetwork:
    def __init__(self, input_layer = 10):
        self.layers = []
        self.layers.append(("input", 10))

    def addLayer(self, type, num_of_neurons):
        self.layers.append((type, num_of_neurons))

def initializeWeights():
    print("Insert weight values to ReCAM. Weights should be stored in a specific format (float / fixed-point).")


def createFullyConnectNN():
    network_structure = [3,5,1]
    print("Create FC networks")

def feedFrowardNN(input):
    print("Feed forward")

def trainOnSingleInput(input):
    print("FW + BW in NN")


def test():
    nn = NeuralNetwork(5)
    print("input layer size =", nn.layers[0])

    nn.addLayer("FC", 10)

test()