import os, sys, time

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
        prev_layer_neurons = self.layers[len(self.layers)-1][1]

        self.layers.append((type, num_of_neurons))

def initializeWeights():
    print("Insert weight values to ReCAM. Weights should be stored in a specific format (float / fixed-point).")




def test():
    nn = NeuralNetwork(5)
    print("input layer size =", nn.layers[0])

    nn.addLayer("FC", 10)

    print("Added FC layer")

test()