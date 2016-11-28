import os,sys
import random
import struct
import copy
from array import array
import MNIST_extract_weights

import aux_functions
import ReCAM, Simulator
import NeuralNetwork
from NumberFormats import FixedPoint
import NNs_on_ReCAM, NNs_on_CPU_no_debug
from MNIST_class import MNIST
import datetime

MNIST_path = 'C:\Dev\MNIST'
binary_output_file = None
numerical_output_file = None

binary_weights_filename = "C:\\Dev\\MNIST\\Weight_extraction\\784HU.binary.Accuracy=15.0002016-11-27.15-20-08.359151.txt"

###--- Handle output files ---###
def load_weights_from_file(nn, weights_filename):
    with open(weights_filename, mode='rb') as weights_file:  # b is important -> binary
        file_contents = weights_file.read()

    bytes_counter = 0
    for layer_index in range(1, len(nn.weightsMatrices)):
        for neuron_index in range(len(nn.weightsMatrices[layer_index])):
            weights_per_neuron = len(nn.weightsMatrices[layer_index][0])

            for weight_index in range(weights_per_neuron):
                float_bin = struct.unpack('<f', file_contents[bytes_counter:bytes_counter+4])[0]
                nn.weightsMatrices[layer_index][neuron_index][weight_index] = float_bin
                bytes_counter += 4


def write_test_labels_to_file(inference_labels):
    global binary_weights_filename
    labels_output_file = open(binary_weights_filename + "-labels.txt", 'w')

    for label_index in range(len(inference_labels)):
        labels_output_file.write(str(label_index) + "," + str(inference_labels[label_index]) + "\n")

    ### Output weight strings once done appending
    labels_output_file.close()
    print("Outputted LABELS to file")


def close_weight_files():
    global binary_output_file
    global numerical_output_file
    binary_output_file.close()
    numerical_output_file.close()


###--- Weight extraction ---###
def convert_to_short_float(float):
    return "%0.3f" % float


################# MNIST #################
def get_MNIST_class_from_output(output_array):
    return output_array.index(max(output_array))

def get_inference_results():
    #Load MNIST data

    mnist_data = MNIST(MNIST_path)
    mnist_data.load_testing()

    nn_input_size = 784  # actual input length will be +1 due to bias
    hidden_layer_size = 1000
    nn = NeuralNetwork.createMNISTWeightExtractionNet(hidden_layer_size=hidden_layer_size, input_size=nn_input_size)
    NN_on_CPU = NNs_on_CPU_no_debug.initialize_NN_on_CPU(nn)
    load_weights_from_file(nn, binary_weights_filename)
    #MNIST_extract_weights.print_net_weights_to_files(nn, str(nn_input_size) + "HU", 15)

    number_of_correct_classifications = 0
    CPU_FF_labels = []
    for testing_iteration in range(len(mnist_data.test_images)):
    ##for testing_iteration in range(1000): #DEBUG
        test_image = mnist_data.test_images[testing_iteration]
        NN_on_CPU.feedforward(nn, test_image)
        CPU_sample_label = get_MNIST_class_from_output(NN_on_CPU.activations[-1])
        if CPU_sample_label == mnist_data.test_labels[testing_iteration]:
            number_of_correct_classifications += 1
        CPU_FF_labels.append(CPU_sample_label)

        if testing_iteration % 100 == 0:
            print("Number of correct classifications {} out of total {}".format(number_of_correct_classifications, testing_iteration))

    fraction_of_correct_classifications = number_of_correct_classifications / len(mnist_data.test_images)
    ##aux_functions.write_to_output_file("epoch number:", epoch_number, ". CPU fraction of correct classifications:", fraction_of_correct_classifications)
    print("CPU fraction of correct classifications:", fraction_of_correct_classifications)
    write_test_labels_to_file(CPU_FF_labels)


#----- Execute -----#
get_inference_results()