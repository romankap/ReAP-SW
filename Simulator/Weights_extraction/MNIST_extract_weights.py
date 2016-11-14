import os,sys
import random
import struct
import copy
from array import array

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

###--- Handle output files ---###
def open_weight_files(output_folder, net_name="", accuracy=None):
    now = str(datetime.datetime.now()).replace(':','-').replace(' ','.')
    binary_output_filename = output_folder + net_name + ".binary"
    numerical_output_filename = output_folder + net_name + ".numerical"
    if accuracy!=None:
        binary_output_filename += ".Accuracy=" + str(convert_to_short_float(accuracy))
        numerical_output_filename += ".Accuracy=" + str(convert_to_short_float(accuracy))

    binary_output_filename += now + ".txt"
    numerical_output_filename += now + ".txt"
    global binary_output_file
    global numerical_output_file
    binary_output_file = open(binary_output_filename, 'wb')
    numerical_output_file = open(numerical_output_filename, 'w')


def write_weights_to_file(weights_file, weights_string):
    if weights_file:
        weights_file.write(weights_string)


def write_to_weights_files(binary_weights_string, numerical_weights_string):
    global binary_output_file
    write_weights_to_file(binary_output_file, bytes(binary_weights_string))

    global numerical_output_file
    write_weights_to_file(numerical_output_file, numerical_weights_string)


def close_weight_files():
    global binary_output_file
    global numerical_output_file
    binary_output_file.close()
    numerical_output_file.close()

###--- Weight extraction ---###
def convert_to_short_float(float):
    return "%0.3f" % float


def print_net_weights_to_files(nn, net_name, fraction_of_correct_classifications):
    weights_numerical_string = ""
    weights_binary_string = None

    open_weight_files(MNIST_path + '\\Weight_extraction\\', net_name, fraction_of_correct_classifications)

    for layer_index in reversed(range(1, len(nn.weightsMatrices))):
        for neuron_index in range(len(nn.weightsMatrices[layer_index])):
            weights_per_neuron = len(nn.weightsMatrices[layer_index][0])

            for weight_index in range(weights_per_neuron):
                weights_to_print = convert_to_short_float(nn.weightsMatrices[layer_index][neuron_index][weight_index])
                weights_numerical_string += str(weights_to_print) + "\n"
                #if ReCAM_pds[index_in_ReCAM] != CPU_pds[layer_index][neuron_index][weight_index]:

                float_bin = struct.pack('<f', nn.weightsMatrices[layer_index][neuron_index][weight_index])
                if not weights_binary_string:
                    weights_binary_string = float_bin
                else:
                    weights_binary_string += float_bin

    ### Output weight strings once done appending
    write_to_weights_files(weights_binary_string, weights_numerical_string)
    close_weight_files()
    print("Outputted weights to files")


################# MNIST #################
def get_MNIST_class_from_output(output_array):
    return output_array.index(max(output_array))

def train_MNIST_and_extract_weights():
    #Load MNIST data

    mnist_data = MNIST(MNIST_path)
    mnist_data.load_training()
    mnist_data.load_testing()

    nn_input_size = 784  # actual input length will be +1 due to bias
    hidden_layer_size = 1000
    nn = NeuralNetwork.createMNISTWeightExtractionNet(hidden_layer_size=hidden_layer_size, input_size=nn_input_size)
    net_name = str(hidden_layer_size) + "HU"
    NN_on_CPU = NNs_on_CPU_no_debug.initialize_NN_on_CPU(nn)

    learning_rate = 0.01
    mini_batch_size = 10

    # --- CPU ---#
    NN_on_CPU.set_SGD_parameters(nn, mini_batch_size, learning_rate)

    total_training_epochs = 30
    #print_net_weights_to_files(nn, net_name, 0.1)
    for epoch_number in range(total_training_epochs):
        ##for training_iteration in range(len(mnist_data.train_images)):
        for training_iteration in range(1000): #DEBUG
            train_image = mnist_data.train_images[training_iteration]
            train_label = mnist_data.train_labels[training_iteration]
            target_output = [0] * 10
            target_output[train_label] = 1

            #--- CPU ---#
            NN_on_CPU.SGD_train(nn, train_image, target_output)[-1]
            output_layer_activations = NN_on_CPU.activations[-1]
            FF_output = get_MNIST_class_from_output(output_layer_activations)
            if training_iteration % 50 ==0:
                print("Finished CPU Execution", training_iteration)
                print("CPU FF output=",FF_output, ". Label=", train_label)

            # --- Verify weights match ---#
            #NNs_unit_tests.compare_NN_matrices(ReCAM_weights, nn.weightsMatrices, "weights")
            #aux_functions.write_to_output_file("Training iteration: ", training_iteration,
                                               #". Target output:", target_output)

        print("Training epoch: ", epoch_number)

        number_of_correct_classifications = 0
        CPU_FF_labels = []
        ##for testing_iteration in range(len(mnist_data.test_images)):
        for testing_iteration in range(500): #DEBUG
            ##test_image = mnist_data.test_images[testing_iteration]
            test_image = mnist_data.train_images[testing_iteration] #DEBUG
            CPU_FF_output = NN_on_CPU.feedforward(nn, test_image)
            CPU_sample_label = get_MNIST_class_from_output(CPU_FF_output[-1])
            ##if CPU_sample_label == mnist_data.test_labels[testing_iteration]:
            if CPU_sample_label == mnist_data.train_labels[testing_iteration]: #DEBUG
                number_of_correct_classifications += 1
            CPU_FF_labels.append(CPU_sample_label)

        fraction_of_correct_classifications = number_of_correct_classifications / len(mnist_data.test_images)
        ##aux_functions.write_to_output_file("epoch number:", epoch_number, ". CPU fraction of correct classifications:", fraction_of_correct_classifications)
        print_net_weights_to_files(nn, net_name, fraction_of_correct_classifications)
        ##print("epoch number:", epoch_number, ". CPU fraction of correct classifications:", fraction_of_correct_classifications)
        print("CPU number of correct classifications:", number_of_correct_classifications) #DEBUG


#----- Execute -----#
train_MNIST_and_extract_weights()