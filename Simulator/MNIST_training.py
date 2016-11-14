import os,sys
import random
import struct
import copy
from array import array

import aux_functions
import ReCAM, Simulator
import NeuralNetwork
from NumberFormats import FixedPoint
import NNs_on_ReCAM, NNs_on_CPU
from MNIST_class import MNIST
import datetime

MNIST_path = 'C:\Dev\MNIST'
#output_file = None


################# MNIST #################
def get_MNIST_class_from_output(output_array):
    return output_array.index(max(output_array))

def train_MNIST():
    #Load MNIST data
    aux_functions.open_output_file(MNIST_path + '\\Outputs\\')
    mnist_data = MNIST(MNIST_path)
    mnist_data.load_training()
    mnist_data.load_testing()

    nn_input_size = 784  # actual input length will be +1 due to bias
    fixed_point_10bit_precision = FixedPoint.FixedPointFormat(6, 10)
    nn = NeuralNetwork.createMNISTFullyConnectNN(fixed_point_10bit_precision, nn_input_size)
    NN_on_CPU = NNs_on_CPU.initialize_NN_on_CPU(nn, fixed_point_10bit_precision)

    learning_rate = 0.02
    mini_batch_size = 50

    # --- CPU ---#
    NN_on_CPU.set_SGD_parameters(nn, mini_batch_size, learning_rate)

    # --- ReCAM ---#
    nn_weights_column = 0
    nn_start_row = 0
    ReCAM_size = 419430400
    NN_on_ReCAM = NNs_on_ReCAM.initialize_NN_on_ReCAM(nn_weights_column, nn_start_row, ReCAM_size)
    NN_on_ReCAM.set_SGD_parameters(mini_batch_size, learning_rate)

    NN_on_ReCAM.loadNNtoStorage(nn)

    total_training_epochs = 100
    for epoch_number in range(total_training_epochs):
        for training_iteration in range(len(mnist_data.train_images)):
            train_image = fixed_point_10bit_precision.convert_array_to_fixed_point(mnist_data.train_images[training_iteration])
            train_label = mnist_data.train_labels[training_iteration]
            target_output = [0] * 10
            target_output[train_label] = 1

            #--- ReCAM ---#
            ###NN_on_ReCAM.SGD_train(nn, fixed_point_10bit_precision, nn_input_size, train_image, target_output)
            ###ReCAM_weights = NN_on_ReCAM.get_NN_matrices(nn, nn_weights_column, nn_start_row)
            #print("Finished ReCAM Execution", training_iteration)

            #--- CPU ---#
            NN_on_CPU.SGD_train(nn, train_image, target_output)
            print("Finished CPU Execution", training_iteration)

            # --- Verify weights match ---#
            #NNs_unit_tests.compare_NN_matrices(ReCAM_weights, nn.weightsMatrices, "weights")
            aux_functions.write_to_output_file("Training iteration: ", training_iteration,
                                               ". Target output:", target_output)

        print("Training epoch: ", epoch_number)
        aux_functions.write_to_output_file("Training epoch: ", epoch_number)

        number_of_correct_classifications = 0
        for testing_iteration in range(len(mnist_data.test_images)):
            test_image = mnist_data.test_images[testing_iteration]
            ReCAM_FF_output = NN_on_ReCAM.get_feedforward_output(nn, fixed_point_10bit_precision, nn_input_size, test_image)
            ReCAM_sample_label = get_MNIST_class_from_output(ReCAM_FF_output)
            if ReCAM_sample_label == mnist_data.test_labels[testing_iteration]:
                number_of_correct_classifications += 1

        percentage_of_correct_classifications = number_of_correct_classifications / len(mnist_data.test_images)
        aux_functions.write_to_output_file("epoch number:", epoch_number, ". ReCAM percentage of correct classifications:", percentage_of_correct_classifications)

    aux_functions.close_output_file()

#----- Execute -----#
train_MNIST()