import os,sys
import random
import struct
import copy
from array import array

import aux_functions
import ReCAM, Simulator
import NeuralNetwork
from NumberFormats import FixedPoint
import NNs_on_ReCAM, NNs_on_CPU, NNs_unit_tests
from MNIST_class import MNIST
import datetime

MNIST_path = 'C:\Dev\MNIST'
#output_file = None

is_ReCAM_active = False
is_CPU_active = True
is_activations_debug_active = False
is_pds_debug_active = False
is_deltas_debug_active = False

################# MNIST #################
def get_MNIST_class_from_output(output_array):
    return output_array.index(max(output_array))

def train_MNIST():
    #Load MNIST data
    #aux_functions.open_output_file(MNIST_path + '\\Outputs\\')
    mnist_data = MNIST(MNIST_path)
    mnist_data.load_training()
    mnist_data.load_testing()

    learning_rate = 0.02
    mini_batch_size = 50
    nn_input_size = 784  # actual input length will be +1 due to bias
    fixed_point_10bit_precision = FixedPoint.FixedPointFormat(6, 10)
    ##nn = NeuralNetwork.createMNISTFullyConnectNN(fixed_point_10bit_precision, nn_input_size)
    ##nn = NeuralNetwork.createDemoFullyConnectNN(fixed_point_10bit_precision, nn_input_size)
    nn = NeuralNetwork.createMNISTConvNet(fixed_point_10bit_precision, nn_input_size)

    # --- CPU ---#
    if is_CPU_active:
        NN_on_CPU = NNs_on_CPU.initialize_NN_on_CPU(nn, is_activations_debug_active, is_pds_debug_active, is_deltas_debug_active, fixed_point_10bit_precision)
        NN_on_CPU.set_SGD_parameters(nn, mini_batch_size, learning_rate)

    # --- ReCAM ---#
    if is_ReCAM_active:
        nn_weights_column = 0
        nn_start_row = 0
        ReCAM_size = 419430400
        NN_on_ReCAM = NNs_on_ReCAM.initialize_NN_on_ReCAM(nn_weights_column, nn_start_row, ReCAM_size, is_activations_debug_active, is_pds_debug_active, is_deltas_debug_active)
        NN_on_ReCAM.set_SGD_parameters(mini_batch_size, learning_rate)

        NN_on_ReCAM.loadNNtoStorage(nn)

    total_training_epochs = 100
    epoch_number = 0
    training_iteration = 0
    #for epoch_number in range(total_training_epochs):
    #    for training_iteration in range(len(mnist_data.train_images)):
    train_image = fixed_point_10bit_precision.convert_array_to_fixed_point(mnist_data.train_images[training_iteration])
    train_label = mnist_data.train_labels[training_iteration]
    target_output = [0] * 10
    target_output[train_label] = 1

    #--- ReCAM ---#
    if is_ReCAM_active:
        if is_activations_debug_active or is_deltas_debug_active:
            (ReCAM_activations, ReCAM_deltas) = NN_on_ReCAM.SGD_train(nn, fixed_point_10bit_precision, nn_input_size, train_image, target_output)
        else:
            NN_on_ReCAM.SGD_train(nn, fixed_point_10bit_precision, nn_input_size, train_image, target_output)

    if is_CPU_active and is_ReCAM_active:
        ReCAM_weights = NN_on_ReCAM.get_NN_matrices(nn, nn_weights_column, nn_start_row)

    if is_pds_debug_active:
        ReCAM_pds = NN_on_ReCAM.get_NN_matrices(nn, NN_on_ReCAM.BP_partial_derivatives_column, nn_start_row)

    if is_ReCAM_active:
        NN_on_ReCAM.storage.printHistogramsToExcel(nn, len(mnist_data.train_images))

    print("Finished ReCAM Execution", training_iteration)

    #--- CPU ---#
    if is_CPU_active:
        if is_activations_debug_active or is_pds_debug_active or is_deltas_debug_active:
            (CPU_activations, CPU_pds, CPU_deltas) = NN_on_CPU.SGD_train(nn, train_image, target_output)
        else:
            NN_on_CPU.SGD_train(nn, train_image, target_output)

        print("Finished CPU Execution", training_iteration)
        # --- Verify weights match ---#
        if is_ReCAM_active:
            NNs_unit_tests.compare_NN_matrices(ReCAM_weights, nn.weightsMatrices, "weights")

        if is_activations_debug_active:
            NNs_unit_tests.compare_activation_vectors(ReCAM_activations, CPU_activations, "activations")
        if is_deltas_debug_active:
            NNs_unit_tests.compare_activation_vectors(ReCAM_deltas, CPU_deltas, "deltas")
        if is_pds_debug_active:
            NNs_unit_tests.compare_NN_matrices(ReCAM_pds, CPU_pds, "partial derivatives")

'''        aux_functions.write_to_output_file("Training iteration: ", training_iteration,
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
'''
#----- Execute -----#
train_MNIST()