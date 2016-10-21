import os,sys
import random
import struct
import copy
from array import array

import ReCAM, Simulator
import NeuralNetwork
from NumberFormats import FixedPoint
import NNs_on_ReCAM, NNs_on_CPU
from MNIST_class import MNIST
import NNs_unit_tests
import datetime

MNIST_path = 'C:\Dev\MNIST'
output_file = None

################# AUX #################
def check_if_folder_exists_and_open(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def open_output_file():
    output_folder = MNIST_path + '\\Outputs\\'
    check_if_folder_exists_and_open(output_folder)

    now = str(datetime.datetime.now()).replace(':','-').replace(' ','_')
    full_output_filename = output_folder + now + ".txt"
    global output_file
    output_file = open(full_output_filename, 'w')


def write_to_output_file(*strings_to_write):
    global output_file
    print(*strings_to_write, file=output_file)
    output_file.flush()


def close_output_file():
    global output_file
    output_file.close()

################# MNIST #################
def get_MNIST_class_from_output(output_array):
    return output_array.index(max(output_array))

def train_MNIST():
    #Load MNIST data
    open_output_file()
    mnist_data = MNIST(MNIST_path)
    mnist_data.load_training()
    mnist_data.load_testing()

    nn_input_size = 784  # actual input length will be +1 due to bias
    fixed_point_10bit_precision = FixedPoint.FixedPointFormat(9, 7)
    nn = NeuralNetwork.createMNISTFullyConnectNN(fixed_point_10bit_precision, nn_input_size)
    NN_on_CPU = NNs_on_CPU.initialize_NN_on_CPU(fixed_point_10bit_precision)

    learning_rate = 0.02
    mini_batch_size = 10
    normalized_learning_rate = learning_rate / mini_batch_size

    # --- CPU ---#
    NN_on_CPU.set_SGD_parameters(nn, mini_batch_size, normalized_learning_rate)

    # --- ReCAM ---#
    nn_weights_column = 0
    nn_start_row = 0
    ReCAM_size = 524288
    NN_on_ReCAM = NNs_on_ReCAM.initialize_NN_on_ReCAM(nn_weights_column, nn_start_row, ReCAM_size)
    NN_on_ReCAM.set_SGD_parameters(mini_batch_size, normalized_learning_rate)

    NN_on_ReCAM.loadNNtoStorage(nn)

    total_training_epochs = 100
    for epoch_number in range(total_training_epochs):
        for training_iteration in range(len(mnist_data.train_images)):
            train_image = fixed_point_10bit_precision.convert_array_to_fixed_point(mnist_data.train_images[training_iteration])
            train_label = mnist_data.train_labels[training_iteration]
            target_output = [0] * 10
            target_output[train_label] = fixed_point_10bit_precision.get_max()

            #--- ReCAM ---#
            NN_on_ReCAM.SGD_train(nn, fixed_point_10bit_precision, nn_input_size, train_image, target_output)
            ReCAM_weights = NN_on_ReCAM.get_NN_matrices(nn, nn_weights_column, nn_start_row)
            #print("Finished ReCAM Execution", training_iteration)

            #--- CPU ---#
            #NN_on_CPU.SGD_train(nn, train_image, target_output)
            #print("Finished CPU Execution", training_iteration)

            # --- Verify weights match ---#
            #NNs_unit_tests.compare_NN_matrices(ReCAM_weights, nn.weightsMatrices, "weights")

        print("Training epoch: ", epoch_number)

        number_of_correct_classifications = 0
        for testing_iteration in range(len(mnist_data.test_images)):
            test_image = mnist_data.test_images[testing_iteration]
            ReCAM_FF_output = NN_on_ReCAM.get_feedforward_output(nn, fixed_point_10bit_precision, nn_input_size, test_image)
            ReCAM_sample_label = get_MNIST_class_from_output(ReCAM_FF_output)
            if ReCAM_sample_label == mnist_data.test_labels[testing_iteration]:
                number_of_correct_classifications += 1

        percentage_of_correct_classifications = number_of_correct_classifications / len(mnist_data.test_images)
        write_to_output_file("epoch number:", epoch_number, ". ReCAM percentage of correct classifications:", percentage_of_correct_classifications)

    close_output_file()

#----- Execute -----#
train_MNIST()