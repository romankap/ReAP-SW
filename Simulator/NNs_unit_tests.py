import os,sys
import random

import ReCAM, Simulator
import NeuralNetwork
from NumberFormats import FixedPoint
import NNs_on_ReCAM, NNs_on_CPU
#import numpy
import random


'''lib_path = os.path.abspath(os.path.join('swalign-0.3.3'))
sys.path.append(lib_path)
import swalign'''

def compare_activation_vectors(ReCAM_matrix, CPU_matrix, type=""):
    index_in_ReCAM = 0

    for layer_index in reversed(range(1, len(CPU_matrix))):
        for neuron_index in range(len(CPU_matrix[layer_index])):
            #if ReCAM_pds[index_in_ReCAM] != CPU_pds[layer_index][neuron_index][weight_index]:
            if ReCAM_matrix[layer_index][neuron_index] != CPU_matrix[layer_index][neuron_index]:
                print("")
                print("Index in ReCAM: {}. CPU[{}][{}]".format(index_in_ReCAM, layer_index, neuron_index))
                exit("ERROR: Mismatching ReCAM and CPU " +  type + "!")

                index_in_ReCAM += 1

    print("VVV ReCAM and CPU " + type + " match VVV")

def compare_NN_matrices(ReCAM_matrix, CPU_matrix, type=""):
    index_in_ReCAM = 0

    for layer_index in reversed(range(1, len(CPU_matrix))):
        for neuron_index in range(len(CPU_matrix[layer_index])):
            weights_per_neuron = len(CPU_matrix[layer_index][0])

            for weight_index in range(weights_per_neuron):
                #if ReCAM_pds[index_in_ReCAM] != CPU_pds[layer_index][neuron_index][weight_index]:
                if ReCAM_matrix[layer_index][neuron_index][weight_index] != CPU_matrix[layer_index][neuron_index][weight_index]:
                    print("")
                    print("Index in ReCAM: {}. CPU[{}][{}][{}]".format(index_in_ReCAM, layer_index, neuron_index, weight_index))
                    exit("ERROR: Mismatching ReCAM and CPU " +  type + "!")

                index_in_ReCAM += 1

    print("VVV ReCAM and CPU " + type + " match VVV")


def test():
    nn_input_size = 3 # actual input length will be +1 due to bias
    fixed_point_10bit_precision = FixedPoint.FixedPointFormat(6,10)
    nn = NeuralNetwork.createDemoFullyConnectNN(fixed_point_10bit_precision, nn_input_size)
    ##nn = NeuralNetwork.createDebugNN(fixed_point_10bit_precision, nn_input_size) #DEBUG
    NN_on_CPU = NNs_on_CPU.initialize_NN_on_CPU(nn, fixed_point_10bit_precision)

    input_vectors = []
    input_vectors.append([3]*nn_input_size)
    input_vectors.append([1]*nn_input_size)
    target_vectors = []
    target_vectors.append([1, 0, 0])
    target_vectors.append([1, 0, 0])

    learning_rate = 0.02
    mini_batch_size = 10

    # --- CPU ---#
    NN_on_CPU.set_SGD_parameters(nn, mini_batch_size, learning_rate)

    #--- ReCAM ---#
    nn_weights_column = 0
    nn_start_row = 0
    NN_on_ReCAM = NNs_on_ReCAM.initialize_NN_on_ReCAM(nn_weights_column, nn_start_row)
    NN_on_ReCAM.set_SGD_parameters(mini_batch_size, learning_rate)

    NN_on_ReCAM.loadNNtoStorage(nn)

    for training_iteration in range(1000):
        ##input_vector = input_vectors[0]
        input_vector = input_vectors[training_iteration % 2]
        ##target_vector = target_vectors[0]
        target_vector = target_vectors[training_iteration % 2]
        #--- ReCAM ---#
        (ReCAM_activations, ReCAM_deltas) = NN_on_ReCAM.SGD_train(nn, fixed_point_10bit_precision, nn_input_size, input_vector, target_vector) #DEBUG
        ##NN_on_ReCAM.SGD_train(nn, fixed_point_10bit_precision, nn_input_size, input_vector, target_output)

        ##ReCAM_pds = NN_on_ReCAM.get_NN_matrices(nn, NN_on_ReCAM.BP_partial_derivatives_column, nn_start_row) #DEBUG

        ReCAM_weights = NN_on_ReCAM.get_NN_matrices(nn, nn_weights_column, nn_start_row)
        print("Finished ReCAM Execution", training_iteration)

        #--- CPU ---#
        (CPU_pds, CPU_activations, CPU_deltas) = NN_on_CPU.SGD_train(nn, input_vector, target_vector) #DEBUG
        ###NN_on_CPU.SGD_train(nn, input_vector, target_output)
        print("Finished CPU Execution", training_iteration)

        # --- Verify weights match ---#
        compare_activation_vectors(ReCAM_activations, CPU_activations, "activations") #DEBUG
        compare_activation_vectors(ReCAM_deltas, CPU_deltas, "deltas")  #DEBUG
        #compare_NN_matrices(ReCAM_pds, CPU_pds, "partial derivatives") #DEBUG
        compare_NN_matrices(ReCAM_weights, nn.weightsMatrices, "weights") #DEBUG
        #storage.printArray()


#################################
####         Execute         ####
#################################

#test()