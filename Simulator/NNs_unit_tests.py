import os,sys
import random

import ReCAM, Simulator
import NeuralNetwork
from NumberFormats import FixedPoint
import NNs_on_ReCAM, NNs_on_CPU
import numpy
import random


'''lib_path = os.path.abspath(os.path.join('swalign-0.3.3'))
sys.path.append(lib_path)
import swalign'''

def compare_NN_matrices(ReCAM_matrix, CPU_matrix, type=""):
    index_in_ReCAM = 0

    for layer_index in range(1, len(CPU_matrix)):
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

    input_vector = [3]*(nn_input_size+1)
    input_vector[nn_input_size] = 1 #bias value
    target_output = [1, 2]
    learning_rate = 0.02

    #--- ReCAM ---#
    storage = ReCAM.ReCAM(2048)
    storage.setVerbose(False)

    nn_weights_column = 0
    nn_start_row = 0
    NNs_on_ReCAM.loadNNtoStorage(storage, nn, nn_weights_column, nn_start_row)

    input_column = 1
    FP_MUL_column = 2
    FP_accumulation_column = 3
    BP_deltas_column = 4
    BP_next_deltas_column = 5

    input_start_row = nn_start_row
    CPU_SGD_weights = [None]

    for training_iteration in range(2):
        NNs_on_ReCAM.loadInputToStorage(storage, fixed_point_10bit_precision, nn_input_size, input_column, input_start_row, input_vector)

        NNs_on_ReCAM.loadTargetOutputToStorage(storage, target_output, nn_start_row + nn.totalNumOfNetWeights, fixed_point_10bit_precision, nn_weights_column)

        (ReCAM_FP_output, ReCAM_FP_output_col_index) = NNs_on_ReCAM.feedforward(nn, storage, nn_weights_column, nn_start_row, input_column, FP_MUL_column, FP_accumulation_column)

        BP_output_column = ReCAM_FP_output_col_index
        BP_partial_derivatives_column = FP_MUL_column
        activations_column = 1 if ReCAM_FP_output_col_index==3 else 3

        NNs_on_ReCAM.backPropagation(nn, storage, nn_start_row, nn_weights_column, BP_output_column, BP_partial_derivatives_column,
                                    activations_column, BP_deltas_column, BP_next_deltas_column)
        ReCAM_pds = NNs_on_ReCAM.get_NN_matrices(nn, storage, BP_partial_derivatives_column, nn_start_row)

        NNs_on_ReCAM.update_weights(nn, storage, nn_start_row, nn_weights_column, BP_partial_derivatives_column, learning_rate)
        ReCAM_weights = NNs_on_ReCAM.get_NN_matrices(nn, storage, nn_weights_column, nn_start_row)
        print("Finished ReCAM Execution", training_iteration)

        #--- CPU ---#

        num_of_net_layers = len(nn.layers)
        CPU_activations = NNs_on_CPU.feedforward(nn, input_vector)
        CPU_pds = NNs_on_CPU.backPropagation(nn, CPU_activations, target_output)
        NNs_on_CPU.update_weights(nn, CPU_SGD_weights, CPU_pds, learning_rate)

        print(CPU_SGD_weights)
        print("Finished CPU Execution", training_iteration)

        # --- Verify partial derivatives match ---#

        compare_NN_matrices(ReCAM_pds, CPU_pds, "partial derivatives")
        compare_NN_matrices(ReCAM_weights, nn.weightsMatrices, "weights")
        storage.printArray()


#################################
####         Execute         ####
#################################

test()