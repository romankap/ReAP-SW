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
    NN_on_CPU = NNs_on_CPU.initialize_NN_on_CPU(fixed_point_10bit_precision)

    input_vector = [3]*(nn_input_size+1)
    input_vector[nn_input_size] = 1 #bias value
    target_output = [1, 2]
    learning_rate = 0.02
    mini_batch_size = 2

    # --- CPU ---#
    NN_on_CPU.set_SGD_parameters(nn, mini_batch_size, learning_rate)

    #--- ReCAM ---#
    NN_on_ReCAM = NNs_on_ReCAM.initialize_NN_on_ReCAM()
    NN_on_ReCAM.set_SGD_parameters(mini_batch_size, learning_rate)

    nn_weights_column = 0
    nn_start_row = 0
    NN_on_ReCAM.loadNNtoStorage(nn, nn_weights_column, nn_start_row)

    input_column = 1
    FP_MUL_column = 2
    FP_accumulation_column = 3
    BP_deltas_column = 4
    BP_next_deltas_column = 5
    SGD_sums_column = 6

    input_start_row = nn_start_row

    for training_iteration in range(4):
        #--- ReCAM ---#
        NN_on_ReCAM.loadInputToStorage(fixed_point_10bit_precision, nn_input_size, input_column, input_start_row, input_vector)

        NN_on_ReCAM.loadTargetOutputToStorage(target_output, nn_start_row + nn.totalNumOfNetWeights, fixed_point_10bit_precision)

        (ReCAM_FP_output, ReCAM_FP_output_col_index) = NN_on_ReCAM.feedforward(nn, FP_MUL_column, FP_accumulation_column)

        BP_output_column = ReCAM_FP_output_col_index
        BP_partial_derivatives_column = FP_MUL_column
        activations_column = 1 if ReCAM_FP_output_col_index==3 else 3

        NN_on_ReCAM.backPropagation(nn, BP_output_column, BP_partial_derivatives_column,
                                    activations_column, BP_deltas_column, BP_next_deltas_column)
        ReCAM_pds = NN_on_ReCAM.get_NN_matrices(nn, BP_partial_derivatives_column, nn_start_row)

        #NN_on_ReCAM.update_weights(nn, nn_start_row, nn_weights_column, BP_partial_derivatives_column, learning_rate)
        NN_on_ReCAM.SGD_train(nn, BP_partial_derivatives_column, BP_deltas_column, BP_next_deltas_column, SGD_sums_column)
        ReCAM_weights = NN_on_ReCAM.get_NN_matrices(nn, nn_weights_column, nn_start_row)
        print("Finished ReCAM Execution", training_iteration)

        #--- CPU ---#
        NN_on_CPU.SGD_train(nn, input_vector, target_output)
        print("Finished CPU Execution", training_iteration)

        # --- Verify weights match ---#
        compare_NN_matrices(ReCAM_weights, nn.weightsMatrices, "weights")
        #storage.printArray()


#################################
####         Execute         ####
#################################

test()