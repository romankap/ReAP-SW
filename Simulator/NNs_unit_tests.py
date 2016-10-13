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

def getReCAMpds(nn, storage, pds_column):
    ReCAM_pds = [None]
    row_index = 0

    for layer_index in range(1, len(nn.layers)):
        ReCAM_pds.append([])

        weights_per_neuron = len(nn.weightsMatrices[layer_index][0])
        for neuron_index in range(len(nn.weightsMatrices[layer_index])):
            ReCAM_pds[layer_index].append([])

            for weight_index in range(weights_per_neuron):
                ReCAM_pds[layer_index][neuron_index].append(storage.crossbarArray[row_index][pds_column])
                row_index += 1

    return ReCAM_pds


def compareReCAMandCPUpds(ReCAM_pds, CPU_pds):
    index_in_ReCAM = 0

    for layer_index in range(1, len(CPU_pds)):
        for neuron_index in range(len(CPU_pds[layer_index])):
            weights_per_neuron = len(CPU_pds[layer_index][0])

            for weight_index in range(weights_per_neuron):
                #if ReCAM_pds[index_in_ReCAM] != CPU_pds[layer_index][neuron_index][weight_index]:
                if ReCAM_pds[layer_index][neuron_index][weight_index] != CPU_pds[layer_index][neuron_index][weight_index]:
                    print("")
                    print("ERROR: Mismatching ReCAM and CPU pds!")
                    print("Index in ReCAM: {}. CPU[{}][{}][{}]".format(index_in_ReCAM, layer_index, neuron_index, weight_index))

                index_in_ReCAM += 1

    print("VVV ReCAM and CPU pds match VVV")


def test():
    nn_input_size = 3 # actual input length will be +1 due to bias
    fixed_point_10bit_precision = FixedPoint.FixedPointFormat(6,10)
    nn = NeuralNetwork.createDemoFullyConnectNN(fixed_point_10bit_precision, nn_input_size)

    input_vector = [1]*(nn_input_size+1)
    target_output = [1, 2]

    ############################
    ####       ReCAM        ####
    ############################
    storage = ReCAM.ReCAM(2048)
    storage.setVerbose(False)

    nn_weights_column = 0
    nn_start_row = 0
    NNs_on_ReCAM.loadNNtoStorage(storage, nn, nn_weights_column, nn_start_row)

    input_column = 1
    FP_MUL_column = 2
    FP_accumulation_column = 3
    input_start_row = nn_start_row
    NNs_on_ReCAM.loadInputToStorage(storage, fixed_point_10bit_precision, nn_input_size, input_column, input_start_row, input_vector)

    NNs_on_ReCAM.loadTargetOutputToStorage(storage, target_output, nn_start_row + nn.totalNumOfNetWeights, fixed_point_10bit_precision, nn_weights_column)

    (ReCAM_FP_output, ReCAM_FP_output_col_index) = NNs_on_ReCAM.forwardPropagation(nn, storage, nn_weights_column, nn_start_row, input_column, FP_MUL_column, FP_accumulation_column)

    BP_output_column = ReCAM_FP_output_col_index
    BP_partial_derivatives_column = FP_MUL_column
    activations_column = 1 if ReCAM_FP_output_col_index==3 else 3
    BP_deltas_column = 4
    BP_next_deltas_column = 5
    NNs_on_ReCAM.backPropagation(nn, storage, nn_start_row, nn_weights_column, BP_output_column, BP_partial_derivatives_column,
                                activations_column, BP_deltas_column, BP_next_deltas_column)
    print("Finished ReCAM Execution")

    ####################################
    ####            CPU             ####
    ####################################
    num_of_net_layers = len(nn.layers)
    CPU_activations = NNs_on_CPU.forwardPropagation(nn, input_vector, fixed_point_10bit_precision)

    CPU_pds = NNs_on_CPU.backPropagation(nn, CPU_activations, target_output, fixed_point_10bit_precision)

    NNs_on_CPU.update_weights(nn, CPU_pds, fixed_point_10bit_precision, 0.05)
    print("Finished CPU Execution")

    ################################################################
    ####            Verify partial derivatives match            ####
    ################################################################
    ReCAM_pds = getReCAMpds(nn, storage, BP_partial_derivatives_column)

    compareReCAMandCPUpds(ReCAM_pds, CPU_pds)



#################################
####         Execute         ####
#################################

test()