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


def test():
    nn_input_size = 3 # actual input length will be +1 due to bias
    fixed_point_10bit_precision = FixedPoint.FixedPointFormat(6,10)
    nn = NeuralNetwork.createDemoFullyConnectNN(fixed_point_10bit_precision, nn_input_size)

    input_vector = NeuralNetwork.generateRandomInput(nn_input_size, fixed_point_10bit_precision)
    target_output = [1, 2]

    ############################
    ####       ReCAM        ####
    ############################
    storage = ReCAM.ReCAM(1024)
    storage.setVerbose(True)

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

    ################################
    ####            CPU         ####
    ################################
    num_of_net_layers = len(nn.layers)
    CPU_activations = NNs_on_CPU.forwardPropagation(nn, input_vector, fixed_point_10bit_precision)

    NNs_on_CPU.backPropagation(nn, CPU_activations, target_output, fixed_point_10bit_precision)
    print("Finished CPU Execution")

    if ReCAM_FP_output == CPU_activations[num_of_net_layers-1]:
        print("VVV ReCAM and CPU FP outputs match VVV")
    else:
        print("--- ReCAM and CPU FP outputs DO NOT match!!!")
        print("ReCAM output, ", ReCAM_FP_output)
        print("CPU output, ", CPU_activations[num_of_net_layers-1])

test()