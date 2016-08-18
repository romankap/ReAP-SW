import os,sys
import random

import ReCAM, Simulator
import NeuralNetwork
from NumberFormats import FixedPoint
import NNs_on_ReCAM, NNs_on_CPU
import random


'''lib_path = os.path.abspath(os.path.join('swalign-0.3.3'))
sys.path.append(lib_path)
import swalign'''


def test():
    nn_input_size = 3 # actual input length will be +1 due to bias
    fixed_point_10bit = FixedPoint.FixedPointFormat(6,10)
    nn = NeuralNetwork.createDemoFullyConnectNN(fixed_point_10bit, nn_input_size)
    input_vector = NeuralNetwork.generateRandomInput(nn_input_size, fixed_point_10bit)

    ############################################################
    ####       ReCAM
    ############################################################
    storage = ReCAM.ReCAM(1024)
    storage.setVerbose(True)

    nn_weights_column = 0
    nn_start_row = 0
    NNs_on_ReCAM.loadNNtoStorage(storage, nn, nn_weights_column, nn_start_row)

    input_column = 1
    FP_MUL_column = 2
    FP_accumulation_column = 3
    input_start_row = nn_start_row
    NNs_on_ReCAM.loadInputToStorage(storage, fixed_point_10bit, nn_input_size, input_column, input_start_row, input_vector)

    target_output = [1,2]
    NNs_on_ReCAM.loadTargetOutputToStorage(storage, target_output, nn_start_row + nn.totalNumOfNetWeights, fixed_point_10bit, nn_weights_column)

    ReCAM_FP_output_column = NNs_on_ReCAM.forwardPropagation(nn, storage, nn_weights_column, nn_start_row, input_column, FP_MUL_column, FP_accumulation_column)

    # BP_output_column = FP_output_column
    # BP_partial_derivatives_column = FP_MUL_column
    # activations_column = 1 if FP_output_column==3 else 3
    # BP_deltas_column = 4
    # BP_next_deltas_column = 5
    #
    # NNs_on_ReCAM.backPropagation(nn, storage, nn_start_row, nn_weights_column, BP_output_column, BP_partial_derivatives_column,
    #                 activations_column, BP_deltas_column, BP_next_deltas_column)
    print("Finished ReCAM Execution")

    ############################################################
    ####       CPU
    ############################################################
    CPU_FP_output = NNs_on_CPU.forwardPropagation(nn, input_vector, fixed_point_10bit)
    print("Finished CPU Execution")

test()