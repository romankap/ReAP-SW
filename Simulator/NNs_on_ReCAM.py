import os, sys, time

import ReCAM, Simulator
import NeuralNetwork
from NumberFormats import FixedPoint
import random
import aux_functions


'''lib_path = os.path.abspath(os.path.join('swalign-0.3.3'))
sys.path.append(lib_path)
import swalign'''


''' Define a NN structure and the appropriate column to store.
Every layer has number of neurons - the net will be presented as a list.
First (last) layer is input (output) layer.
'''

def initialize_NN_on_ReCAM(nn_weights_column, nn_start_row, ReCAM_size=4096, is_activations_debug_active=False, is_pds_debug_active=False, is_deltas_debug_active=False):
    NN_on_ReCAM = ReCAM_NN_Manager(nn_weights_column, nn_start_row, ReCAM_size, is_activations_debug_active, is_pds_debug_active, is_deltas_debug_active)

    return NN_on_ReCAM

class ReCAM_NN_Manager:
    def __init__(self, nn_weights_column, nn_start_row, ReCAM_size=4096,
                 is_activations_debug_active=False, is_pds_debug_active=False, is_deltas_debug_active=False):
        self.storage = ReCAM.ReCAM(ReCAM_size)
        self.storage.setVerbose(False)

        self.nn_weights_column = nn_weights_column
        self.nn_start_row = nn_start_row

        self.nn_input_column = 1
        self.FF_MUL_column = 2
        self.FF_accumulation_column = 3
        self.activations_column = 0
        self.BP_deltas_column = 4
        self.BP_next_deltas_column = 5
        self.SGD_sums_column = 6
        self.BP_output_column = 0
        self.BP_partial_derivatives_column = self.FF_MUL_column

        self.SGD_mini_batch_size = 0
        self.learning_rate = 0.01
        self.samples_trained = 0
        self.epochs = 0
        self.samples_in_dataset = 0

        self.is_activations_debug_active = is_activations_debug_active
        self.is_pds_debug_active = is_pds_debug_active
        self.is_deltas_debug_active = is_deltas_debug_active

    def set_SGD_parameters(self, mini_batch_size, learning_rate):
        self.SGD_mini_batch_size = mini_batch_size
        self.one_over_SGD_mini_batch_size = 1 / mini_batch_size
        self.learning_rate = learning_rate

    ############################################################
    ######  Load NN to ReCAM
    ############################################################
    def loadNNtoStorage(self, nn):
        nn_row_in_ReCAM = self.nn_start_row

        for layer_index in range(1, len(nn.layers)):
            neurons_in_layer = len(nn.weightsMatrices[layer_index])
            weights_in_layer = len(nn.weightsMatrices[layer_index][0])

            for neuron_index in range(neurons_in_layer):
                self.storage.loadData(nn.weightsMatrices[layer_index][neuron_index], nn_row_in_ReCAM,
                                      nn.numbersFormat.total_bits, self.nn_weights_column, count_as_operation=False)
                nn_row_in_ReCAM += weights_in_layer

        if self.storage.verbose:
            self.storage.printArray()


    ############################################################
    ######  Load Input to ReCAM (read from file / generate)
    ############################################################
    def loadInputToStorage(self, input_format, input_size, input_column, input_start_row, input_vector=None):
        self.nn_input_column = input_column

        if not input_vector:
            random_input_vector = []
            for i in range(input_size):
                random_input_vector.append(input_format.convert(random.uniform(0.0001, 1)))
            #bias
                random_input_vector.append(1)
            self.storage.loadData(random_input_vector, input_start_row, input_format.total_bits, self.nn_input_column)

        else:
            print("Loading input from input_vector")
            input_vector.append(1) #Add bias to vector
            #Bias should be added manually (either here or in FP function)
            self.storage.loadData(input_vector, input_start_row, input_format.total_bits, self.nn_input_column)
            input_vector.pop()


    ############################################################
    ######  Extract NN structure in form of 3D matrices
    ############################################################
    def get_NN_matrices(self, nn, column, start_row):
        NN_from_ReCAM = [None]
        row_index = start_row

        for layer_index in range(1, len(nn.layers)):
            NN_from_ReCAM.append([])

            weights_per_neuron = len(nn.weightsMatrices[layer_index][0])
            for neuron_index in range(len(nn.weightsMatrices[layer_index])):
                NN_from_ReCAM[layer_index].append([])

                for weight_index in range(weights_per_neuron):
                    NN_from_ReCAM[layer_index][neuron_index].append(self.storage.crossbarArray[row_index][column])
                    row_index += 1

        return NN_from_ReCAM

    #####################################################################
    ######      Broadcast a single element to multiple ReCAM rows
    #####################################################################
    def loadTargetOutputToStorage(self, target_output, start_row, number_format):
        self.storage.loadData(target_output, start_row, number_format.total_bits, self.nn_weights_column)


    #####################################################################
    ######      Accumulate multiple sums in parallel, row-by-row
    #####################################################################
    def parallelAccumulate(self, col_A, col_B, res_col, start_row, rows_delta,
                           num_of_parallel_sums, num_of_accumulations_per_sum, number_format):
        #first iteration - only accumulate
        tagged_rows = range(start_row, start_row + num_of_parallel_sums*rows_delta, rows_delta)
        self.storage.taggedRowWiseOperation(col_A, col_B, res_col, tagged_rows, '+', number_format)

        start_row_to_accumulate = start_row
        for i in range(1, num_of_accumulations_per_sum):
            self.storage.tagRows(res_col)
            tagged_rows = range(start_row_to_accumulate, start_row_to_accumulate + num_of_parallel_sums*rows_delta, rows_delta)
            self.storage.shiftColumnOnTaggedRows(res_col, tagged_rows, distance_to_shift=1) #shift to higher-indexed rows

            start_row_to_accumulate = start_row + i
            self.storage.tagRows(res_col)
            tagged_rows = range(start_row_to_accumulate, start_row_to_accumulate + num_of_parallel_sums*rows_delta, rows_delta)
            self.storage.taggedRowWiseOperation(col_A, col_B, res_col, tagged_rows, '+', number_format)

        for i in range(1, num_of_parallel_sums+1):
            self.storage.tagRows(res_col)
            accumulation_result_row = start_row + i*num_of_accumulations_per_sum - 1
            final_output_destination_row = start_row + num_of_parallel_sums*num_of_accumulations_per_sum + i-1
            self.storage.broadcastDataElement(res_col, accumulation_result_row, final_output_destination_row, res_col, 0, 1)

    #####################################################################
    ######      Accumulate multiple sums in parallel, row-by-row
    #####################################################################
    def tagThenAccumulate(self, col_A, res_col, start_row, num_of_sums, num_of_accumulations_per_sum, numbers_format):
        accumulation_start_row = start_row

        for i in range(0, num_of_sums):
            # 1. Tag 'num_of_accumulations_per_sum' rows
            self.storage.tagRows(res_col)
            tagged_accumulation_rows = range(accumulation_start_row, accumulation_start_row + num_of_accumulations_per_sum)

            # 2. Send to rows_range to reduction tree, then write result to output row in output column
            is_first_accumulation = True if i == 0 else False

            reduction_output_destination_row = start_row + num_of_sums * num_of_accumulations_per_sum + i
            self.storage.pipelinedReduction(tagged_accumulation_rows, col_A, reduction_output_destination_row, res_col,
                                            is_first_accumulation, numbers_format)

            # 5. Move accumulation start row to following rows
            accumulation_start_row += num_of_accumulations_per_sum

    ############################################################
    ######  Feedforward an input through the net
    ############################################################
    def feedforward_WO_activation_function(self, nn, layer_index, start_row, ACC_result_col, activations_col):
        bias = [1]
        neurons_in_layer = len(nn.weightsMatrices[layer_index])
        weights_per_neuron = len(nn.weightsMatrices[layer_index][0])
        layer_total_weights = neurons_in_layer * weights_per_neuron
        zero_vector = [0] * layer_total_weights

        if self.storage.verbose:
            self.storage.printArray(msg=("beginning of feedforward iteration, layer " + str(layer_index)))
        # 1) Broadcast
        # Load bias
        activations_from_prev_layer = nn.layers[layer_index - 1][1]
        self.storage.loadData(bias, start_row + activations_from_prev_layer, nn.numbersFormat.total_bits,
                              activations_col)

        broadcast_start_row = start_row + weights_per_neuron  # first instance of input is aligned with first neuron weights
        self.storage.broadcastData(activations_col, start_row, weights_per_neuron,
                                   broadcast_start_row, 1, activations_col, weights_per_neuron,
                                   neurons_in_layer - 1)  # first appearance of input is already aligned with first neuron weights

        if self.storage.verbose:
            self.storage.printArray(msg="after broadcast")

        # 2) MUL
        self.storage.loadData(zero_vector, start_row, nn.numbersFormat.total_bits, self.FF_MUL_column, count_as_operation=False)
        #self.storage.MULConsecutiveRows(start_row, start_row + layer_total_weights - 1, self.FF_MUL_column, self.nn_weights_column, activations_col, nn.numbersFormat)
        self.storage.rowWiseOperation(self.nn_weights_column, activations_col, self.FF_MUL_column,
                              start_row, start_row + layer_total_weights - 1, '*', nn.numbersFormat)

        if self.storage.verbose:
            self.storage.printArray(msg="after MUL")

        # 3) Accumulate

        self.storage.loadData(zero_vector, start_row, nn.numbersFormat.total_bits, ACC_result_col, count_as_operation=False)

        self.tagThenAccumulate(self.FF_MUL_column, ACC_result_col, start_row, neurons_in_layer, weights_per_neuron, nn.numbersFormat)

        # Old inefficient version
        #self.parallelAccumulate(self.FF_MUL_column, ACC_result_col, ACC_result_col, start_row, weights_per_neuron,
        #                        neurons_in_layer, weights_per_neuron, nn.numbersFormat)


    ############################################################
    ######  Feedforward an input through the net
    ############################################################
    def feedforward_FC_layer(self, nn, layer_index, start_row, ACC_result_col, activations_col):
        neurons_in_layer = len(nn.weightsMatrices[layer_index])
        weights_per_neuron = len(nn.weightsMatrices[layer_index][0])
        layer_total_weights = neurons_in_layer * weights_per_neuron

        self.feedforward_WO_activation_function(nn, layer_index, start_row, ACC_result_col, activations_col)

        # ReLU: Applying 'max' function for all feedforward outputs
        self.storage.rowWiseOperationWithConstant(ACC_result_col, 0, ACC_result_col,
                              start_row + layer_total_weights, start_row + layer_total_weights+neurons_in_layer-1, 'max', nn.numbersFormat)

    ############################################################
    ######  Get softmax outputs
    ############################################################
    def feedforward_softmax_layer(self, nn, layer_index, start_row, ACC_result_col, activations_col):

        neurons_in_layer = len(nn.weightsMatrices[layer_index])
        weights_per_neuron = len(nn.weightsMatrices[layer_index][0])
        layer_total_weights = neurons_in_layer * weights_per_neuron

        self.feedforward_WO_activation_function(nn, layer_index, start_row, ACC_result_col, activations_col)

        sums_start_row = start_row + layer_total_weights
        sums_end_row = sums_start_row + neurons_in_layer-1
        # 1. Get max of all sums
        max_weighted_sum = self.storage.getScalarFromColumn(ACC_result_col, sums_start_row, sums_end_row, 'max', nn.numbersFormat)[0]
        # Softmax is the last layer ==> activations_col is not in use.
        self.storage.rowWiseOperationWithConstant(ACC_result_col, max_weighted_sum, activations_col,
                                                  sums_start_row, sums_end_row, '-', nn.numbersFormat)

        # Extract sums from ReCAM to send to CPU
        sums_vector_from_ReCAM = []
        for i in range(neurons_in_layer):
            sums_vector_from_ReCAM.append(self.storage.crossbarArray[sums_start_row + i][activations_col])

        # 2. Send sums + max to CPU
        self.storage.calculateSoftmaxOnCPU(sums_vector_from_ReCAM, nn.numbersFormat)
        self.storage.loadData(sums_vector_from_ReCAM, sums_start_row, nn.numbersFormat.total_bits, ACC_result_col)

        # 3. Get activations vector

    ############################################################
    ######  Feedforward an input through the net
    ############################################################
    def feedforward(self, nn):
        number_of_nn_layers = len(nn.layers)
        start_row = self.nn_start_row
        activations_col = self.nn_input_column
        ACC_result_col = self.FF_accumulation_column

        if self.is_activations_debug_active:    #DEBUG
            activations_to_return = [[] for x in range(number_of_nn_layers)]

        table_header_row = ["NN", "input", "MUL", "ACC"]
        self.storage.setPrintHeader(table_header_row)

        if self.is_activations_debug_active:    #DEBUG
            for neuron_activation_index in range(len(nn.weightsMatrices[1][0])):
                activations_to_return[0].append(self.storage.crossbarArray[start_row + neuron_activation_index][activations_col])

        for layer_index in range(1, number_of_nn_layers):
            neurons_in_layer = len(nn.weightsMatrices[layer_index])
            weights_per_neuron = len(nn.weightsMatrices[layer_index][0])
            layer_total_weights = neurons_in_layer * weights_per_neuron

            if nn.layers[layer_index][0] == "FC":
                self.feedforward_FC_layer(nn, layer_index, start_row, ACC_result_col, activations_col)
                start_row += layer_total_weights
            elif nn.layers[layer_index][0] == "softmax":
                self.feedforward_softmax_layer(nn, layer_index, start_row, ACC_result_col, activations_col)
                start_row += layer_total_weights

            activations_col, ACC_result_col = ACC_result_col, activations_col

            if self.is_activations_debug_active:  # DEBUG
                for neuron_activation_index in range(neurons_in_layer):
                    activations_to_return[layer_index].append(self.storage.crossbarArray[start_row + neuron_activation_index][activations_col])
                if layer_index != number_of_nn_layers-1:
                    activations_to_return[layer_index].append(1)

            if self.storage.verbose:
                self.storage.printArray(msg="feedforward Accumulate")

        output_col = activations_col #after each iteration, activations_col holds layer output
        net_output = []
        for i in range(nn.layers[number_of_nn_layers-1][1]):
            net_output.append(self.storage.crossbarArray[start_row+i][output_col])
        print("")
        print("=== NN output is: ", net_output)
        aux_functions.write_to_output_file("=== NN output is: ", net_output)

        if self.is_activations_debug_active:
            return (net_output, output_col, activations_to_return)
        return net_output, output_col

    ############################################################
    ######  Exectute feedforward and return net's output
    ############################################################
    def get_feedforward_output(self, nn, number_format, nn_input_size, input_vector):
        #1. load input
        self.loadInputToStorage(number_format, nn_input_size, self.nn_input_column, self.nn_start_row, input_vector)

        #3. feedforward
        if self.is_activations_debug_active:    #DEBUG
            ReCAM_FF_output, ReCAM_FF_output_col_index, ReCAM_activations = self.feedforward(nn) #DEBUG
        else:
            ReCAM_FF_output, ReCAM_FF_output_col_index = self.feedforward(nn)

        return ReCAM_FF_output
    ############################################################
    ######  Backward propagation of an output through the net
    ############################################################
    def backPropagation(self, nn, activations_col):
        if self.is_deltas_debug_active: #DEBUG
            deltas = [[] for x in range(len(nn.layers))]
            deltas[0] = None
        output_col = self.BP_output_column
        deltas_col = self.BP_deltas_column
        next_deltas_col = self.BP_next_deltas_column
        partial_derivatives_col = self.BP_partial_derivatives_column
        num_of_layers = len(nn.layers)

        if self.samples_trained == 0:
            zero_vector = [0] * nn.totalNumOfNetWeights
            self.storage.loadData(zero_vector, self.nn_start_row, nn.numbersFormat.total_bits, deltas_col, count_as_operation=False)
            self.storage.loadData(zero_vector, self.nn_start_row, nn.numbersFormat.total_bits, next_deltas_col, count_as_operation=False)

        output_start_row = self.nn_start_row + nn.totalNumOfNetWeights
        neurons_in_output_layer = nn.layers[num_of_layers - 1][1]
        if nn.layers[num_of_layers-1][0] == "FC":
            # 1. Calculate deltas for all outputs
            self.storage.rowWiseOperation(output_col, self.nn_weights_column, next_deltas_col,
                                            output_start_row, output_start_row + neurons_in_output_layer - 1, '-',
                                            nn.numbersFormat)
            # 2. tag rows with activations equal to zero
            tagged_rows_list = self.storage.tagRowsEqualToConstant(output_col, 0.0, output_start_row, output_start_row+neurons_in_output_layer-1)

            # 3. Write zero to tagged rows in delta column
            self.storage.taggedRowWiseOperationWithConstant(next_deltas_col, 0.0, next_deltas_col, tagged_rows_list, 'write', nn.numbersFormat)
        elif nn.layers[num_of_layers - 1][0] == "softmax":
            # softmax delta - subtract output activations and target
            # Target is in weights column
            self.storage.rowWiseOperation(output_col, self.nn_weights_column, next_deltas_col,
                                          output_start_row, output_start_row + neurons_in_output_layer - 1, '-', nn.numbersFormat)

        for layer_index in reversed(range(1, num_of_layers)):
            # Layer structure
            neurons_in_layer = len(nn.weightsMatrices[layer_index])
            weights_per_neuron = len(nn.weightsMatrices[layer_index][0])
            total_layer_weights = neurons_in_layer * weights_per_neuron
            layer_start_row = output_start_row - total_layer_weights

            if self.is_deltas_debug_active:  # DEBUG
                for neuron_index in range(neurons_in_layer):
                    deltas[layer_index].append(self.storage.crossbarArray[output_start_row+neuron_index][next_deltas_col])
            # Get layer deltas to 'deltas_col'
            self.storage.broadcastData(next_deltas_col, output_start_row, neurons_in_layer,
                          layer_start_row, weights_per_neuron, deltas_col, 1, weights_per_neuron)

            # Calculate partial derivatives for each weight
            self.storage.rowWiseOperation(deltas_col, activations_col, partial_derivatives_col,
                                     layer_start_row, layer_start_row + total_layer_weights-1, '*', nn.numbersFormat)


            # Calculating delta(prev_layer) = delta(curr_layer)*W(curr_layer)
            if layer_index!=1:     # No need for next_layer_delta in first hidden layer
                self.storage.rowWiseOperation(deltas_col, self.nn_weights_column, deltas_col,
                                         layer_start_row, layer_start_row + total_layer_weights-1, '*', nn.numbersFormat)

                next_deltas_sum_start_row = output_start_row - weights_per_neuron

                #----- Reduction-tree sum on all i-th deltas -----(all 1st deltas belong to 1st sum, 2nd deltas to 2nd sum, ...)
                # Input: Column=deltas_col, rows= next_deltas_sum_start_row, hops=weights_per_neuron
                # Output: Column = next_deltas_col, rows=output_start_row - neurons_per_layer*weights_per_neuron

                for weight_index in range(weights_per_neuron):
                    reduction_sum_rows = range(layer_start_row+weight_index, output_start_row-1, weights_per_neuron)
                    is_first_reduction = True if weight_index == 0 else False
                    self.storage.pipelinedReduction(reduction_sum_rows, deltas_col, layer_start_row+weight_index, next_deltas_col,
                                                    is_first_reduction, nn.numbersFormat)

                #----- Shift-then-add parallel accumulation -----

                # TODO: Add 'write' operation in rowWiseOperation
                # Copying from deltas_col to next_deltas_col

                # self.storage.rowWiseOperation(deltas_col, deltas_col, next_deltas_col,
                #                          next_deltas_sum_start_row, next_deltas_sum_start_row + weights_per_neuron-1, 'max', nn.numbersFormat)
                #
                # for neuron in range(neurons_in_layer-1):
                #     rows_to_shift = range(next_deltas_sum_start_row, next_deltas_sum_start_row + weights_per_neuron)
                #     self.storage.shiftColumnOnTaggedRows(next_deltas_col, rows_to_shift, -weights_per_neuron)
                #
                #     next_deltas_sum_start_row -= weights_per_neuron
                #     self.storage.rowWiseOperation(deltas_col, next_deltas_col, next_deltas_col,
                #                             next_deltas_sum_start_row, next_deltas_sum_start_row+weights_per_neuron-1, '+', nn.numbersFormat)

                #Checking whether the next layer requires ReLU deltas
                if nn.layers[layer_index-1][0] == "FC":
                    neurons_in_prev_layer = nn.layers[layer_index-1][1]
                    # 2. tag rows with activations equal to zero
                    tagged_rows_list = self.storage.tagRowsEqualToConstant(activations_col, 0.0, layer_start_row,
                                                                           layer_start_row + neurons_in_prev_layer - 1)

                    # 3. Write zero to tagged rows in delta column
                    self.storage.taggedRowWiseOperationWithConstant(next_deltas_col, 0.0, next_deltas_col,
                                                                    tagged_rows_list, 'write', nn.numbersFormat)

            output_start_row -= total_layer_weights
            output_col, activations_col = activations_col, output_col

        if self.is_deltas_debug_active: #DEBUG
            return deltas
    ############################################################
    ######  Update net weights - all in parallel
    ############################################################
    def update_weights(self, nn, nn_weights_column, partial_derivatives_column, learning_rate=0.05):
        output_start_row = self.nn_start_row + nn.totalNumOfNetWeights
        self.storage.rowWiseOperationWithConstant(partial_derivatives_column, learning_rate, partial_derivatives_column,
                                                  self.nn_start_row, output_start_row-1, '*', nn.numbersFormat)

        self.storage.rowWiseOperation(nn_weights_column, partial_derivatives_column, nn_weights_column,
                                      self.nn_start_row, output_start_row-1, '-', nn.numbersFormat)
        print("Updated NN weights in ReCAM")


    ############################################################
    ######  Update net weights - all weights in parallel
    ############################################################
    def SGD_on_single_sample(self, nn):
        partial_derivatives_column = self.BP_partial_derivatives_column
        SGD_sums_column = self.SGD_sums_column

        table_header_row = ["NN", "Out/Activ", "dErr/dW", "Activ/Out", "deltas", "next layer"]
        self.storage.setPrintHeader(table_header_row)

        if self.samples_trained == 0:
            zero_vector = [0] * nn.totalNumOfNetWeights
            self.storage.loadData(zero_vector, self.nn_start_row, nn.numbersFormat.total_bits, SGD_sums_column, count_as_operation=False)

        output_start_row = self.nn_start_row + nn.totalNumOfNetWeights
        if self.samples_trained % self.SGD_mini_batch_size == 0:  # First sample in mini-batch
            self.storage.rowWiseOperationWithConstant(partial_derivatives_column, self.learning_rate, SGD_sums_column,
                                                      self.nn_start_row, output_start_row - 1, '*', nn.numbersFormat)
        else:
            self.storage.rowWiseOperationWithConstant(partial_derivatives_column, self.learning_rate,
                                                      partial_derivatives_column,
                                                      self.nn_start_row, output_start_row - 1, '*', nn.numbersFormat)
            self.storage.rowWiseOperation(SGD_sums_column, partial_derivatives_column, SGD_sums_column,
                                          self.nn_start_row, output_start_row - 1, '+', nn.numbersFormat)

        self.samples_trained += 1
        if self.samples_trained % self.SGD_mini_batch_size==0:
            self.storage.rowWiseOperationWithConstant(SGD_sums_column, self.one_over_SGD_mini_batch_size, SGD_sums_column,
                                                      self.nn_start_row, output_start_row - 1, '*', nn.numbersFormat)
            self.storage.rowWiseOperation(self.nn_weights_column, SGD_sums_column, self.nn_weights_column,
                                          self.nn_start_row, output_start_row - 1, '-', nn.numbersFormat)

        print("Finished sample", self.samples_trained, "with a mini-batch size of", self.SGD_mini_batch_size)

    ############################################################
    ######  Complete training iteration on a single sample
    ############################################################
    def SGD_train(self, nn, number_format, nn_input_size, input_vector, target_output):
        #1. load input
        self.loadInputToStorage(number_format, nn_input_size, self.nn_input_column, self.nn_start_row, input_vector)

        #2. load target output
        self.loadTargetOutputToStorage(target_output, self.nn_start_row + nn.totalNumOfNetWeights, number_format)

        #3. feedforward
        self.storage.set_histogram_scope("feedforward")
        if self.is_activations_debug_active:    #DEBUG
            ReCAM_FP_output, ReCAM_FF_output_col_index, ReCAM_activations = self.feedforward(nn)
        else:
            ReCAM_FP_output, ReCAM_FF_output_col_index = self.feedforward(nn)

        #4. backpropagation
        self.storage.set_histogram_scope("backprop")
        self.BP_output_column = ReCAM_FF_output_col_index
        activations_column = self.nn_input_column if ReCAM_FF_output_col_index == self.FF_accumulation_column else self.FF_accumulation_column

        if self.is_deltas_debug_active: # DEBUG
            ReCAM_deltas = self.backPropagation(nn, activations_column)
        else:
            self.backPropagation(nn, activations_column)

        self.storage.remove_histogram_scope()
        #5. SGD on a single sample
        self.SGD_on_single_sample(nn)

        if self.is_activations_debug_active and self.is_deltas_debug_active:    #DEBUG
            return (ReCAM_activations, ReCAM_deltas)
        elif self.is_activations_debug_active and not self.is_deltas_debug_active:    #DEBUG
            return (ReCAM_activations, None)
        elif not self.is_activations_debug_active and self.is_deltas_debug_active:  # DEBUG
            return (None, ReCAM_deltas)


############################################################
######  Test function
############################################################
def NN_on_ReCAM_test():
    NN_Manager = initialize_NN_on_ReCAM()

    nn_input_size = 3 # actual input length will be +1 due to bias
    fixed_point_10bit = FixedPoint.FixedPointFormat(6,10)
    nn = NeuralNetwork.createDemoFullyConnectNN(fixed_point_10bit, nn_input_size)

    nn_weights_column = 0
    nn_start_row = 0
    NN_Manager.loadNNtoStorage(nn, nn_start_row, nn_weights_column)

    input_column = 1
    FP_MUL_column = 2
    FP_accumulation_column = 3
    input_start_row = nn_start_row
    NN_Manager.loadInputToStorage(fixed_point_10bit, nn_input_size, input_column, input_start_row)

    target_output = [1,2]
    NN_Manager.loadTargetOutputToStorage(target_output, nn_start_row + nn.totalNumOfNetWeights, fixed_point_10bit, nn_weights_column)

    FP_output = NN_Manager.feedforward(nn, nn_weights_column, nn_start_row, input_column, FP_MUL_column, FP_accumulation_column)

    BP_output_column = FP_output[1]
    BP_partial_derivatives_column = FP_MUL_column
    activations_column = 1 if BP_output_column==3 else 3
    BP_deltas_column = 4
    BP_next_deltas_column = 5

    NN_Manager.backPropagation(nn, nn_start_row, nn_weights_column, BP_output_column, BP_partial_derivatives_column,
                    activations_column, BP_deltas_column, BP_next_deltas_column)


############################################################
######  Execute
############################################################
#NN_on_ReCAM_test()