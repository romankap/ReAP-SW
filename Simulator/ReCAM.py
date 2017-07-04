__author__ = 'Roman'

try:
    import matplotlib, numpy
    matplotlib.use('qt4agg')
    import matplotlib.pyplot as plt
    HAVE_MATPLOTLIB = True
except ImportError:
    HAVE_MATPLOTLIB = False

import math
import os
import sys
import datetime
import ProteinSequencing.BLOSUM62 as blosum62

lib_path = os.path.abspath(os.path.join('spfpm-1.1'))
sys.path.append(lib_path)
from tabulate import tabulate
import xlsxwriter

################################################
####        AUX functions & definitions
################################################
max_operation_string = "max"
copy_operation_string = "copy"
write_operation_string = "write"

#--- Instructions Names ---#
load_data_element_hist_name = 'load-data-element'
row_by_row_hist_name = 'vector-vector'
row_by_const_hist_name = 'vector-constant'
shift_operation_hist_name = 'shift-operation'
shifted_rows_num_hist_name = 'shifted-rows-num'
reduce_scalar_from_column_hist_name = 'reduce-scalar'
reduction_tree_sum = 'reduction-tree-sum'
is_reduction_tree_pipelined = True
broadcast = 'broadcast'

#--- CPU Instructions Constants ---#
CPU_softmax_cycles = 100

#--- Number Format Conversion ---#

def convert_if_needed(result, number_format=None):
    if number_format:
        return number_format.convert(result)
    return result

def convert_to_non_zero_if_needed(result, number_format=None):
    if number_format:
        return number_format.convert_to_non_zero(result)
    return result

################################################
####        ReCAM API
################################################
class ReCAM:
    def __init__(self, size_Bytes, bytesPerRow=32):
        self.sizeInBytes = size_Bytes
        self.bitsPerRow = bytesPerRow*8
        self.bytesPerRow = bytesPerRow
        self.rowsNum = size_Bytes // bytesPerRow
        self.columnsNumber = 0

        self.crossbarArray = [[] for x in range(self.rowsNum)]
        self.crossbarColumns = []

        self.verbose = False
        self.printHeader = ""

        ### ----- for Simulation Purposes--------- ###
        self.cycleCounter = 0
        self.frequency = 500 * 10**6
        self.cycles_per_reduction_pipe_stage = 2
        self.cycles_for_full_reduction = 30

        self.instructionsHistogram = {}
        self.cyclesPerInstructionsHistogram = {}
        self.histogramScope = ""

        self._tagged_rows_list = None
        #self.blosum_matrix = blosum62.blosum62()

    ### ------------------------------------------------------------ ###
    ### ------ Cycle Counter ------- ###

    def resetCycleCouter(self):
        self.cycleCounter = 0

    def advanceCycleCouter(self, cycles_executed):
        self.cycleCounter += cycles_executed

    def getCyclesCounter(self):
        return self.cycleCounter

    def setFrequency(self, _freq):
        self.frequency = _freq

    def getFrequency(self):
        return self.frequency

    ### ------------------------------------------------------------ ###
    def set_histogram_scope(self, scope):
        self.histogramScope = scope

    def remove_histogram_scope(self):
        self.histogramScope = ""

    def addOperationToInstructionsHistogram(self, instruction_name, bits=32, operations_to_add=1):
        add_to_histogram_flag = True
        full_instruction_name = instruction_name if self.histogramScope == "" else self.histogramScope + '.' + instruction_name
        self.addOrSetInHistogram(self.instructionsHistogram, add_to_histogram_flag, full_instruction_name, bits, operations_to_add)

    def addCyclesPerInstructionToHistogram(self, instruction_name, bits=32, cycles_per_instruction=1):
        add_to_histogram_flag = True #set value in histogram
        full_instruction_name = instruction_name if self.histogramScope == "" else self.histogramScope + '.' + instruction_name
        self.addOrSetInHistogram(self.cyclesPerInstructionsHistogram, add_to_histogram_flag, full_instruction_name, bits, cycles_per_instruction)

    def addOrSetInHistogram(self, histogram, add_to_histogram_flag, instruction_name, bits, value):
        if instruction_name not in histogram:
            histogram[instruction_name] = {}

        if bits not in histogram[instruction_name]:
            histogram[instruction_name][bits] = value
        elif add_to_histogram_flag:
            histogram[instruction_name][bits] += value

    def get_histogram_as_string(self):
        histogram_string = ""
        for instruction, count in self.instructionsHistogram.items():
            histogram_string += self.histogramScope + "." + str(instruction) + ": " + str(count)

        return histogram_string

    ### ------------------------------------------------------------ ###
    # Set the width of each column
    def tagRowsEqualToConstant(self, col_index, const, start_row, end_row):
        tagged_rows_list = []
        for row_index in range(start_row, end_row+1):
            if self.crossbarArray[row_index][col_index] == const:
                tagged_rows_list.append(row_index)

        cycles_executed = self.crossbarColumns[col_index]
        self.advanceCycleCouter(cycles_executed)
        self._tagged_rows_list = tagged_rows_list
        return tagged_rows_list

    def tagRows(self, col_index):
        cycles_executed = 3
        self.advanceCycleCouter(cycles_executed)

    ### ------------------------------------------------------------ ###
    def loadData(self, column_data, start_row, column_width, column_index=-1, count_as_operation = True):
        if column_index == -1 or column_index+1 > self.columnsNumber:
            self.crossbarColumns.append(column_width)
            for i in range(0, self.rowsNum):
                self.crossbarArray[i].append(None)

            for curr_row in range(start_row, min(self.rowsNum, start_row+len(column_data))):
                self.crossbarArray[curr_row][self.columnsNumber] = column_data[curr_row - start_row]

            self.columnsNumber += 1
        else:
            #self.crossbarColumns.append(column_width)
            for curr_row in range(start_row, start_row + min(self.rowsNum, len(column_data))):
                self.crossbarArray[curr_row][column_index] = column_data[curr_row - start_row]

        if self.verbose:
            operation_to_print = "load data in column " + str(column_index)
            self.printArray(operation=operation_to_print)

        # cycle count
        if count_as_operation:
            elements_num = min(self.rowsNum, len(column_data))
            self.addOperationToInstructionsHistogram(load_data_element_hist_name, self.crossbarColumns[column_index], elements_num)
            cycles_executed = 3*elements_num   # 3 cycles per loaded element (Read, match, write)
            self.addCyclesPerInstructionToHistogram(load_data_element_hist_name, self.crossbarColumns[column_index], cycles_executed)

    #####################################################################
    #####   Shift specific column values several rows up or down
    #####################################################################
    def shiftColumnOnTaggedRows(self, col_index, tagged_rows_list, distance_to_shift=1):
        # directions_of_shift determines whether to shift up or down

        for i in tagged_rows_list:
            self.crossbarArray[i+distance_to_shift][col_index] = self.crossbarArray[i][col_index]

        if self.verbose:
            operation_to_print = "shift column " + str(col_index) + " from row "
            if distance_to_shift > 0:
                operation_to_print = operation_to_print + tagged_rows_list[-1] + " to row " + tagged_rows_list[0]
            else:
                operation_to_print = operation_to_print + tagged_rows_list[0] + " to row " + tagged_rows_list[-1]

            self.printArray(operation=operation_to_print)
        ## DEBUG
        #print("Shifted %d rows in shiftColumnOnTaggedRows operation" % distance_to_shift)

        # cycle count - several rows are piping the TAGs
        self.addOperationToInstructionsHistogram(shift_operation_hist_name, bits=self.crossbarColumns[col_index])
        self.addOperationToInstructionsHistogram(shifted_rows_num_hist_name, bits=self.crossbarColumns[col_index],
                                                 operations_to_add=abs(distance_to_shift)-1)
        cycles_executed = 3 * self.crossbarColumns[col_index] # 3 cycles per shifted bit
        self.addCyclesPerInstructionToHistogram(shift_operation_hist_name, self.crossbarColumns[col_index], cycles_executed)
        # Every additional row to shift, beyond the first, requires 'bit-length' additional cycles
        self.addCyclesPerInstructionToHistogram(shifted_rows_num_hist_name, self.crossbarColumns[col_index],
                                                (abs(distance_to_shift)-1)*self.crossbarColumns[col_index])


    def shiftColumn(self, col_index, start_row, end_row, distance_to_shift=1):
        # Decide whether to shift up or down
        if distance_to_shift > 0:  # Shift down
            shift_range = range(end_row, start_row - 1, -1)
        else:  # Shift up
            shift_range = range(start_row, end_row)

        tagged_rows_list = list(shift_range)
        self.shiftColumnOnTaggedRows(col_index, tagged_rows_list, distance_to_shift)


    #####################################################################
    ######      Broadcast a single element to multiple ReCAM rows
    #####################################################################
    def broadcastDataElement(self, data_col_index, data_row_index,
                      destination_start_row, destination_col_index, destination_delta, destination_rows_per_element):
        data_to_broadcast = self.crossbarArray[data_row_index][data_col_index]

        for i in range(destination_rows_per_element):
            target_row = destination_start_row + i*destination_delta
            self.crossbarArray[target_row][destination_col_index] = data_to_broadcast

        if self.verbose:
            self.printArray(msg="broadcastDataElement")

        # cycle count
        self.addOperationToInstructionsHistogram(broadcast, self.crossbarColumns[data_col_index])
        cycles_executed = 4
        self.addCyclesPerInstructionToHistogram(broadcast, self.crossbarColumns[data_col_index], cycles_executed)

        #self.addOperationToInstructionsHistogram(broadcast, self.crossbarColumns[data_col_index])
        #cycles_executed = 1 + self.crossbarColumns[data_col_index]
        #self.addCyclesPerInstructionToHistogram(broadcast, self.crossbarColumns[data_col_index], cycles_executed)

    #####################################################################
    ######      Broadcast a single element to multiple ReCAM rows
    #####################################################################
    def broadcastData(self, data_col_index, data_start_row_index, data_length,
                      destination_start_row, destination_row_hops, destination_col_index, destination_delta,
                      destination_rows_per_element):

        destination_row = destination_start_row
        for data_row in range(data_start_row_index, data_start_row_index + data_length):
            self.broadcastDataElement(data_col_index, data_row, destination_row,
                                         destination_col_index, destination_delta, destination_rows_per_element)
            destination_row += destination_row_hops

            # cycle count is set by broadcastDataElement()


    #####################################################################
    ######      Simple arithmetic - Add, Subtract, Max
    #####################################################################
    def taggedRowWiseOperation(self, colA, colB, res_col, tagged_rows_list, operation, number_format=None):
        if operation == '+':
            for i in tagged_rows_list:
                self.crossbarArray[i][res_col] = convert_if_needed(self.crossbarArray[i][colA] + self.crossbarArray[i][colB], number_format)

        elif operation == '-':
            for i in tagged_rows_list:
                self.crossbarArray[i][res_col] = convert_if_needed(self.crossbarArray[i][colA] - self.crossbarArray[i][colB], number_format)

        elif operation == '*':
            for i in tagged_rows_list:
                self.crossbarArray[i][res_col] = convert_if_needed(self.crossbarArray[i][colA] * self.crossbarArray[i][colB], number_format)

        elif operation == copy_operation_string:
            for i in tagged_rows_list:
                self.crossbarArray[i][res_col] = self.crossbarArray[i][colA]

        elif operation == max_operation_string:
            for i in tagged_rows_list:
                self.crossbarArray[i][res_col] = convert_if_needed(max(self.crossbarArray[i][colA], self.crossbarArray[i][colB]), number_format)
        else:
            print("!!! Unknown Operation !!!")
            return

        if self.verbose:
            operation_to_print = "taggedRowWiseOperation() with operation = " + operation
            self.printArray(operation = operation_to_print)

        if operation == '-' or operation == '+':
            if res_col != colA:
                cycles_per_bit = 2 ** 3
            else:
                cycles_per_bit = 2 ** 2
        elif operation == copy_operation_string:
            cycles_per_bit = 0
        elif operation == max_operation_string:
            cycles_per_bit = 2
        elif operation == '*':
            cycles_per_bit = min(self.crossbarColumns[colA], self.crossbarColumns[colB]) * 16

        instruction_full_name = row_by_row_hist_name + '.' + operation
        if not colB:
            instruction_bits = self.crossbarColumns[colA]
        else:
            instruction_bits = max(self.crossbarColumns[colA], self.crossbarColumns[colB])
        self.addOperationToInstructionsHistogram(instruction_full_name, instruction_bits)
        if operation != copy_operation_string:
            cycles_executed = cycles_per_bit * instruction_bits
        else:
            cycles_executed = 2
        self.addCyclesPerInstructionToHistogram(instruction_full_name, instruction_bits, cycles_executed)

    ### ----------------------------

    def rowWiseOperation(self, colA, colB, res_col, start_row, end_row, operation, number_format=None):
        tagged_rows_list = list(range(start_row,end_row+1))
        self.taggedRowWiseOperation(colA, colB, res_col, tagged_rows_list, operation, number_format)


    ### ------------------------------------------------------------ ###
    # Vector-scalar arithmetic  - Add / Subtract / MUL / Max
    def taggedRowWiseOperationWithConstant(self, colA, const_scalar, res_col, tagged_rows_list, operation, number_format=None):
        if const_scalar == 0:
            converted_scalar = convert_if_needed(const_scalar, number_format)
        else:
            converted_scalar = convert_to_non_zero_if_needed(const_scalar, number_format)

        if operation == '+':
            for row_index in tagged_rows_list:
                self.crossbarArray[row_index][res_col] = convert_if_needed(self.crossbarArray[row_index][colA] + converted_scalar, number_format)
        elif operation == '-':
            for row_index in tagged_rows_list:
                self.crossbarArray[row_index][res_col] = convert_if_needed(self.crossbarArray[row_index][colA] - converted_scalar, number_format)
        elif operation == '*':
            for row_index in tagged_rows_list:
                self.crossbarArray[row_index][res_col] = convert_if_needed(self.crossbarArray[row_index][colA] * converted_scalar, number_format)
        elif operation == max_operation_string:
            for row_index in tagged_rows_list:
                self.crossbarArray[row_index][res_col] = convert_if_needed(max(self.crossbarArray[row_index][colA], converted_scalar), number_format)
        elif operation == write_operation_string:
            for row_index in tagged_rows_list:
                self.crossbarArray[row_index][res_col] = converted_scalar
        else:
            print("!!! Unknown Operation in taggedRowWiseOperationWithConstant: " + operation + " !!!")
            exit()

        if self.verbose:
            self.printArray(msg="rowWiseOperationWithConstant")

        if operation == max_operation_string:
            cycles_per_bit = 2
        elif operation == '*':
            cycles_per_bit = self.crossbarColumns[colA] * 2
        else:
            cycles_per_bit = 2 ** 3

        instruction_full_name = row_by_const_hist_name + '.' + operation
        instruction_bits = self.crossbarColumns[colA] if not write_operation_string else self.crossbarColumns[res_col]
        self.addOperationToInstructionsHistogram(instruction_full_name, instruction_bits)
        cycles_executed = cycles_per_bit * instruction_bits
        self.addCyclesPerInstructionToHistogram(instruction_full_name, instruction_bits, cycles_executed)

    ### ------------------------------------------------------------ ###
    # Vector-scalar arithmetic  - Add / Subtract / MUL / Max
    def rowWiseOperationWithConstant(self, colA, const_scalar, res_col, start_row, end_row, operation, number_format=None):
        tagged_rows_list = list(range(start_row, end_row + 1))
        self.rowWiseOperationWithConstant(colA, const_scalar, res_col, tagged_rows_list, operation, number_format)

    ### ------------------------------------------------------------ ###
    # Fixed-point multiplication
    '''def MULConsecutiveRows(self, start_row, end_row, colRes, colA, colB, number_format=None):
        for i in range(start_row, end_row+1):
            self.crossbarArray[i][colRes] = convert_if_needed(self.crossbarArray[i][colA] * self.crossbarArray[i][colB], number_format)

        if self.verbose:
            self.printArray(msg="MULConsecutiveRows")

        # cycle count
        cycles_executed = (max(self.crossbarColumns[colA], self.crossbarColumns[colA]))**2
        self.advanceCycleCouter(cycles_executed)
    '''
    ### ------------------------------------------------------------ ###
    # Fixed-point multiplication
    '''def MULTaggedRows(self, tagged_rows_list, colRes, colA, colB, number_format=None):
        for row_num in tagged_rows_list:
            self.crossbarArray[row_num][colRes] = convert_if_needed(self.crossbarArray[row_num][colA] * self.crossbarArray[row_num][colB], number_format)

        if self.verbose:
            self.printArray()

        # cycle count
        cycles_executed = (max(self.crossbarColumns[colA], self.crossbarColumns[colA]))**2
        self.advanceCycleCouter(cycles_executed)
    '''
    ### ------------------------------------------------------------ ###
    # Simple variable-constant arithmetic  - Add / Subtract
    def getScalarFromColumnOnTaggedRows(self, col_index, tagged_rows_list, operation, numbers_format=None):
        result = 0
        result_row_index = -1
        if operation == '+':
            for i in tagged_rows_list:
                result = convert_if_needed(result + self.crossbarArray[i][col_index], numbers_format)
        elif operation == max_operation_string:
            for i in tagged_rows_list:
                if self.crossbarArray[i][col_index] > result:
                    result = self.crossbarArray[i][col_index]
                    result_row_index = i
        else:
            print("!!! Unknown Operation !!!")

        if self.verbose:
            self.printArray()

        if operation == max_operation_string:
            cycles_per_bit = 2
            cycles_executed = cycles_per_bit * self.crossbarColumns[col_index]
        else:
            cycles_per_bit = 2 ** 2
            cycles_executed = cycles_per_bit * self.crossbarColumns[col_index] * math.ceil(
                math.log(len(self.crossbarColumns[col_index])))

        full_instruction_name = reduce_scalar_from_column_hist_name + '.' + operation
        self.addOperationToInstructionsHistogram(full_instruction_name, self.crossbarColumns[col_index])
        self.addCyclesPerInstructionToHistogram(full_instruction_name, self.crossbarColumns[col_index], cycles_executed)
        return result, result_row_index


    def getScalarFromColumn(self, col_index, start_row, end_row, operation, numbers_format=None):
        tagged_rows_list = list(range(start_row, end_row+1))
        return self.getScalarFromColumnOnTaggedRows(col_index, tagged_rows_list, operation, numbers_format)


    ### ------------------------------------------------------------ ###
    # Pipelined Reduction
    def pipelinedReduction(self, rows_to_sum_range, input_col, output_row, output_col, is_first_accumulation, numbers_format):
        reduction_sum = 0
        for row_index in rows_to_sum_range:
            reduction_sum = convert_if_needed(reduction_sum + self.crossbarArray[row_index][input_col], numbers_format)

        self.crossbarArray[output_row][output_col] = reduction_sum

        if is_reduction_tree_pipelined: #-------- Pipelined reduction tree: output-per-cycle --------
            if is_first_accumulation:
                full_instruction_name = reduction_tree_sum + '.' + "first"
            else:
                full_instruction_name = reduction_tree_sum

            self.addOperationToInstructionsHistogram(full_instruction_name, self.crossbarColumns[input_col])
            cycles_to_count = self.cycles_for_full_reduction if is_first_accumulation else self.cycles_per_reduction_pipe_stage
            self.addCyclesPerInstructionToHistogram(full_instruction_name, self.crossbarColumns[input_col], cycles_to_count)

        else: #-------- Non-pipelined reduction tree: logN cycles per output --------
            full_instruction_name = reduction_tree_sum

            self.addOperationToInstructionsHistogram(full_instruction_name, self.crossbarColumns[input_col])
            cycles_to_count = self.cycles_for_full_reduction
            self.addCyclesPerInstructionToHistogram(full_instruction_name, self.crossbarColumns[input_col], cycles_to_count)

        return reduction_sum

    ### ------------------------------------------------------------ ###
    ###         Calculate softmax on CPU                             ###
    ### ------------------------------------------------------------ ###
    def calculateSoftmaxOnCPU(self, sums_vector_from_ReCAM, numbers_format):
        # 1. Send vector to CPU
        # 2. Perform calclation and place result in result vector
        for i in range(len(sums_vector_from_ReCAM)):
            sums_vector_from_ReCAM[i] = math.exp(sums_vector_from_ReCAM[i])

        sum_of_exponents = 0
        for i in range(len(sums_vector_from_ReCAM)):
            sum_of_exponents += sums_vector_from_ReCAM[i]

        for i in range(len(sums_vector_from_ReCAM)):
            sums_vector_from_ReCAM[i] = convert_if_needed(sums_vector_from_ReCAM[i] / sum_of_exponents, numbers_format)

        self.addOperationToInstructionsHistogram("CPU.softmax", numbers_format.total_bits, len(sums_vector_from_ReCAM))
        self.addCyclesPerInstructionToHistogram("CPU.softmax", numbers_format.total_bits, CPU_softmax_cycles)

    ### ------------------------------------------------------------ ###
    def setVerbose(self, _verbose):
        self.verbose = _verbose

    def setPrintHeader(self, header=""):
        self.printHeader = header

    # Print array contents
    def printArray(self, start_row=0, end_row=-1, start_col=0, end_col=-1, header="", tablefmt="grid", operation=None, msg=None):
        if end_row == -1: end_row=self.rowsNum
        if end_col == -1: end_col = self.rowsNum

        # for row in range(start_row, end_row):
        #     print(self.crossbarArray[row])
        if operation:
            print("%%%  Performed ", operation)
        if msg:
            print("%%%  ", msg)

        if header == "":
            print(tabulate(self.crossbarArray, self.printHeader, tablefmt, stralign="center")) #other format option is "grid"
        else:
            print(tabulate(self.crossbarArray, header, tablefmt, stralign="center"))  # other format option is "grid"

        print("\n")

    # Print histogram contents
    def printHistogram(self):
        print(self.instructionsHistogram)

    def get_col_letter(self, col_num):
        return chr(ord('A')+col_num)

    def get_bold_format(self, workbook, font_size):
        headline_format = workbook.add_format()
        headline_format.set_bold()
        headline_format.set_font_size(font_size)
        headline_format.set_align('center')
        headline_format.set_align('vcenter')
        return headline_format

    def get_blue_bold_format(self, workbook, font_size):
        headline_format = workbook.add_format()
        headline_format.set_bold()
        headline_format.set_font_size(font_size)
        headline_format.set_align('center')
        headline_format.set_align('vcenter')
        headline_format.set_color('#5364fc')
        return headline_format

    def write_col_name_and_fig(self, worksheet, row_num, name_col_num, name_col_string, fig_col_value, custom_format=None):
        if custom_format:
            worksheet.write(row_num, name_col_num, name_col_string, custom_format)
            worksheet.write(row_num, name_col_num + 1, fig_col_value, custom_format)
        else:
            worksheet.write(row_num, name_col_num, name_col_string)
            worksheet.write(row_num, name_col_num + 1, fig_col_value)

    def printHistogramsToExcel(self, nn, total_samples, net_name="", epoch_num=""):
        curr_time = datetime.datetime.now().isoformat().replace(':', '.')
        workbook = xlsxwriter.Workbook('C:\\Dev\\MNIST\\test' + curr_time + '.xlsx')
        worksheet = workbook.add_worksheet()

        #set the headline format
        headline_format = self.get_bold_format(workbook, 12)
        blue_format = self.get_blue_bold_format(workbook, 12)

        worksheet.set_column(0, 0, 20)
        worksheet.set_column(5, 5, 25)

        col_num = 0
        worksheet.write(0, col_num, "Operation Type", headline_format)
        col_num += 1
        worksheet.write(0, col_num, "bits", headline_format)
        col_num += 1
        worksheet.write(0, col_num, "calls", headline_format)
        col_num += 1
        worksheet.write(0, col_num, "cycles", headline_format)
        cycles_col_char = self.get_col_letter(col_num)
        col_num += 1

        i=1
        for operation_name, bits_and_calls in self.instructionsHistogram.items():
            for bits, calls in bits_and_calls.items():
                worksheet.write(i, 0, operation_name)
                worksheet.write(i, 1, bits)
                worksheet.write(i, 2, calls)
                worksheet.write(i, 3, self.cyclesPerInstructionsHistogram[operation_name][bits])
                '''operation_without_scope = ""
                if "." in operation_name:
                    operation_without_scope = operation_name[operation_name.index(".") + 1:]
                if (operation_without_scope != "") and (operation_without_scope in self.cyclesPerInstructionsHistogram):
                    worksheet.write(i, 3, self.cyclesPerInstructionsHistogram[operation_without_scope][bits])
                else:
                    worksheet.write(i, 3, self.cyclesPerInstructionsHistogram[operation_name][bits])
                '''
                i+=1
        num_of_unique_operations = i

        i+=1    # Add an empty line
        #########  Globals #########
        worksheet.merge_range(i, 0, i, 1, "Global Parameters", headline_format)
        i += 1
        worksheet.write(i, 0, "ReCAM Frequency")
        worksheet.write(i, 1, self.frequency)
        ReCAM_freq_row_num = i
        i += 1
        worksheet.write(i, 0, "Training Set Samples")
        worksheet.write(i, 1, total_samples)
        total_samples_row_num = i

        #########  Results #########
        results_row = 0
        results_name_col_num = 5
        results_name_col_letter = self.get_col_letter(results_name_col_num)
        results_figure_col_letter = self.get_col_letter(results_name_col_num+1)
        # single sample iteration
        worksheet.merge_range(results_row, results_name_col_num, results_row, results_name_col_num+1, "Final Results", headline_format)
        results_row += 1

        # Cycles per sample
        sum_string = cycles_col_char + str(1) + ":" + cycles_col_char + str(num_of_unique_operations)
        self.write_col_name_and_fig(worksheet, results_row, results_name_col_num, 'Cycles per 1 input sample', '=SUM(' + sum_string + ')')
        cycles_per_sample_row_num = results_row
        results_row += 1

        # Time per sample
        time_per_sample_formula = '=' + results_figure_col_letter + str(results_row) + '/' + 'B' + str(ReCAM_freq_row_num + 1)
        self.write_col_name_and_fig(worksheet, results_row, results_name_col_num, 'Time per 1 sample (sec)', time_per_sample_formula, headline_format)
        results_row += 1

        # single epoch
        self.write_col_name_and_fig(worksheet, results_row, results_name_col_num, 'Cycles per 1 epoch','=' + results_figure_col_letter + str(cycles_per_sample_row_num+1) + '*' + 'B' + str(total_samples_row_num+1))
        results_row += 1

        # Time per epoch
        time_per_epoch_formula = '=' + results_figure_col_letter + str(results_row) + '/' + 'B' + str(ReCAM_freq_row_num+ 1)
        self.write_col_name_and_fig(worksheet, results_row, results_name_col_num, 'Time per 1 epoch (sec)', time_per_epoch_formula, blue_format)
        headline_format.set_color('black')

        results_row += 2

        #########  Net Parameters #########
        net_row = results_row
        headline_format.set_top(1)
        worksheet.merge_range(net_row, results_name_col_num, net_row, results_name_col_num + 1, "Net Parameters",
                              headline_format)
        net_row += 1
        headline_format.set_top(0)

        total_net_operations=0
        for layer_index in range(1, len(nn.layers)):
            neurons_in_layer = len(nn.weightsMatrices[layer_index])
            weights_in_layer = len(nn.weightsMatrices[layer_index][0])

            patemeters_in_layer = neurons_in_layer * weights_in_layer
            total_net_operations += neurons_in_layer * (2*weights_in_layer-1)
            if layer_index < len(nn.layers)-1:
                layer_name = "Layer " + str(layer_index)
            else:
                layer_name = "Output Layer"
            self.write_col_name_and_fig(worksheet, net_row, results_name_col_num, layer_name, patemeters_in_layer)
            net_row +=1

        total_net_parameters_formula = '=SUM(' + results_figure_col_letter + str(results_row+2) + ":" + results_figure_col_letter  + str(results_row+len(nn.layers)) + ")"
        self.write_col_name_and_fig(worksheet, net_row, results_name_col_num, "Total Net Parameters", total_net_parameters_formula, blue_format)

        net_row += 1
        #total_net_operations_formula = '=SUM(' + results_figure_col_letter + str(results_row + 2) + ":" + results_figure_col_letter + str(results_row + len(nn.layers)) + ")"
        #self.write_col_name_and_fig(worksheet, net_row, results_name_col_num, "Total training Operations", total_net_operations, blue_format)

        '''net_row += 1
        ops_formula = '=' + results_figure_col_letter + str(total_net_operations) + "/"
        # total_net_operations_formula = '=SUM(' + results_figure_col_letter + str(results_row + 2) + ":" + results_figure_col_letter + str(results_row + len(nn.layers)) + ")"
        self.write_col_name_and_fig(worksheet, net_row, results_name_col_num, "ReCAM Op/s",total_net_operations, blue_format)
        '''

    ### ------------------------------------------------------------ ###
    # Calculate match score
    def set_match_matrix(self, seq_type, protein_matrix=None, DNA_match_score=0, DNA_mismatch_score=0):
        self.seq_type = seq_type
        if seq_type == 'protein':
            self.protein_matrix = protein_matrix
        else:
            self.DNA_match_score = DNA_match_score
            self.DNA_mismatch_score = DNA_mismatch_score

    def DNAbpMatch(self, colA, colB, res_col, start_row, end_row, bp_match_score, bp_mismatch_score):
        for curr_row in range(start_row, end_row+1):
            is_bp_match = (self.crossbarArray[curr_row][colA] == self.crossbarArray[curr_row][colB])
            self.crossbarArray[curr_row][res_col] = bp_match_score if is_bp_match else bp_mismatch_score

        cycles_executed = 2 + 4*(max(self.crossbarColumns[colA], self.crossbarColumns[colA]))
        self.advanceCycleCouter(cycles_executed)
        self.addOperationToInstructionsHistogram("DNA base-pair match")


    def get_match_score(self, a, b):
        if self.seq_type == 'protein':
            return self.protein_matrix.match_dict[(a,b)]
        else: #seq type is DNA
            return self.DNA_match_score if a==b else self.DNA_mismatch_score

    def get_cycles_executed(self):
        if self.seq_type == 'protein':
            return 2*self.protein_matrix.match_table_rows - self.protein_matrix.batched_write_saved_cycles
        else: #DNA
            return 10

    def SeqMatchOnTaggedRows(self, colA, colB, res_col, tagged_rows_list):
        for curr_row in tagged_rows_list:
            match_score = self.get_match_score(self.crossbarArray[curr_row][colA], self.crossbarArray[curr_row][colB])
            self.crossbarArray[curr_row][res_col] = match_score

        cycles_executed = self.get_cycles_executed()
        self.advanceCycleCouter(cycles_executed)
        self.addOperationToInstructionsHistogram("DNA base-pair match")

    '''
    You have only 4 equal combinations (0/0, 1/1, 2/2 and 3/3). The rest are mismatch.
    So what u do is you initialize the result bitcolumn with 0 (that’s TAGSET + write – 2 cycles), and then 4x2 (compare + write) cycles- total of 10 cycles.
    (TAGSET is a command that sets all tags…. Can be implemented by triggering the “set” input of TAG latch).
    '''

    ### ------------------------------------------------------------ ###

'''
def test():
    for i in range (2):
        tmp = Simulator.ReCAM(100+i)
        print "size in bytes = ",tmp.sizeInBytes
        print "bits per row = ", tmp.bitsPerRow




test()
'''