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

lib_path = os.path.abspath(os.path.join('spfpm-1.1'))
sys.path.append(lib_path)
from tabulate import tabulate

################################################
####        AUX functions & definitions
################################################
max_operation_string = "max"

#--- Instructions Names ---#
row_by_row_hist_name = 'vector-vector'
row_by_const_hist_name = 'vector-constant'
shift_rows_hist_name = 'shift rows'
broadcast = 'broadcast'


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

        self.instructionsHistogram = {}
        self.histogramScope = ""
        self.initialize_instructions_histogram()

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
    # Set the width of each column
    def setColumns(self, column_widths):
        self.crossbarColumns = column_widths

    ### ------------------------------------------------------------ ###
    def set_histogram_scope(self, scope):
        self.histogramScope = scope


    def remove_histogram_scope(self, scope):
        self.histogramScope = ""


    def initialize_instructions_histogram(self):
        self.instructionsHistogram[row_by_row_hist_name + '.+'] = 0
        self.instructionsHistogram[row_by_row_hist_name + '.-'] = 0
        self.instructionsHistogram[row_by_row_hist_name + '.*'] = 0
        self.instructionsHistogram[row_by_row_hist_name + '.max'] = 0
        self.instructionsHistogram[row_by_const_hist_name + '.+'] = 0
        self.instructionsHistogram[row_by_const_hist_name + '.-'] = 0
        self.instructionsHistogram[row_by_const_hist_name + '.*'] = 0
        self.instructionsHistogram[row_by_const_hist_name + '.max'] = 0
        self.instructionsHistogram[shift_rows_hist_name] = 0
        self.instructionsHistogram[broadcast] = 0

    def get_histogram_as_string(self):
        histogram_string = ""
        for instruction, count in self.instructionsHistogram.items():
            histogram_string += self.histogramScope + "." + str(instruction) + ": " + str(count)

        return histogram_string

    ### ------------------------------------------------------------ ###
    # Set the width of each column
    def tagRows(self, col_index):
        cycles_executed = self.crossbarColumns[col_index]
        self.advanceCycleCouter(cycles_executed)

    ### ------------------------------------------------------------ ###
    def loadData(self, column_data, start_row, column_width, column_index=-1):
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

    ### ------------------------------------------------------------ ###
    # Shift specific column values several rows up or down
    def shiftColumn(self, col_index, start_row, end_row, numOfRowsToShift=1):
        # Decide whether to shift up or down
        if numOfRowsToShift > 0: #Shift down
            shift_range = range(end_row, start_row-1, -1)
        else:   #Shift up
            shift_range = range(start_row, end_row)

        for i in shift_range:
            self.crossbarArray[i+numOfRowsToShift][col_index] = self.crossbarArray[i][col_index]

        # Zero-fill empty rows
        # if numOfRowsToShift > 0:  # Shift down
        #     zero_fill_range = range(start_row, start_row + numOfRowsToShift)
        # else:  # Shift up
        #     zero_fill_range = range(start_row + numOfRowsToShift + 1, start_row)
        #
        # for j in zero_fill_range:
        #     self.crossbarArray[j][col] = 0

        if self.verbose:
            operation_to_print = "shift column " + str(col_index) + " from row "
            if numOfRowsToShift > 0:
                operation_to_print = operation_to_print + str(end_row) + " to row " + str(start_row)
            else:
                operation_to_print = operation_to_print + str(start_row) + " to row " + str(end_row)

            self.printArray(operation=operation_to_print)

        # cycle count
        cycles_executed = 3 * self.crossbarColumns[col_index] # 3 cycles per shifted bit
        self.instructionsHistogram[shift_rows_hist_name] += 1
        self.advanceCycleCouter(cycles_executed)

    #####################################################################
    #####   Shift specific column values several rows up or down
    #####################################################################
    def shiftColumnOnTaggedRows(self, col_index, tagged_rows_list, distance_to_shift=1):
        # directions_of_shift determines whether to shift up or down

        for i in tagged_rows_list:
            self.crossbarArray[i+distance_to_shift][col_index] = self.crossbarArray[i][col_index]

        if self.verbose:
            operation_to_print = "shift tagged rows in column " + str(col_index) + " direction: " + ("up" if distance_to_shift>1 else "down")
            self.printArray(operation=operation_to_print)

        # cycle count - several rows are piping the TAGs
        cycles_executed = (abs(distance_to_shift) + 2) * self.crossbarColumns[col_index]
        self.instructionsHistogram[shift_rows_hist_name] += 1
        self.advanceCycleCouter(cycles_executed)


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
        cycles_executed = 1 + self.crossbarColumns[data_col_index]
        self.instructionsHistogram[broadcast] += 1
        self.advanceCycleCouter(cycles_executed)

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
    def rowWiseOperation(self, colA, colB, res_col, start_row, end_row, operation, number_format=None):
        if operation == '+':
            for i in range(start_row,end_row+1):
                self.crossbarArray[i][res_col] = convert_if_needed(self.crossbarArray[i][colA] + self.crossbarArray[i][colB], number_format)

        elif operation == '-':
            for i in range(start_row, end_row+1):
                self.crossbarArray[i][res_col] = convert_if_needed(self.crossbarArray[i][colA] - self.crossbarArray[i][colB], number_format)

        elif operation == '*':
            for i in range(start_row, end_row + 1):
                self.crossbarArray[i][res_col] = convert_if_needed(self.crossbarArray[i][colA] * self.crossbarArray[i][colB], number_format)

        elif operation == max_operation_string:
            for i in range(start_row, end_row + 1):
                self.crossbarArray[i][res_col] = convert_if_needed(max(self.crossbarArray[i][colA], self.crossbarArray[i][colB]), number_format)
        else:
            print("!!! Unknown Operation !!!")
            return

        if self.verbose:
            operation_to_print = "rowWiseOperation() with operation = " + operation
            self.printArray(msg=operation_to_print)

        if operation == '-' or operation == '+':
            if res_col != colA:
                cycles_per_bit = 2**3
            else:
                cycles_per_bit = 2**2
        elif operation == max_operation_string:
            cycles_per_bit = 2
        elif operation == '*':
            cycles_per_bit = min(self.crossbarColumns[colA],self.crossbarColumns[colB]) * 2

        cycles_executed = cycles_per_bit * max(self.crossbarColumns[colA],self.crossbarColumns[colB])
        self.instructionsHistogram[row_by_row_hist_name + '.' + operation]
        self.advanceCycleCouter(cycles_executed)


    ### ------------------------------------------------------------ ###
    # Simple arithmetic - Add, Subtract, Max
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
        elif operation == max_operation_string:
            cycles_per_bit = 2
        elif operation == '*':
            cycles_per_bit = min(self.crossbarColumns[colA], self.crossbarColumns[colB]) * 2

        cycles_executed = cycles_per_bit * max(self.crossbarColumns[colA], self.crossbarColumns[colB])
        self.instructionsHistogram[row_by_row_hist_name + '.' + operation]
        self.advanceCycleCouter(cycles_executed)

    ### ------------------------------------------------------------ ###
    # Simple variable-constant arithmetic  - Add / Subtract
    def rowWiseOperationWithConstant(self, colA, const_scalar, res_col, start_row, end_row, operation, number_format=None):
        max_operation_string = "max"
        if const_scalar == 0:
            converted_scalar = convert_if_needed(const_scalar, number_format)
        else:
            converted_scalar = convert_to_non_zero_if_needed(const_scalar, number_format)

        if operation == '+':
            for i in range(start_row, end_row+1):
                self.crossbarArray[i][res_col] = convert_if_needed(self.crossbarArray[i][colA] + converted_scalar, number_format)
        elif operation == '-':
            for i in range(start_row, end_row+1):
                self.crossbarArray[i][res_col] = convert_if_needed(self.crossbarArray[i][colA] - converted_scalar, number_format)
        elif operation == '*':
            for i in range(start_row, end_row + 1):
                self.crossbarArray[i][res_col] = convert_if_needed(self.crossbarArray[i][colA] * converted_scalar, number_format)
        elif operation == max_operation_string:
            for i in range(start_row, end_row + 1):
                self.crossbarArray[i][res_col] = convert_if_needed(max(self.crossbarArray[i][colA], converted_scalar), number_format)
        else:
            print("!!! Unknown Operation in rowWiseOperationWithConstant: " + operation + " !!!")
            exit()

        if self.verbose:
            self.printArray(msg="rowWiseOperationWithConstant")

        if operation == max_operation_string:
            cycles_per_bit = 2
        elif operation == '*':
            cycles_per_bit = self.crossbarColumns[colA] * 2
        else:
            cycles_per_bit = 2 ** 2

        cycles_executed = cycles_per_bit * self.crossbarColumns[colA]
        self.instructionsHistogram[row_by_const_hist_name + '.' + operation]
        self.advanceCycleCouter(cycles_executed)

    ### ------------------------------------------------------------ ###
    # Fixed-point multiplication
    def MULConsecutiveRows(self, start_row, end_row, colRes, colA, colB, number_format=None):
        for i in range(start_row, end_row+1):
            self.crossbarArray[i][colRes] = convert_if_needed(self.crossbarArray[i][colA] * self.crossbarArray[i][colB], number_format)

        if self.verbose:
            self.printArray(msg="MULConsecutiveRows")

        # cycle count
        cycles_executed = (max(self.crossbarColumns[colA], self.crossbarColumns[colA]))**2
        self.advanceCycleCouter(cycles_executed)

    ### ------------------------------------------------------------ ###
    # Fixed-point multiplication
    def MULTaggedRows(self, tagged_rows_list, colRes, colA, colB, number_format=None):
        for row_num in tagged_rows_list:
            self.crossbarArray[row_num][colRes] = convert_if_needed(self.crossbarArray[row_num][colA] * self.crossbarArray[row_num][colB], number_format)

        if self.verbose:
            self.printArray()

        # cycle count
        cycles_executed = (max(self.crossbarColumns[colA], self.crossbarColumns[colA]))**2
        self.advanceCycleCouter(cycles_executed)

    ### ------------------------------------------------------------ ###
    # Simple variable-constant arithmetic  - Add / Subtract
    def getScalarFromColumn(self, col_index, start_row, end_row, operation):
        max_operation_string = "max"

        result = 0
        result_row_index = -1
        if operation == '+':
            for i in range(start_row, end_row+1):
                result += self.crossbarArray[i][col_index]
        elif operation == max_operation_string:
            for i in range(start_row, end_row + 1):
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
            cycles_executed = cycles_per_bit * self.crossbarColumns[col_index] * math.ceil( math.log( len(self.crossbarColumns[col_index])))

        self.advanceCycleCouter(cycles_executed)
        return cycles_executed, result, result_row_index

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

    def printHistogram(self):
        print(self.instructionsHistogram)

    ### ------------------------------------------------------------ ###
    # Calculate match score
    def DNAbpMatch(self, colA, colB, res_col, start_row, end_row, bp_match_score, bp_mismatch_score):
        for curr_row in range(start_row, end_row+1):
            is_bp_match = (self.crossbarArray[curr_row][colA] == self.crossbarArray[curr_row][colB])
            self.crossbarArray[curr_row][res_col] = bp_match_score if is_bp_match else bp_mismatch_score

        cycles_executed = 2 + 4*(max(self.crossbarColumns[colA], self.crossbarColumns[colA]))
        self.advanceCycleCouter(cycles_executed)

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