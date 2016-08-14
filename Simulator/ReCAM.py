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
            operation_to_print = "load data in column " + column_index
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
            operation_to_print = "shift column " + col_index + " from row "
            if numOfRowsToShift > 0:
                operation_to_print = operation_to_print + end_row + " to row " + start_row
            else:
                operation_to_print = operation_to_print + start_row + " to row " + end_row

            self.printArray(operation=operation_to_print)

        # cycle count
        cycles_executed = 3 * self.crossbarColumns[col_index]
        self.advanceCycleCouter(cycles_executed)

    #####################################################################
    #####   Shift specific column values several rows up or down
    #####################################################################
    def shiftColumnOnTaggedRows(self, col_index, tagged_rows_list, direction_of_shift=1):
        # directions_of_shift determines whether to shift up or down

        for i in tagged_rows_list:
            self.crossbarArray[i+direction_of_shift][col_index] = self.crossbarArray[i][col_index]

        if self.verbose:
            operation_to_print = "shift tagged rows in column " + col_index + " direction: " + direction_of_shift
            self.printArray(operation=operation_to_print)

        # cycle count
        cycles_executed = 3 * self.crossbarColumns[col_index]
        self.advanceCycleCouter(cycles_executed)


    #####################################################################
    ######      Broadcast a single element to multiple ReCAM rows
    #####################################################################
    def broadcastDataElement(self, data_col_index, data_row_index,
                      destination_start_row, destination_col_index, destination_delta, total_destination_rows):
        data_to_broadcast = self.crossbarArray[data_row_index][data_col_index]

        for i in range(total_destination_rows):
            target_row = destination_start_row + i*destination_delta

            self.crossbarArray[target_row][destination_col_index] = data_to_broadcast

        if self.verbose:
            self.printArray()

        # cycle count
        cycles_executed = 1 + self.crossbarColumns[data_col_index]
        self.advanceCycleCouter(cycles_executed)


    #####################################################################
    ######      Simple arithmetic - Add, Subtract, Max
    #####################################################################
    def rowWiseOperation(self, colA, colB, res_col, start_row, end_row, operation):
        max_operation_string = "max"

        if operation == '+':
            for i in range(start_row,end_row+1):
                self.crossbarArray[i][res_col] = self.crossbarArray[i][colA] + self.crossbarArray[i][colB]
        elif operation == '-':
            for i in range(start_row, end_row+1):
                self.crossbarArray[i][res_col] = self.crossbarArray[i][colA] - self.crossbarArray[i][colB]
        elif operation == max_operation_string:
            for i in range(start_row, end_row + 1):
                self.crossbarArray[i][res_col] = max(self.crossbarArray[i][colA], self.crossbarArray[i][colB])
        else:
            print("!!! Unknown Operation !!!")

        if self.verbose:
            self.printArray()

        if operation == '-' or operation == '+':
            if res_col != colA:
                cycles_per_bit = 2**3
            else:
                cycles_per_bit = 2**2
        elif operation == max_operation_string:
            cycles_per_bit = 2

        cycles_executed = cycles_per_bit * max(self.crossbarColumns[colA],self.crossbarColumns[colB])
        self.advanceCycleCouter(cycles_executed)


    ### ------------------------------------------------------------ ###
    # Simple arithmetic - Add, Subtract, Max
    def taggedRowWiseOperation(self, colA, colB, res_col, tagged_rows_list, operation, number_format=None):
        max_operation_string = "max"

        if operation == '+':
            for i in tagged_rows_list:
                if number_format:
                    self.crossbarArray[i][res_col] = number_format.convert(self.crossbarArray[i][colA] + self.crossbarArray[i][colB])
                else:
                    self.crossbarArray[i][res_col] = self.crossbarArray[i][colA] + self.crossbarArray[i][colB]

        elif operation == '-':
            for i in tagged_rows_list:
                if number_format:
                    self.crossbarArray[i][res_col] = number_format.convert(self.crossbarArray[i][colA] - self.crossbarArray[i][colB])
                else:
                    self.crossbarArray[i][res_col] = self.crossbarArray[i][colA] - self.crossbarArray[i][colB]

        elif operation == max_operation_string:
            for i in tagged_rows_list:
                self.crossbarArray[i][res_col] = max(self.crossbarArray[i][colA], self.crossbarArray[i][colB])
        else:
            print("!!! Unknown Operation !!!")

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

        cycles_executed = cycles_per_bit * max(self.crossbarColumns[colA], self.crossbarColumns[colB])
        self.advanceCycleCouter(cycles_executed)

    ### ------------------------------------------------------------ ###
    # Simple variable-constant arithmetic  - Add / Subtract
    def rowWiseOperationWithConstant(self, colA, const_scalar, res_col, start_row, end_row, operation):
        max_operation_string = "max"

        if operation == '+':
            for i in range(start_row, end_row+1):
                self.crossbarArray[i][res_col] = self.crossbarArray[i][colA] + const_scalar
        elif operation == '-':
            for i in range(start_row, end_row+1):
                self.crossbarArray[i][res_col] = self.crossbarArray[i][colA] - const_scalar
        elif operation == max_operation_string:
            for i in range(start_row, end_row + 1):
                self.crossbarArray[i][res_col] = max(self.crossbarArray[i][colA], const_scalar)
        else:
            print("!!! Unknown Operation !!!")

        if self.verbose:
            self.printArray()

        if operation == max_operation_string:
            cycles_per_bit = 2
        else:
            cycles_per_bit = 2 ** 2

        cycles_executed = cycles_per_bit * self.crossbarColumns[colA]
        self.advanceCycleCouter(cycles_executed)

    ### ------------------------------------------------------------ ###
    # Fixed-point multiplication
    def MULConsecutiveRows(self, start_row, end_row, colRes, colA, colB, numbersFormat=None):
        for i in range(start_row, end_row+1):
            if not numbersFormat:
                self.crossbarArray[i][colRes] = self.crossbarArray[i][colA] * self.crossbarArray[i][colB]
            else:
                self.crossbarArray[i][colRes] = numbersFormat.convert(self.crossbarArray[i][colA] *
                                                                     self.crossbarArray[i][colB])

        if self.verbose:
            self.printArray()

        # cycle count
        cycles_executed = (max(self.crossbarColumns[colA], self.crossbarColumns[colA]))**2
        self.advanceCycleCouter(cycles_executed)

    ### ------------------------------------------------------------ ###
    # Fixed-point multiplication
    def MULTaggedRows(self, tagged_rows_list, colRes, colA, colB, numbersFormat=None):
        for row_num in tagged_rows_list:
            if not numbersFormat:
                self.crossbarArray[row_num][colRes] = self.crossbarArray[row_num][colA] * self.crossbarArray[row_num][colB]
            else:
                self.crossbarArray[row_num][colRes] = numbersFormat.convert(self.crossbarArray[row_num][colA] *
                                                                            self.crossbarArray[row_num][colB])

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
    def printArray(self, start_row=0, end_row=-1, start_col=0, end_col=-1, header="", tablefmt="grid", operation=None):
        if end_row == -1: end_row=self.rowsNum
        if end_col == -1: end_col = self.rowsNum

        # for row in range(start_row, end_row):
        #     print(self.crossbarArray[row])
        if operation:
            print("%%%  Performed ", operation)

        if header == "":
            print(tabulate(self.crossbarArray, self.printHeader, tablefmt, stralign="center")) #other format option is "grid"
        else:
            print(tabulate(self.crossbarArray, header, tablefmt, stralign="center"))  # other format option is "grid"

        print("\n")

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