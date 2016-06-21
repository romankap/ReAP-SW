__author__ = 'Roman'

try:
    import matplotlib, numpy
    matplotlib.use('qt4agg')
    import matplotlib.pyplot as plt
    HAVE_MATPLOTLIB = True
except ImportError:
    HAVE_MATPLOTLIB = False

import os,sys
import math
lib_path = os.path.abspath(os.path.join('spfpm-1.1'))
sys.path.append(lib_path)
import FixedPoint
from tabulate import tabulate

#TODO: Add verbose mode - allow by setting a flag to print array contents after each operation
class ReCAM:
    def __init__(self, size_Bytes, bytesPerRow=32):
        self.sizeInBytes = size_Bytes
        self.bitsPerRow = bytesPerRow*8
        self.bytesPerRow = bytesPerRow
        self.rowsNum = size_Bytes // bytesPerRow
        self.columnsNumber = 0

        self.crossbarArray = [[] for x in range(self.rowsNum)]
        self.crossbarColumns = []

    ### ------------------------------------------------------------ ###
    # Set the width of each column
    def setColumns(self, column_widths):
        self.crossbarColumns = column_widths

    ### ------------------------------------------------------------ ###
    def loadData(self, column_width, column_data, start_row, column_index=-1):
        if column_index == -1 or column_index+1 > self.columnsNumber:
            self.crossbarColumns.append(column_width)
            for i in range(0, self.rowsNum):
                self.crossbarArray[i].append(None)

            for curr_row in range(start_row, min(self.rowsNum, start_row+len(column_data))):
                self.crossbarArray[curr_row][self.columnsNumber] = column_data[curr_row - start_row]

            self.columnsNumber += 1

        else:
            self.crossbarColumns.append(column_width)
            for curr_row in range(start_row, self.rowsNum):
                self.crossbarArray[curr_row][column_index] = column_data[curr_row - start_row]

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

        # cycle count
        return 3 * numOfRowsToShift * self.crossbarColumns[col_index]


    ### ------------------------------------------------------------ ###
    # Simple arithmetic - Add / Subtract
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


        if operation == '-' or operation == '+':
            if res_col != colA:
                cycles_per_bit = 2**3
            else:
                cycles_per_bit = 2**2
        elif operation == max_operation_string:
            cycles_per_bit = 2

        return cycles_per_bit * max(self.crossbarColumns[colA],self.crossbarColumns[colB])

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

        if operation == max_operation_string:
            cycles_per_bit = 2
        else:
            cycles_per_bit = 2 ** 2
        return cycles_per_bit * self.crossbarColumns[colA]

    ### ------------------------------------------------------------ ###
    # Fixed-point multiplication
    def MUL(self, start_row, end_row, colRes, colA, colB):
        for i in range(start_row, end_row):
            self.crossbarArray[i][colRes] = self.crossbarArray[i][colA] * self.crossbarArray[i][colB]

        # cycle count
        return (max(self.crossbarColumns[colA], self.crossbarColumns[colA]))**2

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


        if operation == max_operation_string:
            cycles_per_bit = 2
            return cycles_per_bit * self.crossbarColumns[col_index], result, result_row_index
        else:
            cycles_per_bit = 2 ** 2
            return cycles_per_bit * self.crossbarColumns[col_index] * math.ceil( math.log( len(self.crossbarColumns[col_index])))


    ### ------------------------------------------------------------ ###
    # Print array contents
    def printArray(self, start_row=0, end_row=-1, start_col=0, end_col=-1, header="", tablefmt="plain"):
        if end_row == -1: end_row=self.rowsNum
        if end_col == -1: end_col = self.rowsNum

        # for row in range(start_row, end_row):
        #     print(self.crossbarArray[row])

        print(tabulate(self.crossbarArray, header, tablefmt, stralign="center")) #other format option is "grid"

    ### ------------------------------------------------------------ ###
    # Calculate match score
    def DNAbpMatch(self, colA, colB, res_col, start_row, end_row, bp_match_score, bp_mismatch_score):
        for curr_row in range(start_row, end_row+1):
            is_bp_match = (self.crossbarArray[curr_row][colA] == self.crossbarArray[curr_row][colB])
            self.crossbarArray[curr_row][res_col] = bp_match_score if is_bp_match else bp_mismatch_score

        return 2 + 4*(max(self.crossbarColumns[colA], self.crossbarColumns[colA]))

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