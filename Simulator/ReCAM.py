__author__ = 'Roman'

try:
    import matplotlib, numpy
    matplotlib.use('qt4agg')
    import matplotlib.pyplot as plt
    HAVE_MATPLOTLIB = True
except ImportError:
    HAVE_MATPLOTLIB = False

import os,sys
lib_path = os.path.abspath(os.path.join('spfpm-1.1'))
sys.path.append(lib_path)
import FixedPoint


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
    def shiftColumn(self, start_row, end_row, col, numOfRowsToShift=1):
        # Decide whether to shift up or down
        if numOfRowsToShift > 0: #Shift down
            shift_range = range(end_row, start_row)
        else:   #Shift up
            shift_range = range(start_row, end_row)

        for i in shift_range:
            self.crossbarArray[i+numOfRowsToShift][col] = self.crossbarArray[i][col]

        # Zero-fill empty rows
        if numOfRowsToShift > 0:  # Shift down
            zero_fill_range = range(start_row, start_row + numOfRowsToShift)
        else:  # Shift up
            zero_fill_range = range(start_row + numOfRowsToShift + 1, start_row)

        for j in zero_fill_range:
            self.crossbarArray[j][col] = 0

        # cycle count
        return 3 * self.crossbarColumns[col]

    ### ------------------------------------------------------------ ###
    # Simple arithmetic - Add / Subtract
    def addSub(self, start_row, end_row, res_col, colA, colB, operation):
        if operation == '+':
            for i in range(start_row,end_row+1):
                self.crossbarArray[i][res_col] = self.crossbarArray[i][colA] + self.crossbarArray[i][colB]

        elif operation == '-':
            for i in range(start_row, end_row+1):
                self.crossbarArray[i][res_col] = self.crossbarArray[i][colA] - self.crossbarArray[i][colB]

        if res_col != colA:
            cycles_per_bit = 2**3
        else:
            cycles_per_bit = 2**2

        return cycles_per_bit * max(self.crossbarColumns[colA],self.crossbarColumns[colB])

    ### ------------------------------------------------------------ ###
    # Simple variable-constant arithmetic  - Add / Subtract
    def addSubWithConstant(self, start_row, end_row, res_col, colA, const, operation):
        if operation == '+':
            for i in range(start_row, end_row):
                self.crossbarArray[i][res_col] = self.crossbarArray[i][colA] + const

        elif operation == '-':
            for i in range(start_row, end_row):
                self.crossbarArray[i][res_col] = self.crossbarArray[i][colA] - const

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
    # Print array contents
    def printArray(self, start_row=0, end_row=-1, start_col=0, end_col=-1):
        if end_row == -1: end_row=self.rowsNum
        if end_col == -1: end_col = self.rowsNum

        for row in range(start_row, end_row):
            print(self.crossbarArray[row])

'''
def test():
    for i in range (2):
        tmp = Simulator.ReCAM(100+i)
        print "size in bytes = ",tmp.sizeInBytes
        print "bits per row = ", tmp.bitsPerRow




test()
'''