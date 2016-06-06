__author__ = 'Roman'


class Simulator:
    cycleCount = 0

    class ReCAM:
        def __init__(self, sizeInBytes, bitsPerRow=128):
            self.sizeInBytes = sizeInBytes
            self.bitsPerRow = bitsPerRow
            self.bytesPerRow = bitsPerRow/8
            self.rowsNum = sizeInBytes / (bitsPerRow/8)

            self.crossbarArray = [[] for x in xrange(self.rowsNum)]
            self.crossbarColumns = []

        def initColumns(self, columnWidths):
            self.crossbarColumns = columnWidths

        def add(self, start_row, end_row, colA, colB, colRes):
            for i in range(start_row,end_row):
                self.crossbarArray[i][colRes] = self.crossbarArray[i][colA] + self.crossbarArray[i][colB]

            Simulator.cycleCount += max(self.crossbarColumns[colA],self.crossbarColumns[colA])
            self.crossbarColumns[colRes] = max(self.crossbarColumns[colA],self.crossbarColumns[colA])

'''
def test():
    for i in range (2):
        tmp = Simulator.ReCAM(100+i)
        print "size in bytes = ",tmp.sizeInBytes
        print "bits per row = ", tmp.bitsPerRow


test()
'''