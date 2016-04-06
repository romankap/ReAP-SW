__author__ = 'Roman'


class ReAP:
    def __init__(self, sizeInBytes, bitsPerRow=128):
        self.sizeInBytes = sizeInBytes
        self.bitsPerRow = bitsPerRow
        self.bytesPerRow = bitsPerRow/8
        self.rowsNum = sizeInBytes / (bitsPerRow/8)


def test():
    tmp = ReAP(100)
    print "size in bytes = ",tmp.sizeInBytes


test()