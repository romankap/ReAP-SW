__author__ = 'Roman'

import os,sys


class Simulator:
    def __init__(self):
        self.cycleCount = 0

    def reset(self):
        self.cycleCount = 0

    def getCycleCount(self):
        return self.cycleCount

    def execute(self, cycles_executed):
        self.cycleCount += cycles_executed

'''
def test():
    for i in range (2):
        tmp = Simulator.ReCAM(100+i)
        print "size in bytes = ",tmp.sizeInBytes
        print "bits per row = ", tmp.bitsPerRow




test()
'''