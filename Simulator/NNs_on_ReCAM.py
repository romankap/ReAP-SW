import os, sys, time

import ReCAM, Simulator
import FixedPoint

'''lib_path = os.path.abspath(os.path.join('swalign-0.3.3'))
sys.path.append(lib_path)
import swalign'''


''' Define a NN structure and the appropriate column to store.
Every layer has number of neurons - the net will be presented as a list.
First (last) layer is input (output) layer.
'''
def createFullyConnectNN():
    network_structure = [3,5,1]
    print("Create FC networks")

def feedFrowardNN(input):
    print("Feed forward")

def trainOnSingleInput(input):
    print("FW + BW in NN")

#SW_on_ReCAM()