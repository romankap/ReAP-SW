import os,sys
import random

import ReCAM, Simulator
import SW_on_ReCAM
import Serial_SmithWaterman

'''lib_path = os.path.abspath(os.path.join('swalign-0.3.3'))
sys.path.append(lib_path)
import swalign'''


def compare_SW_max_score(serial_result, ReCAM_result):
    if serial_result != ReCAM_result:
        return False
    return True


def get_random_basepair():
    bp = random.randint(0,3)
    if bp == 0:
        return 'A'
    elif bp == 1:
        return 'C'
    elif bp == 2:
        return 'G'
    return 'T'


def get_random_sequence(len):
    seq = []
    for i in range(0, len):
        seq.append(get_random_basepair())

    return seq


def generate_random_sequences():
    seqA_len = random.randint(10,128)
    seqA = get_random_sequence(seqA_len)

    seqB_len = random.randint(10,seqA_len)
    seqB = get_random_sequence(seqB_len)

    return (seqA, seqB)


def SW_test():
    random.seed()
    # choose your own values here… 2 and -1 are common.
    for i in range (0,100):
        (seqA, seqB) = generate_random_sequences()

        serial_result = Serial_SmithWaterman.main(input_seqA=seqA, input_seqB=seqB)
        ReCAM_result = SW_on_ReCAM.SW_on_ReCAM(input_seqA=seqA, input_seqB=seqB)

        if not compare_SW_max_score(serial_result, ReCAM_result):
            print("!!!!!!!!!!!!!!!!")
            print("! ERROR: Serial != ReCAM !")
            print("Serial result: ", serial_result)
            print("ReCAM result: ", ReCAM_result)
            print("(seqA, seqB): ", seqA, ", ", seqB)
            print("!!!!!!!!!!!!!!!!")


def test():
    simulator = Simulator.Simulator()
    storage=ReCAM.ReCAM(1000)
    print("size in bytes = ", storage.sizeInBytes)
    print("bits per row = ", storage.bitsPerRow)

    firstCol = [1,2,3,4]
    secondCol = [10,20,30,40]

    storage.loadData(32, firstCol, 0, len(firstCol))
    storage.loadData(32, secondCol, 0, len(secondCol))

    print("==== Performing ADD ==== ")
    simulator.execute(storage.addSub(0, 4, 0, 0, 1, '-'))

    print("==== printArray() ==== ")
    storage.printArray(0,2,0,2)

    print("==== Printing ReCAM contents==== ")
    for i in range(len(firstCol)):
        print("i =",i, ":", storage.crossbarArray[i][0],storage.crossbarArray[i][1])


    print("total cycles = ", simulator.getCycleCount())

#test()
SW_test()