import os,sys
import ReCAM, Simulator
import Serial_SmithWaterman

'''lib_path = os.path.abspath(os.path.join('swalign-0.3.3'))
sys.path.append(lib_path)
import swalign'''


def SW_test():
    # choose your own values here… 2 and -1 are common.
    Serial_SmithWaterman.main("TGCA", "ATGCCAGT")

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