import ReCAM, Simulator

def test():
    simulator = Simulator.Simulator()
    storage=ReCAM.ReCAM(1000)
    print("size in bytes = ", storage.sizeInBytes)
    print("bits per row = ", storage.bitsPerRow)

    firstCol = [1,2,3,4]
    secondCol = [10,20,30,40]

    storage.loadData(32, 0, firstCol, 0, len(firstCol))
    storage.loadData(32, 1, secondCol, 0, len(secondCol))

    print("==== Performing ADD ==== ")
    simulator.execute(storage.addSub(0, 4, 0, 0, 1, '-'))

    print("==== printArray() ==== ")
    storage.printArray(0,2,0,2)

    print("==== Printing ReCAM contents==== ")
    for i in range(len(firstCol)):
        print("i =",i, ":", storage.crossbarArray[i][0],storage.crossbarArray[i][1])



    print("total cycles = ", simulator.getCycleCount())

test()