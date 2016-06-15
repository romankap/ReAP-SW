import os,sys
import ReCAM, Simulator

'''lib_path = os.path.abspath(os.path.join('swalign-0.3.3'))
sys.path.append(lib_path)
import swalign'''


def SW_on_ReCAM(input_seq1="ATGCAT", input_seq2="TGCAAG"):
    storage = ReCAM.ReCAM(2048)
    print("size in bytes = ", storage.sizeInBytes)
    print("bits per row = ", storage.bitsPerRow)

    seq1 = list(input_seq1)
    seq2 = list(input_seq2)
    zero_vector = [0]*(len(seq1)+len(seq2)+1)

    # Pushing seq2 above seq1 and in an adjacent column
    # In every iteration, seq2 will be pushed down and the appropriate rows will be compared

    first_row = len(seq2)
    storage.loadData(2, seq2, 0)

    storage.loadData(2, seq1, first_row, 1)

    for i in range(6):
        storage.loadData(32, zero_vector, 0)

    # Definitions
    E_col = 2; F_col = 3; first_AD_col = 4; last_AD_col = 6; temp_col = 7


    storage.printArray()

SW_on_ReCAM()