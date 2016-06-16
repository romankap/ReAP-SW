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

    #Initialization
    seqA_col_index = 0; seqB_col_index = 1
    storage.loadData(2, seq1, first_row)
    storage.loadData(2, seq2, 0, seqB_col_index)

    for i in range(6):
        storage.loadData(32, zero_vector, 0)

    # Definitions
    E_col_index = 2; F_col_index = 3; first_AD_col_index = 4; last_AD_col_index = 6; temp_col_index = 7

    # for i in range (0, len(seq1)+len(seq2)+2):
    #     right_AD = (i % 3) + first_AD_col_index;
    #     middle_AD = ((i-1) % 3) + first_AD_col_index;
    #     left_AD = ((i-2) % 3) + first_AD_col_index;
    #
    #     storage.shiftColumn(0, len(seq2)-1, seqB_col_index,1)

    storage.shiftColumn(0, len(seq2)-1, seqB_col_index, 1)


    storage.printArray()

SW_on_ReCAM()