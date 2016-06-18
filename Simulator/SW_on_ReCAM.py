import os,sys, time

import ReCAM, Simulator

'''lib_path = os.path.abspath(os.path.join('swalign-0.3.3'))
sys.path.append(lib_path)
import swalign'''

DNA_match_score = 2; DNA_mismatch_score = -1; DNA_gap_first = -1; DNA_gap_extend = -1

def SW_on_ReCAM(input_seq1="ATGCAT", input_seq2="TGCAAG"):
    storage = ReCAM.ReCAM(512)
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

    #for i in range (0, len(seq1)+len(seq2)+2):
    for i in range (0, 2):
        right_AD = (i % 3) + first_AD_col_index;
        middle_AD = ((i-1) % 3) + first_AD_col_index;
        left_AD = ((i-2) % 3) + first_AD_col_index;

        storage.shiftColumn(seqB_col_index, i, len(seq2)+i-1, 1)
        storage.DNAbpMatch(seqA_col_index, seqB_col_index, temp_col_index, first_row, first_row+i, DNA_match_score, DNA_mismatch_score)
        storage.rowWiseOperation(left_AD, temp_col_index, right_AD, first_row, first_row+i, '+')
        storage.rowWiseOperationWithConstant(right_AD, 0, right_AD, first_row, first_row+i, "max")

        storage.rowWiseOperationWithConstant(middle_AD, DNA_gap_first, left_AD, first_row, first_row+i, '-')
        storage.rowWiseOperationWithConstant(F_col_index, DNA_gap_extend, temp_col_index, first_row, first_row+i, '-')
        storage.rowWiseOperation(left_AD, temp_col_index, F_col_index, first_row, first_row + i, "max")
        storage.rowWiseOperation(right_AD, F_col_index, right_AD, first_row, first_row + i, "max")

        storage.rowWiseOperationWithConstant(E_col_index, DNA_gap_extend, temp_col_index, first_row, first_row + i, '-')
        storage.rowWiseOperation(left_AD, temp_col_index, E_col_index, first_row, first_row + i, "max")
        storage.shiftColumn(E_col_index, i, len(seq2) + i - 1, 1)
        storage.rowWiseOperation(right_AD, E_col_index, right_AD, first_row, first_row + i, "max")

        time.sleep(2)
        os.system('cls')
        storage.printArray(header=["seqA", "seqB", "E[]", "F[]", "AD[0]", "AD[1]", "AD[2]", "temp[]"], tablefmt="grid")

SW_on_ReCAM()