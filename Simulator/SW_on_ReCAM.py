import os,sys, time

import ReCAM, Simulator

'''lib_path = os.path.abspath(os.path.join('swalign-0.3.3'))
sys.path.append(lib_path)
import swalign'''

DNA_match_score = 2; DNA_mismatch_score = -1; DNA_gap_first = -1; DNA_gap_extend = -1

def test_func(test_param):
    test_param += 1

def getOperatingRows(i, offset, lenA, lenB):
    start_row = -1; end_row = -1

    if i < lenB:
        start_row = 0; end_row = i
    elif i < lenA:
        start_row = i-lenB+1; end_row = i
    else:
        start_row = i-lenA+lenB-1; end_row = lenA-1

    return start_row+offset, end_row+offset

def SW_on_ReCAM(input_seqA="AGCT", input_seqB="GCT"):
    storage = ReCAM.ReCAM(512)
    print("size in bytes = ", storage.sizeInBytes)
    print("bits per row = ", storage.bitsPerRow)

    seqA = list(input_seqA)
    seqB = list(input_seqB)
    zero_vector = [0]*(len(seqA)+len(seqB)+1)

    # Pushing seqB above seqA and in an adjacent column
    # In every iteration, seqB will be pushed down and the appropriate rows will be compared

    seqA_start_row = len(seqB)
    rev_seqB = seqB
    rev_seqB.reverse()

    #Initialization
    seqA_col_index = 0; seqB_col_index = 1
    storage.loadData(2, seqA, seqA_start_row)
    storage.loadData(2, rev_seqB, 0, seqB_col_index)

    for i in range(6):
        storage.loadData(32, zero_vector, 0)
    table_header_row = ["seqA", "seqB", "E[]", "F[]", "AD[0]", "AD[1]", "AD[2]", "temp[]"]


    # Definitions
    E_col_index = 2; F_col_index = 3; first_AD_col_index = 4; last_AD_col_index = 6; temp_col_index = 7
    total_max_score = 0; total_max_row_index = 0; total_max_col_index = 0

    storage.setVerbose(True)
    storage.setPrintHeader(table_header_row)

    #for i in range (0, len(seqA)+len(seqB)+2):
    for i in range (0, len(seqA)+len(seqB)-1):
        start_row, end_row = getOperatingRows(i, seqA_start_row, len(seqA), len(seqB))

        right_AD = (i % 3) + first_AD_col_index
        middle_AD = ((i-1) % 3) + first_AD_col_index
        left_AD = ((i-2) % 3) + first_AD_col_index

        storage.shiftColumn(seqB_col_index, i, len(seqB)+i-1, 1) # Prepare SeqB for match score

        if i >= len(seqB):
            storage.shiftColumn(left_AD, start_row-1, end_row-1, 1)
        else:
            storage.shiftColumn(left_AD, start_row, end_row, 1)

        storage.DNAbpMatch(seqA_col_index, seqB_col_index, temp_col_index, start_row, end_row, DNA_match_score, DNA_mismatch_score)
        storage.rowWiseOperation(left_AD, temp_col_index, right_AD, start_row, end_row, '+')
        storage.rowWiseOperationWithConstant(right_AD, 0, right_AD, start_row, end_row, "max")

        if i >= len(seqB):
            storage.rowWiseOperationWithConstant(middle_AD, DNA_gap_first, left_AD, start_row-1, end_row, '+')
        else:
            storage.rowWiseOperationWithConstant(middle_AD, DNA_gap_first, left_AD, start_row, end_row, '+')

        storage.rowWiseOperationWithConstant(F_col_index, DNA_gap_extend, temp_col_index, start_row, end_row, '+')
        storage.rowWiseOperation(left_AD, temp_col_index, F_col_index, start_row, end_row, "max")
        storage.rowWiseOperation(right_AD, F_col_index, right_AD, start_row, end_row, "max")
        storage.rowWiseOperationWithConstant(E_col_index, DNA_gap_extend, temp_col_index, start_row, end_row, '+')

        if i >= len(seqB):
            storage.shiftColumn(left_AD, start_row-1, end_row-1, 1)
        else:
            storage.shiftColumn(left_AD, start_row, end_row, 1)
        storage.rowWiseOperation(left_AD, temp_col_index, E_col_index, start_row, end_row, "max")
        storage.rowWiseOperation(right_AD, E_col_index, right_AD, start_row, end_row, "max")

        (cycles, max_score_in_column, row_of_max_score_in_column) = storage.getScalarFromColumn(right_AD, start_row, end_row, "max")
        if max_score_in_column > total_max_score:
            total_max_score = max_score_in_column
            total_max_row_index = row_of_max_score_in_column-(seqA_start_row-1) #seqA starts from row 1
            total_max_col_index = i-total_max_row_index+2

        #time.sleep(1)
        os.system('cls')
        print("\n\n\n")
        print("Max score = ", total_max_score, ", in (row,col)=(", total_max_row_index, ", ", total_max_col_index, ")")
        print("\n")
        storage.printArray(header=table_header_row, tablefmt="grid")

SW_on_ReCAM()