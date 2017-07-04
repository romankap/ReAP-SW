import os
import sys
import ReCAM
import ProteinSequencing.protein_matrix_class as Protein_Matrix_Class
import ProteinSequencing.BLOSUM62 as BLOSUM62

lib_path = os.path.abspath(os.path.join('spfpm-1.1'))
sys.path.append(lib_path)

DNA_bp_bits = 2
amino_acid_bits = 5
DNA_match_score = 2; DNA_mismatch_score = -1; DNA_gap_first = -1; DNA_gap_extend = -1

def test_func(test_param):
    test_param += 1

def getOperatingRows(iteration, offset, lenA, lenB):
    start_row = -1; end_row = -1

    if iteration < lenB:
        start_row = 0; end_row = iteration
    elif iteration < lenA:
        start_row = iteration - lenB + 1; end_row = iteration
    else:
        start_row = iteration - lenB + 1; end_row = lenA - 1

    return start_row+offset, end_row+offset


def place_DB_seq_in_ReCAM(storage, start_row, first_row_col, last_row_col, seq_col, DB_seq_column_bits, seq_as_list):
    # 1. set first_row bit

    first_last_row_vec = [0] * len(seq_as_list)
    first_last_row_vec[0] = 1
    storage.loadData(first_last_row_vec, start_row, 1, first_row_col)

    # 2. set last_row bit
    first_last_row_vec[0] = 0
    first_last_row_vec[len(seq_as_list)-1] = 1
    storage.loadData(first_last_row_vec, start_row, 1, last_row_col)

    # 3. load sequence
    storage.loadData(seq_as_list, start_row, DB_seq_column_bits, seq_col)


def Parallel_SW_on_ReCAM(DB_sequences, query_seq):
    storage = ReCAM.ReCAM(32768)

    verbose_prints = False
    if verbose_prints:
        print("size in bytes = ", storage.sizeInBytes)
        print("bits per row = ", storage.bitsPerRow)

    zero_vector = [0] * storage.rowsNum

    table_header_row = ["first_row_bit", "last_row_bit", "active_row_bit", "seqA", "seqB", "E[]", "F[]", "AD[0]", "AD[1]", "AD[2]", "Max-AD-Scores", "temp[]"]
    storage.setVerbose(verbose_prints)
    storage.setPrintHeader(table_header_row)

    # Definitions
    first_row_col = 0; last_row_col = 1; active_bit_col = 2
    DB_seq_col = 3; query_seq_col = 4
    E_col_index = 5; F_col_index = 6; first_AD_col_index = 7; last_AD_col_index = 8; max_AD_scores_col_index = 9; temp_col_index = 10
    total_max_score = 0; total_max_row_index = 0; total_max_col_index = 0

    for i in range(3):
        storage.loadData(zero_vector, 0, 1) # load first/last row columns
    for i in range(2):
        storage.loadData(zero_vector, 0, 5) # load protein columns
    for i in range(6):
        storage.loadData(zero_vector, 0, 32)  # load scores columns

    copy_start_row = 0; max_seq_len=0
    for seq in DB_sequences:
        seq_as_list = list(seq)
        place_DB_seq_in_ReCAM(storage, copy_start_row, first_row_col, last_row_col, DB_seq_col, amino_acid_bits, seq_as_list)
        copy_start_row += len(seq)
        if max_seq_len < len(seq_as_list):
            max_seq_len = len(seq_as_list)

    alg_end_row = copy_start_row
    alg_start_row = 0
    query_seq_len = len(query_seq)

    ##### --- Algorithm --- #####

    # Init
    # 1) Copy first_row bit to active_bit
    # 2) Copy first query_seq character

    protein_matrix = Protein_Matrix_Class.blosum_matrix(BLOSUM62.full_blosum62())
    ReCAM.set_match_matrix('protein', protein_matrix)
    #for i in range (0, len(seqA)+len(seqB)+2):
    for i in range(0, max_seq_len + query_seq_len - 1):
        # Alg iteration outline:
        # 1) Shift query seq down on all active rows
        # 2) shift down active rows bit
        # 3) Once right_AD score was calculated, max-copy it to a max-AD-scores column

        # x) After the execution has finished, perform a max-reduction on max-AD-scores of all DB sequences

        #start_row, end_row = getOperatingRows(i, seqA_start_row, len(seqA), len(seqB))

        right_AD = (i % 3) + first_AD_col_index
        middle_AD = ((i-1) % 3) + first_AD_col_index
        left_AD = ((i-2) % 3) + first_AD_col_index

        if i==0:
            tagged_rows_list = storage.tagRowsEqualToConstant(first_row_col, 1, alg_start_row, alg_end_row)
            storage.taggedRowWiseOperation(first_row_col, None, active_bit_col, tagged_rows_list, 'copy')
            storage.taggedRowWiseOperationWithConstant(None, query_seq[0], query_seq_col, tagged_rows_list, 'write')
        else:
            tagged_rows_list = storage.tagRowsEqualToConstant(active_bit_col, 1, alg_start_row, alg_end_row)
            storage.shiftColumnOnTaggedRows(query_seq_col, tagged_rows_list)
            storage.shiftColumnOnTaggedRows(left_AD, tagged_rows_list)

        storage.SeqMatchOnTaggedRows(DB_seq_col, query_seq_col, temp_col_index, tagged_rows_list)
        storage.taggedRowWiseOperation(left_AD, temp_col_index, right_AD, tagged_rows_list, '+')
        storage.taggedRowWiseOperation(right_AD, 0, right_AD, tagged_rows_list, "max")

        storage.taggedRowWiseOperationWithConstant(middle_AD, DNA_gap_first, left_AD, tagged_rows_list, '+')
        storage.taggedRowWiseOperationWithConstant(F_col_index, DNA_gap_extend, temp_col_index, tagged_rows_list, '+')
        storage.taggedRowWiseOperation(left_AD, temp_col_index, F_col_index, tagged_rows_list, "max")
        storage.taggedRowWiseOperation(right_AD, F_col_index, right_AD, tagged_rows_list, "max")

        # Shifting active-bit one row up to shift row values down. Later the bit will be shifted back down
        storage.shiftColumnOnTaggedRows(active_bit_col, tagged_rows_list, -1)
        tagged_rows_list = storage.tagRowsEqualToConstant(active_bit_col, 1, alg_start_row, alg_end_row)
        storage.shiftColumnOnTaggedRows(E_col_index, tagged_rows_list, 1)
        storage.shiftColumnOnTaggedRows(left_AD, tagged_rows_list, 1)

        # Shifting active-bit one row down.
        storage.shiftColumnOnTaggedRows(active_bit_col, tagged_rows_list, 1)
        tagged_rows_list = storage.tagRowsEqualToConstant(active_bit_col, 1, alg_start_row, alg_end_row)
        storage.taggedRowWiseOperationWithConstant(E_col_index, DNA_gap_extend, E_col_index, tagged_rows_list, '+')
        storage.taggedRowWiseOperation(left_AD, E_col_index, E_col_index, tagged_rows_list, "max")

        storage.taggedRowWiseOperation(right_AD, E_col_index, right_AD, tagged_rows_list, "max")
        storage.taggedRowWiseOperation(right_AD, max_AD_scores_col_index, right_AD, tagged_rows_list, "max")


    print("=== ReCAM Cycles executed: ", storage.getCyclesCounter())
    print("* SeqA length = ", len(seqA), " seqB length = ", len(seqB))
    print("** Cycles: ", storage.getCyclesCounter())
    print("*** Performance (CUPs): ", len(seqA)*len(seqB) * storage.getFrequency()//storage.getCyclesCounter())
    return (total_max_score, total_max_row_index, total_max_col_index)

###################################################################

def test_Parallel_SW_on_ReCAM():
    seq_list = []
    seq_list.append("AGTTTC")
    seq_list.append("ACCG")
    Parallel_SW_on_ReCAM(seq_list, "TGCC")

test_Parallel_SW_on_ReCAM()