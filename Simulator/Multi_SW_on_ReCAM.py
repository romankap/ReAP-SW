import os
import sys
import ReCAM
import ProteinSequencing.protein_matrix_class as Protein_Matrix_Class
import ProteinSequencing.BLOSUM62 as BLOSUM62
import Serial_SmithWaterman

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


def place_DB_seq_in_ReCAM(storage, start_row, first_row_col, buffer_row_col, shift_E_left_AD_col,
                          seq_col, DB_seq_column_bits, seq_as_list):
    # 1. set first_row bit

    first_row_vec = [0] * (len(seq_as_list)+1)
    first_row_vec[0] = 0
    first_row_vec[1] = 1
    storage.loadData(first_row_vec, start_row, 1, first_row_col)

    # 2. set buffer_row bit (reset first_row_bit)
    buffer_row_vec = [0] * (len(seq_as_list) + 1)
    buffer_row_vec[0] = 1
    storage.loadData(buffer_row_vec, start_row, 1, buffer_row_col)

    temp_vec = [0] * (len(seq_as_list) + 1)
    temp_vec[0] = 1
    storage.loadData(temp_vec, start_row, 1, shift_E_left_AD_col)

    # 3. load sequence
    storage.loadData(seq_as_list, start_row+1, DB_seq_column_bits, seq_col)


def Multi_SW_on_ReCAM(DB_sequences, query_seq, sequence_type='DNA'):
    storage = ReCAM.ReCAM(256*len(DB_sequences)* len(max(DB_sequences, key=len)))

    storage.seq_type = sequence_type
    if sequence_type == 'protein':
        bits_per_sequence_char = amino_acid_bits
    else:
        bits_per_sequence_char = DNA_bp_bits


    verbose_prints = False
    if verbose_prints:
        print("size in bytes = ", storage.sizeInBytes)
        print("bits per row = ", storage.bitsPerRow)

    zero_vector = [0] * storage.rowsNum

    table_header_row = ["first_row_bit", "buffer_row_bit", "active_row_bit", "shift_E_left_AD_col",
                        "seqA", "seqB", "E[]", "F[]", "AD[0]", "AD[1]", "AD[2]", "Max-AD-Scores", "temp[]"]
    storage.setVerbose(verbose_prints)
    storage.setPrintHeader(table_header_row)

    # Definitions
    num_of_bit_columns = 4
    first_row_col = 0; buffer_row_col = 1; active_bit_col = 2; shift_E_left_AD_col = 3 #; top_active_row_bit_col = 4
    DB_seq_col = num_of_bit_columns; query_seq_col = num_of_bit_columns+1


    E_col_index = num_of_bit_columns+2; F_col_index = num_of_bit_columns+3; first_AD_col_index = num_of_bit_columns+4
    temp_col_index = num_of_bit_columns+7; max_AD_scores_col_index = num_of_bit_columns+8
    total_max_score = 0; total_max_row_index = 0; total_max_col_index = 0

    for i in range(num_of_bit_columns):
        storage.loadData(zero_vector, 0, 1) # load first/last row columns
    for i in range(2):
        storage.loadData(zero_vector, 0, bits_per_sequence_char) # load DNA/protein columns
    for i in range(7):
        storage.loadData(zero_vector, 0, 32)  # load scores columns

    # Load database of sequences to ReCAM
    # TODO: put the code in a separate function
    copy_start_row = 0; max_DB_seq_len=0; DB_seq_lengths=[]
    for seq in DB_sequences:
        seq_as_list = list(seq)
        place_DB_seq_in_ReCAM(storage, copy_start_row, first_row_col, buffer_row_col, shift_E_left_AD_col,
                              DB_seq_col, bits_per_sequence_char, seq_as_list)
        copy_start_row += (len(seq)+1)
        DB_seq_lengths.append(len(seq))
        if max_DB_seq_len < len(seq_as_list):
            max_DB_seq_len = len(seq_as_list)
    storage.loadData([1], copy_start_row, 1, buffer_row_col)


    alg_end_row = copy_start_row
    alg_start_row = 0
    query_seq_len = len(query_seq)

    ##### --- Algorithm --- #####

    # Init
    # 1) Copy first_row bit to active_bit
    # 2) Copy first query_seq character

    protein_matrix = Protein_Matrix_Class.blosum_matrix(BLOSUM62.full_blosum62())
    storage.set_match_matrix(sequence_type, protein_matrix, DNA_match_score, DNA_mismatch_score)
    #for i in range (0, len(seqA)+len(seqB)+2):
    for i in range(0, max_DB_seq_len + query_seq_len - 1):
        # Alg iteration outline:
        # 1) Shift query seq down on all active rows
        # 2) shift down active rows bit
        # 3) Once right_AD score was calculated, max-copy it to a max-AD-scores column

        # x) After the execution has finished, perform a max-reduction on max-AD-scores of all DB sequences

        #start_row, end_row = getOperatingRows(i, seqA_start_row, len(seqA), len(seqB))

        right_AD = (i % 3) + first_AD_col_index
        middle_AD = ((i-1) % 3) + first_AD_col_index
        left_AD = ((i-2) % 3) + first_AD_col_index

        # tagged_rows_list = storage.tagRowsEqualToConstant(buffer_row_col, 1, alg_start_row, alg_end_row)
        # storage.taggedRowWiseOperationWithConstant(None, 0, right_AD, tagged_rows_list, 'write')
        # storage.taggedRowWiseOperationWithConstant(None, 0, middle_AD, tagged_rows_list, 'write')
        # storage.taggedRowWiseOperationWithConstant(None, 0, E_col_index, tagged_rows_list, 'write')
        # storage.taggedRowWiseOperationWithConstant(None, 0, F_col_index, tagged_rows_list, 'write')
        # storage.taggedRowWiseOperationWithConstant(None, 0, temp_col_index, tagged_rows_list, 'write')

        if i > 0:
            if i < len(query_seq):
                tagged_rows_list = storage.tagRowsEqualToConstant(active_bit_col, 1, alg_start_row, alg_end_row)
            else: # All query sequence characters were written
                if i == len(query_seq):
                    tagged_rows_list = storage.tagRowsEqualToConstant(first_row_col, 1, alg_start_row, alg_end_row)
                    storage.taggedRowWiseOperationWithConstant(None, 0, active_bit_col, tagged_rows_list, 'write')

                    tagged_rows_list = storage.tagRowsEqualToConstant(buffer_row_col, 1, alg_start_row, alg_end_row)
                    storage.taggedRowWiseOperationWithConstant(None, 0, shift_E_left_AD_col, tagged_rows_list, 'write')

                    tagged_rows_list = storage.tagRowsEqualToConstant(active_bit_col, 1, alg_start_row, alg_end_row)
                    #tagged_rows_list = storage.tagRowsEqualToConstant(shift_E_left_AD_col, 1, alg_start_row, alg_end_row, tagged_rows_list)
                    tagged_rows_list = storage.untagRowsNotEqualToConstant(buffer_row_col, 0, tagged_rows_list)
                else:
                    tagged_rows_list = storage.tagRowsEqualToConstant(active_bit_col, 1, alg_start_row, alg_end_row)
                    tagged_rows_list = storage.tagRowsEqualToConstant(shift_E_left_AD_col, 1, alg_start_row,
                                                                      alg_end_row, tagged_rows_list)
                    tagged_rows_list = storage.untagRowsNotEqualToConstant(buffer_row_col, 0, tagged_rows_list)
                # Adding trailing zeros to active_bit_col

            storage.taggedRowWiseOperation(active_bit_col, None, shift_E_left_AD_col, tagged_rows_list, 'copy')
            storage.shiftColumnOnTaggedRows(active_bit_col, tagged_rows_list)
            tagged_rows_list = storage.tagRowsEqualToConstant(shift_E_left_AD_col, 1, alg_start_row, alg_end_row)
            tagged_rows_list = storage.untagRowsNotEqualToConstant(buffer_row_col, 0, tagged_rows_list)
            storage.shiftColumnOnTaggedRows(query_seq_col, tagged_rows_list)

            #prevent shifting non-zero values
            tagged_rows_list = storage.tagRowsEqualToConstant(buffer_row_col, 1, alg_start_row, alg_end_row)
            storage.taggedRowWiseOperationWithConstant(None, 0, left_AD, tagged_rows_list, 'write')
            tagged_rows_list = storage.tagRowsEqualToConstant(shift_E_left_AD_col, 1, alg_start_row, alg_end_row)
            storage.shiftColumnOnTaggedRows(left_AD, tagged_rows_list)

            #Write query_seq to top row
            if i < len(query_seq):
                tagged_rows_list = storage.tagRowsEqualToConstant(first_row_col, 1, alg_start_row, alg_end_row)
                storage.taggedRowWiseOperationWithConstant(None, query_seq[i], query_seq_col, tagged_rows_list, 'write')
        else: # i == 0
            tagged_rows_list = storage.tagRowsEqualToConstant(first_row_col, 1, alg_start_row, alg_end_row)
            storage.taggedRowWiseOperation(first_row_col, None, active_bit_col, tagged_rows_list, 'copy')
            #storage.taggedRowWiseOperation(first_row_col, None, shift_E_left_AD_col, tagged_rows_list, 'copy')
            storage.taggedRowWiseOperationWithConstant(None, query_seq[i], query_seq_col, tagged_rows_list, 'write')
        #tagged_rows_list = storage.tagRowsEqualToConstant(active_bit_col, 1, alg_start_row, alg_end_row)

        tagged_rows_list = storage.tagRowsEqualToConstant(active_bit_col, 1, alg_start_row, alg_end_row)
        storage.SeqMatchOnTaggedRows(DB_seq_col, query_seq_col, temp_col_index, tagged_rows_list)
        storage.taggedRowWiseOperation(left_AD, temp_col_index, right_AD, tagged_rows_list, '+')
        storage.taggedRowWiseOperationWithConstant(right_AD, 0, right_AD, tagged_rows_list, "max")

        storage.taggedRowWiseOperationWithConstant(middle_AD, DNA_gap_first, left_AD, tagged_rows_list, '+')
        storage.taggedRowWiseOperationWithConstant(F_col_index, DNA_gap_extend, temp_col_index, tagged_rows_list, '+')
        storage.taggedRowWiseOperation(left_AD, temp_col_index, F_col_index, tagged_rows_list, "max")
        storage.taggedRowWiseOperation(right_AD, F_col_index, right_AD, tagged_rows_list, "max")

        #storage.shiftColumnOnTaggedRows(active_bit_col, tagged_rows_list, -1)
        # prevent shifting non-zero values: zero buffer row
        tagged_rows_list = storage.tagRowsEqualToConstant(buffer_row_col, 1, alg_start_row, alg_end_row) #Only in multi-SW
        storage.taggedRowWiseOperationWithConstant(None, 0, left_AD, tagged_rows_list, 'write') #Only in multi-SW
        storage.taggedRowWiseOperationWithConstant(None, 0, E_col_index, tagged_rows_list, 'write') #Only in multi-SW
        #shift values
        tagged_rows_list = storage.tagRowsEqualToConstant(shift_E_left_AD_col, 1, alg_start_row, alg_end_row)
        storage.shiftColumnOnTaggedRows(E_col_index, tagged_rows_list, 1)
        storage.shiftColumnOnTaggedRows(left_AD, tagged_rows_list, 1)

        # Shifting active-bit one row down.
        #storage.shiftColumnOnTaggedRows(active_bit_col, tagged_rows_list, 1)
        tagged_rows_list = storage.tagRowsEqualToConstant(active_bit_col, 1, alg_start_row, alg_end_row)
        tagged_rows_list = storage.untagRowsNotEqualToConstant(buffer_row_col, 0, tagged_rows_list)
        storage.taggedRowWiseOperationWithConstant(E_col_index, DNA_gap_extend, E_col_index, tagged_rows_list, '+')
        storage.taggedRowWiseOperation(left_AD, E_col_index, E_col_index, tagged_rows_list, "max")

        storage.taggedRowWiseOperation(right_AD, E_col_index, right_AD, tagged_rows_list, "max")

        # update max scores in max_AD_scores_col
        storage.taggedRowWiseOperation(right_AD, max_AD_scores_col_index, max_AD_scores_col_index, tagged_rows_list, "max")

    max_scores = get_max_scores_from_DB_alignment(storage, DB_sequences, max_AD_scores_col_index)

    print("=== ReCAM Cycles executed: ", storage.getCyclesCounter())
    print("* Query seq length = ", query_seq_len, ". Max DB seq length = ", max_DB_seq_len)
    #print("** Cycles: ", storage.getCyclesCounter())
    #print("*** Performance (CUPs): ", len(seqA)*len(seqB) * storage.getFrequency()//storage.getCyclesCounter())
    return max_scores

###################################################################

def get_max_scores_from_DB_alignment(storage, DB_seq_list, column_index):
    # find Max scores
    seq_start, seq_end = 0, 0
    max_scores = []
    for DB_seq in DB_seq_list:
        seq_start = seq_end + 1
        seq_end = seq_start + len(DB_seq)
        max_scalar = storage.find_max_scalar_in_rows_range(column_index, seq_start, seq_end)
        max_scores.append(max_scalar)

    return max_scores

def test_Multi_SW_on_ReCAM():
    DB_seq_list = []
    DB_seq_list.append("TGGCCCT")
    DB_seq_list.append("TCGC")
    DB_seq_list.append("AATAGCGAG")
    DB_seq_list.append("TTTGC")

    query_seq = "GC"
    max_scores = Multi_SW_on_ReCAM(DB_seq_list, query_seq, 'DNA')
    print("Mutli-SW Scores:", max_scores)

    serial_execution_scores = []
    for DB_seq in DB_seq_list:
        serial_execution_scores.append(Serial_SmithWaterman.main(input_seqA=DB_seq, input_seqB=query_seq)[0])

    print("Serial results:", serial_execution_scores)

#test_Multi_SW_on_ReCAM()