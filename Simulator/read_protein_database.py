import os,sys


def get_protein_database_stats():
    total_num_of_sequences=0
    total_amino_acids=0
    directory_in_str = 'C:\\Dev\\Protein Databases\\'
    directory = directory_in_str

    for file in os.listdir(directory):
        filename = file
        if filename.endswith(".tab"):
            print(os.path.join(directory, filename))
            with open(os.path.join(directory, filename)) as f:
                file_num_of_sequences = 0
                file_amino_acids = 0

                for line in f:
                    val = line.split()[1]
                    if val == "Length":
                        continue
                    else:  # val is an integer
                        val = int(val)

                    file_amino_acids += val
                    file_num_of_sequences += 1

                total_amino_acids += file_amino_acids
                total_num_of_sequences += file_num_of_sequences
                print("File amino acids: ", file_amino_acids)
                print("File number of protein sequences: ", file_num_of_sequences)
        else:
            continue

    print("Total amino acids in database: ", total_amino_acids)
    print("Total number of protein sequences in database: ", total_num_of_sequences)


get_protein_database_stats()
