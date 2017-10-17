import os,sys
import ProteinSequencing.BLOSUM62 as BLOSUM62
import operator


class blosum_matrix:
    def __init__(self, blosum_full_matrix):
        self.match_dict = blosum_full_matrix
        self.init_cycles_per_ReCAM_match()

    def init_cycles_per_ReCAM_match(self):
        scores_dict = {}

        for (prot_a, prot_b), score in self.match_dict.items():
            # print(prot_a, prot_a, score)
            scores_dict[score] = scores_dict.get(score, 0) + 1

        occurences_sum = 0
        for (score, occurences) in scores_dict.items():
            occurences_sum += occurences

        saved_ReCAM_cycles = occurences_sum - len(scores_dict)
        self.match_table_rows = len(self.match_dict)
        self.batched_write_saved_cycles = saved_ReCAM_cycles


def analyze_BLOSUM62():
    blosum_mat = BLOSUM62.full_blosum62()
    analyze_protein_matrix(blosum_mat)

def analyze_protein_matrix(protein_matrix):
    proteins_dict = {}

    score_dict = {}
    for (prot_a, prot_b), score in protein_matrix.items():
        #print(prot_a, prot_a, score)
        score_dict[score] = score_dict.get(score, 0) + 1
        proteins_dict[prot_a] = proteins_dict.get(prot_a, 1) + 1

    sorted_score_dict = sorted(score_dict.items(), key=operator.itemgetter(1), reverse=True)
    occurences_sum = 0
    for (score, occurences) in sorted_score_dict:
        print(score, occurences)
        occurences_sum += occurences

    saved_ReCAM_cycles = occurences_sum - len(score_dict)
    print("Protein Matrix Records", len(protein_matrix))
    print("Protein Matrix different proteins", len(proteins_dict))
    print("Number of different scores", len(score_dict))
    print("Saved ReCAM Cycles", saved_ReCAM_cycles)

analyze_BLOSUM62()
