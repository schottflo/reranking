import time
import autograd.numpy as np

from parsing.data_structures.dep_tree import Node, Arc, DependencyTree


def compute_num_matching_subgraphs_dp(t1, t2, lamb=1, show_DP_table=False):
    """
    Given two dependency trees, this function returns the number of matching subgraphs.

    :param t1: DependencyTree
    :param t2: DependencyTree
    :return: int
    """
    DP_table = {}

    # Initialize the DP entries for all words
    for word_t1 in t1.words():
        for word_t2 in t2.words():
            DP_table[(word_t1, word_t1.pos, word_t2.pos)] = 0.0

    # Run the algorithm from Collins & Duffy (2001) - Technical Report
    for word_t1, arc_t1 in t1:
        for word_t2, arc_t2 in t2:
            if word_t1 == word_t2:

                # Extract the set of common dependencies
                common_dep = arc_t1.compare_arcs(arc_t2)

                # If there are none, skip to the next word
                if len(common_dep) == 0:
                    continue

                # Else: Build the factors of the product
                factors = [(2 + DP_table[(child, arc_t1[ind].pos, arc_t2[ind_other].pos)]) for ind, ind_other, child in common_dep]

                # Update the DP table (No short hand notation for autograd)
                DP_table[(word_t1, word_t1.pos, word_t2.pos)] = \
                    DP_table[(word_t1, word_t1.pos, word_t2.pos)] + (lamb * np.prod(factors)) - 1

    if show_DP_table:
        print(DP_table)

    return sum(DP_table.values())


if __name__ == "__main__":

    from parsing.dep_trees.helpers.test_helpers import convert_to_custom_data_structure, parse_sentences

    sentence_1 = "Hello my dear friend, Mel ."  # The comma changes the grammatical relationship
    sentence_2 = "Hello my dear friend Ricardo ."

    dep_graph_1, dep_graph_2 = parse_sentences(sent_1=sentence_1, sent_2=sentence_2)
    t1, t2 = convert_to_custom_data_structure(dep_graph=dep_graph_1), convert_to_custom_data_structure(dep_graph=dep_graph_2)

    print(compute_num_matching_subgraphs_dp(t1=t1, t2=t2, lamb=1))