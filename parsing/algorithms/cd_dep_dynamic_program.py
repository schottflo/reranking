import autograd.numpy as np
from collections import Counter


def compute_num_matching_subgraphs_dp(t1, t2, lamb=1, show_DP_table=False):
    """
    Given two dependency trees, this function returns the number of matching subgraphs.

    :param t1: DependencyTree
    :param t2: DependencyTree
    :return: int
    """
    DP_table = {}

    # Initialize the DP entries for all words
    for word_t1 in t1.nodes():#t1.words():
        for word_t2 in t2.nodes():#t2.words():
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

def compute_num_matching_subgraphs_dp_new(t1, t2, struc_lamb=1, rel_lamb=1, show_DP_table=False):
    """
    Given two dependency trees, this function returns the number of matching subgraphs.

    :param t1: DependencyTree
    :param t2: DependencyTree
    :return: int
    """

    rel = 1
    struc = 1

    rel_table = {} # it's still indexed by the nodes (always the heads tho!)
    struc_table = {}


    # Initialize the DP entries for all words
    for word_t1 in t1.nodes():#t1.words():
        for word_t2 in t2.nodes():#t2.words():
            rel_table[(word_t1.pos, word_t2.pos)] = 0.0
            struc_table[(word_t1, word_t1.pos, word_t2.pos)] = 0.0


    # Run the algorithm from Collins & Duffy (2001) - Technical Report
    for word_t1, arc_t1 in t1:
        for word_t2, arc_t2 in t2:

            if Counter(arc_t1.label) == Counter(arc_t2.label):  # Ordered? - No. Because it is the abstract meaning of this structure that is important

                # adjust the labels to the order of one of them, I think
                rel_factors = [(2 + rel_table[(arc_t1[ind].pos, arc_t2[ind_other].pos)]) for ind, ind_other in zip(range(len(arc_t1)), range(len(arc_t1)))]
                rel_table[(word_t1.pos, word_t2.pos)] = rel_table[(word_t1.pos, word_t2.pos)] + \
                                                                   (rel_lamb * np.prod(rel_factors)) - 1

            if word_t1 == word_t2:

                # Extract the set of common dependencies
                common_dep = arc_t1.compare_arcs_new(arc_t2)

                # If there are none, skip to the next word
                if len(common_dep) == 0:
                    continue

                # Else: Build the factors of the product
                struc_factors = [(2 + struc_table[(child, arc_t1[ind].pos, arc_t2[ind_other].pos)]) for ind, ind_other, child in common_dep]

                # Update the DP table (No short hand notation for autograd)
                struc_table[(word_t1, word_t1.pos, word_t2.pos)] = \
                    struc_table[(word_t1, word_t1.pos, word_t2.pos)] + (struc_lamb * np.prod(struc_factors)) - 1

    if show_DP_table:
        print(rel_table)
        print("Rel")
        print(sum(rel_table.values()))
        print("---")
        print("Struc")
        print(struc_table)
        print(sum(struc_table.values()))

    return rel * sum(rel_table.values()) + struc * sum(struc_table.values())

def compute_num_matching_struc_subgraphs_dp(t1, t2, lamb, show_DP_table=False):
    """
    Given two dependency trees, this function returns the number of matching subgraphs.

    :param t1: DependencyTree
    :param t2: DependencyTree
    :return: int
    """
    struc_table = {}

    # Initialize the DP entries for all words
    for word_t1 in t1.nodes():#t1.words():
        for word_t2 in t2.nodes():#t2.words():
            #rel_table[(word_t1.pos, word_t2.pos)] = 0.0
            struc_table[(word_t1, word_t1.pos, word_t2.pos)] = 0.0

    # Run the algorithm from Collins & Duffy (2001) - Technical Report
    for word_t1, arc_t1 in t1:
        for word_t2, arc_t2 in t2:

            if word_t1 == word_t2:

                # Extract the set of common dependencies
                common_dep = arc_t1.compare_arcs_new(arc_t2)

                # If there are none, skip to the next word
                if len(common_dep) == 0:
                    continue

                # Else: Build the factors of the product
                struc_factors = [(2 + struc_table[(child, arc_t1[ind].pos, arc_t2[ind_other].pos)]) for ind, ind_other, child in common_dep]

                # Update the DP table (No short hand notation for autograd)
                struc_table[(word_t1, word_t1.pos, word_t2.pos)] = \
                    struc_table[(word_t1, word_t1.pos, word_t2.pos)] + (lamb * np.prod(struc_factors)) - 1

    if show_DP_table:
        print(struc_table)
        print(sum(struc_table.values()))

    return sum(struc_table.values())


def compute_num_matching_rel_subgraphs_dp(t1, t2, lamb, show_DP_table=False):
    """
    Given two dependency trees, this function returns the number of matching subgraphs.

    :param t1: DependencyTree
    :param t2: DependencyTree
    :return: int
    """
    rel_table = {}

    # Initialize the DP entries for all words
    for word_t1 in t1.nodes():#t1.words():
        for word_t2 in t2.nodes():#t2.words():
            rel_table[(word_t1.pos, word_t2.pos)] = 0.0

    # Run the algorithm from Collins & Duffy (2001) - Technical Report
    for word_t1, arc_t1 in t1:
        for word_t2, arc_t2 in t2:

            if Counter(arc_t1.label) == Counter(arc_t2.label):  # Ordered? - No. Because it is the abstract meaning of this structure that is important

                # adjust the labels to the order of one of them, I think
                rel_factors = [(2 + rel_table[(arc_t1[ind].pos, arc_t2[ind_other].pos)]) for ind, ind_other in zip(range(len(arc_t1)), range(len(arc_t1)))]
                rel_table[(word_t1.pos, word_t2.pos)] = rel_table[(word_t1.pos, word_t2.pos)] + \
                                                                   (lamb * np.prod(rel_factors)) - 1

    if show_DP_table:
        print(rel_table)
        print(sum(rel_table.values()))

    return sum(rel_table.values())


if __name__ == "__main__":

    from parsing.old_dep_tree_models.helpers.test_helpers import convert_to_custom_data_structure, parse_sentences

    sentence_1 = "Hello my dear friend, Mel ."  # The comma changes the grammatical relationship
    sentence_2 = "Hello my dear friend Ricardo ."

    dep_graph_1, dep_graph_2 = parse_sentences(sent_1=sentence_1, sent_2=sentence_2)

    t1, t2 = convert_to_custom_data_structure(dep_graph=dep_graph_1), convert_to_custom_data_structure(dep_graph=dep_graph_2)

    print(t1)
    print(t2)

    print(compute_num_matching_subgraphs_dp_new(t1=t1, t2=t2, show_DP_table=True))
    print(compute_num_matching_subgraphs_dp(t1=t1, t2=t2, show_DP_table=True))