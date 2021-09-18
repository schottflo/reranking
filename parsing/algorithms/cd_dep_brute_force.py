from collections import defaultdict
from parsing.new_dep_tree_models.helpers.algorithm_helpers import extract_all_subgraphs, build_dep_dict, extract_edges, compare_labels


def compute_num_matching_subgraphs_naive(t1, t2):
    """
    Return the number of matching subgraphs and the matching subgraphs themselves by enumerating all subgraphs and
    comparing them one by one.

    :param subtrees1: nltk.DependencyGraph
    :param subtrees2: nltk.DependencyGraph
    :return: int
    """
    subgraphs1 = extract_all_subgraphs(dep_graph=t1)
    subgraphs2 = extract_all_subgraphs(dep_graph=t2)

    arc_to_dep1 = build_dep_dict(dep_graph=t1)
    arc_to_dep2 = build_dep_dict(dep_graph=t2)

    matching_subtrees = defaultdict(int)
    for s1 in subgraphs1:
        edges1 = extract_edges(subgraph=s1)
        for s2 in subgraphs2:
            edges2 = extract_edges(subgraph=s2)
            if compare_labels(edges1, edges2, arc_to_dep1, arc_to_dep2): # s1 == s2 not possible bec. addresses included in node names
                matching_subtrees[str(s1)] += 1

    print(matching_subtrees)

    return sum(matching_subtrees.values())


if __name__ == "__main__":

    from parsing.old_dep_tree_models.helpers.test_helpers import parse_sentences

    sentence_1 = "Hello my dear friend, Mel ." # The comma changes the grammatical relationship
    sentence_2 = "Hello my dear friend Ricardo ."

    dep_graph_1, dep_graph_2 = parse_sentences(sent_1=sentence_1, sent_2=sentence_2)

    # Less subgraphs than one might think, because the labels (i.e. grammatical relations) can be different.
    # Only solution would be new pretty_print() method, which is not the priority for now

    print(compute_num_matching_subgraphs_naive(t1=dep_graph_1, t2=dep_graph_2))