from nltk import ParentedTree

def compute_num_matching_subtrees_dp(t1, t2):
    """
    Given two trees (in CNF) returns the number of matching subtrees.

    To be fixed: Cannot account for duplicate non-terminals (idea: memoize observed subtrees)

    :param t1: nltk.Tree
    :param t2: nltk.Tree
    :return: int
    """
    t1.pretty_print()
    t2.pretty_print()

    DP_table = {}
    for pos in t1.treepositions(order="postorder"):
        tr = t1[pos]
        node_pos = pos
        if type(tr) != str:  # and len(tr.productions()) > 1
            node = tr.productions()[0].lhs()
        else:
            node = tr
        DP_table[(node, node_pos)] = 0

    for pos1 in t1.treepositions(order="postorder"):
        subtree1 = t1[pos1]
        if type(subtree1) != str:
            node1 = subtree1.productions()[0].lhs()
            prod1 = subtree1.productions()[0]

            for pos2 in t2.treepositions(order="postorder"):
                subtree2 = t2[pos2]
                if type(subtree2) != str:
                    node2 = subtree2.productions()[0].lhs()
                    prod2 = subtree2.productions()[0]

                    if node1 == node2 and prod1 == prod2:
                        child_1 = prod1.rhs()[0]

                        pos_child_1 = pos1 + (0,)
                        pos_child_2 = pos1 + (1,)
                        try:
                            child_2 = prod1.rhs()[1]
                            DP_table[(node1, pos1)] += (1 + DP_table[(child_1, pos_child_1)]) * (
                                    1 + DP_table[(child_2, pos_child_2)])
                        except:
                            DP_table[(node1, pos1)] += (1 + DP_table[(child_1, pos_child_1)])


    print(DP_table)
    return sum(DP_table.values())

if __name__ == "__main__":
    t1 = ParentedTree.fromstring(
        "(S (NP (D the) (N dog)) (VP (V chased) (NP (NP (D the) (N cat)) (NP (Conj and) (NP (D the) (N mouse))))))")
    t2 = ParentedTree.fromstring(
        "(S (NP (D the) (N woman)) (VP (V cooks) (NP (Adj nice) (NP meat))))")
    print(compute_num_matching_subtrees_dp(t1=t1, t2=t2))