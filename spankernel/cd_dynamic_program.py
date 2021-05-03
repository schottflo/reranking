from nltk import ParentedTree

def compute_num_matching_subtrees_dp(t1, t2, show_DP_table=False):
    """
    Given two trees (in CNF), this function returns the number of matching subtrees.

    :param t1: nltk.Tree
    :param t2: nltk.Tree
    :return: int
    """
    # Initialize the DP table: it needs specification by the position of the match in each tree (ow: you overcount)
    # General remark on the DP_table: Always need to cast the first element of the tuple key as a str

    DP_table = {}

    for pos_t1 in t1.treepositions(order="postorder"): # "postorder" means bottom-up
        subtree = t1[pos_t1]
        if isinstance(subtree,str): # Case 1: Terminals (since they are directly returned as strings)
            node = subtree # need to initialize the terminals as well (because they are referenced later)
        else: # Case 2: Any subtree consisting at least of one production
            node = str(subtree.label()) # label extracts root node of the tree
        for pos_t2 in t2.treepositions(order="postorder"):
            DP_table[(node, pos_t1, pos_t2)] = 0

    # General idea: For every subtree rooted at node_t1 in t1 we look for a match in t2 that is rooted at node_t2
    for pos_t1 in t1.treepositions(order="postorder"):
        subtree_t1 = t1[pos_t1]
        if not isinstance(subtree_t1, str): # we only have to go through actual subtrees (not the terminals)
            node_t1 = str(subtree_t1.label())
            prods_t1 = subtree_t1.productions()
            prod_t1 = prods_t1[0] # get the production from the root node of the subtree

            for pos_t2 in t2.treepositions(order="postorder"):
                subtree_t2 = t2[pos_t2]
                if not isinstance(subtree_t2, str):
                    node_t2 = str(subtree_t2.label())
                    prod_t2 = subtree_t2.productions()[0]

                    if node_t1 == node_t2 and prod_t1 == prod_t2: # condition from Collins & Duffy (2001)
                        # If nodes and productions match, we update the DP table

                        # Need DP_table values from the child node(s) to update the DP_table for the matching subtrees
                        # Those DP_table values are also specified by the position of the match in each tree
                        child_1 = str(prod_t1.rhs()[0])
                        pos_child_1_t1 = pos_t1 + (0,)
                        pos_child_1_t2 = pos_t2 + (0,)

                        DP_table_val_1 = DP_table[(child_1, pos_child_1_t1, pos_child_1_t2)]

                        if len(prods_t1) > 1: # condition checks if we work with a pre-terminal tree or not
                            # If yes, we also have to look at the second child and infer its position

                            child_2 = str(prod_t1.rhs()[1])
                            pos_child_2_t1 = pos_t1 + (1,)
                            pos_child_2_t2 = pos_t2 + (1,)

                            DP_table_val_2 = DP_table[(child_2, pos_child_2_t1, pos_child_2_t2)]

                            # Formula by Collins & Duffy (2001)
                            DP_table[(node_t1, pos_t1, pos_t2)] += (1 + DP_table_val_1) * (1 + DP_table_val_2)

                        else:
                            # Formula by Collins & Duffy (2001)
                            DP_table[(node_t1, pos_t1, pos_t2)] += (1 + DP_table_val_1)

    if show_DP_table:
        print(DP_table)

    return sum(DP_table.values())

if __name__ == "__main__":
    t1 = ParentedTree.fromstring(
        "(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
    t2 = ParentedTree.fromstring(
        "(S (NP (D the) (N dog)) (VP (V insulted) (NP (D the) (N cat))))")
    print(compute_num_matching_subtrees_dp(t1=t1, t2=t2))