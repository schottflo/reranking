import nltk

"""
Brute force idea for testing (probably put in new file):
Given a tree t, we want to create all possible subtrees of t.

Base case: if t is a pre-terminal, return t

Recursive case: then t has two children, t1 and t2.
subtrees = []
subtrees1 = Find all subtrees of t1
subtrees2 = Find all subtrees of t2
subtrees.extend(subtrees1)
subtrees.extend(subtrees2)
For each s1 of subtrees1:
    subtrees.append(new_tree(t.head, s1, t2.head))
For each s2 of subtrees2:
    subtrees.append(new_tree(t.head, t1.head, s2))
For each s1 of subtrees1:
    For each subtree s2 of subtrees2:
        subtrees.append(new_tree(t.head, t1, t2))
"""


def compute_num_matching_subtrees(t1, t2):
    """
    Given two trees (in CNF) returns the number of matching subtrees.

    To be fixed: Cannot account for duplicate non-terminals (idea: memoize observed subtrees)

    :param t1: nltk.Tree
    :param t2: nltk.Tree
    :return: int
    """
    t1.pretty_print()
    t2.pretty_print()

    # Extract all productions
    prods_1 = t1.productions()
    prods_2 = t2.productions()

    # Initialize the DP table (still a dictionary)
    DP_table = {prod: 0 for prod in set(prods_1 + prods_2)}

    for prod_1 in prods_1:
        for prod_2 in prods_2:
            # TODO (RAN): You can move the prod_1.is_lexical() to before the prod_2 loop to save some time
            if prod_1.is_lexical() and prod_1 == prod_2:
                DP_table[prod_1] = 1

    # Algorithm
    for pos1 in t1.treepositions(order="postorder"):
        subtree1 = t1[pos1]
        if type(subtree1) != str and len(subtree1.productions()) > 1: # first condition excludes terminals, the second pre-terminals
            prod_1 = subtree1.productions()[0]

            for pos2 in t2.treepositions(order="postorder"):
                subtree2 = t2[pos2]
                if type(subtree2) != str and len(subtree2.productions()) > 1:
                    prod_2 = subtree2.productions()[0]

                    if prod_1 == prod_2:

                        # TODO (RAN): What is prev_prod?
                        prev_prod_1_1 = subtree1[(0,)].productions()[0]
                        prev_prod_1_2 = subtree1[(1,)].productions()[0]

                        prev_prod_2_1 = subtree2[(0,)].productions()[0]
                        prev_prod_2_2 = subtree2[(1,)].productions()[0]

                        if prev_prod_1_1 == prev_prod_2_1 and prev_prod_1_2 == prev_prod_2_2:
                            DP_table[prod_1] += (1 + DP_table[prev_prod_1_1]) * (1 + DP_table[prev_prod_1_2])
                            continue

                        if prev_prod_1_1 == prev_prod_2_1:
                            DP_table[prod_1] += (1 + DP_table[prev_prod_1_1])

                        elif prev_prod_1_2 == prev_prod_2_2:
                            DP_table[prod_1] += (1 + DP_table[prev_prod_1_2])

                        else:
                            DP_table[prod_1] += 1

    # Accounting for terminal productions that occur more than once
    # TODO (RAN): You can fit this into the for loop at the start, also, why do we not initialize them with the counts?
    counts = {prod: 0 for prod in set(prods_1 + prods_2)}
    for prod_1 in prods_1:
        for prod_2 in prods_2:
            if prod_1.is_lexical() and prod_1 == prod_2:
                counts[prod_1] += 1

    for key, value in counts.items():
        if value > 1:
            DP_table[key] *= value

    print(DP_table)
    print(sum(DP_table.values()))


if __name__ == "__main__":

    t1 = nltk.ParentedTree.fromstring(
        "(S (NP (D the) (N dog)) (VP (V chased) (NP (NP (D the) (N cat)) (NP (Conj and) (NP (D the) (N mouse))))))")
    t2 = nltk.ParentedTree.fromstring("(S (NP (D the) (N woman)) (AdvP (Adv thoughtfully) (VP (V cooks) (NP meat))))")
    compute_num_matching_subtrees(t1=t1, t2=t2)