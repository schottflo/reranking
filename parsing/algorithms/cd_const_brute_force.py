from collections import defaultdict
from parsing.const_tree_models.helpers.algorithm_helpers import extract_all_subtrees

def compute_num_matching_subtrees_naive(t1, t2):
    """
    Return the number of matching subtrees and the matching subtrees themselves by enumerating all subtrees and
    comparing them one by one.

    :param subtrees1: nltk.Tree
    :param subtrees2: nltk.Tree
    :return: int
    """
    subtrees1 = extract_all_subtrees(t1)
    subtrees2 = extract_all_subtrees(t2)

    matching_subtrees = defaultdict(int)
    for s1 in subtrees1:
        for s2 in subtrees2:
            if s1 == s2:
                matching_subtrees[str(s1)] += 1

    count = sum(matching_subtrees.values())

    # Alternative bit of code for runtime comparisons (since it doesn't keep track which subtrees matched)
    # count = 0
    # for s1 in subtrees1:
    #     for s2 in subtrees2:
    #         if s1 == s2:
    #             count += 1

    print(matching_subtrees)
    return count

if __name__ == "__main__":

    from nltk.tree import ParentedTree

    t1 = ParentedTree.fromstring("(S (NP (Conj but) (NP (D the) (N house))) (VP (V wounded) (NP (Conj and) (NP (D the) (NP (N flower) (NP (Conj and) (NP (mod both) (N head))))))))")
    t2 = ParentedTree.fromstring("(S (NP (D the) (N contempt)) (VP (V puckered) (NP (AdjP (D a) (Adj hairy)) (PP (IN that) (N envoy)))))")
    print(compute_num_matching_subtrees_naive(t1, t2))