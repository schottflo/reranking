import unittest
from nltk import Tree, ParentedTree
from num_matching_subtrees import compute_num_matching_subtrees_dp

def extract_all_subtrees(tree):
    """
    Extract all possible subtrees according to the definition in Collins and Duffy (2001).

    :param tree: nltk.Tree
    :return: list of lists of nltk.Trees
    """
    subtrees = []

    if tree.height() == 2:
        subtrees.append(tree)
        return subtrees, subtrees

    else:
        add_subtrees = []

        # Get trees rooted at children
        left_child = tree[(0,)]
        right_child = tree[(1,)]

        # Get nodes
        top_node = tree.label()
        left_node = left_child.label()
        right_node = right_child.label()

        # Get upper production (and transform into tree)
        upper = Tree(top_node, [left_node, right_node])
        add_subtrees.append(upper)

        # Recursive extraction
        left_add, left_all = extract_all_subtrees(left_child)
        right_add, right_all = extract_all_subtrees(right_child)

        subtrees.extend(left_all)
        subtrees.extend(right_all)

        for lower_subtree_left in left_add:
            add_subtrees.append(Tree(top_node, [lower_subtree_left, right_node]))

        for lower_subtree_right in right_add:
            add_subtrees.append(Tree(top_node, [left_node, lower_subtree_right]))

        for lower_subtree_left in left_add:
            for lower_subtree_right in right_add:
                add_subtrees.append(Tree(top_node, [lower_subtree_left, lower_subtree_right]))

        subtrees.extend(add_subtrees)

        return add_subtrees, subtrees

def compute_num_matching_subtrees_naive(t1, t2):
    """
    Return the number of matching subtrees and the matching subtrees themselves by enumerating all subtrees and
    comparing them one by one.

    :param subtrees1: nltk.Tree
    :param subtrees2: nltk.Tree
    :return: int
    """
    _, subtrees1 = extract_all_subtrees(t1)
    _, subtrees2 = extract_all_subtrees(t2)

    count = 0
    matching_subtrees = []
    for s1 in subtrees1:
        for s2 in subtrees2:
            if s1 == s2:
                count += 1
                matching_subtrees.append(s1)

    return count, matching_subtrees

class TestNumSubtrees(unittest.TestCase):

    def trees_of_same_size(self):
        t1 = ParentedTree.fromstring("(S (NP (D the) (N man)) (VP (V eats) (NP fish)))")
        t2 = ParentedTree.fromstring("(S (NP (D the) (N woman)) (VP (V cooks) (NP meat)))")
        res, trees = compute_num_matching_subtrees_naive(t1, t2)
        self.assertEqual(compute_num_matching_subtrees_dp(t1=t1, t2=t2), res, f"Should be {res}")
        print("\n")
        print("--- Expected matching trees ---")
        print(trees)

    def trees_of_different_size(self):
        t1 = ParentedTree.fromstring("(S (NP (D the) (N man)) (VP (V eats) (NP fish)))")
        t2 = ParentedTree.fromstring(
            "(S (NP (D the) (N woman)) (AdvP (Adv thoughtfully) (VP (V cooks) (NP meat))))")
        res, trees = compute_num_matching_subtrees_naive(t1, t2)
        self.assertEqual(compute_num_matching_subtrees_dp(t1=t1, t2=t2), res, f"Should be {res}")
        print("\n")
        print("--- Expected matching trees ---")
        print(trees)

    def trees_with_duplicate_subtrees(self):
        t1 = ParentedTree.fromstring(
                "(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
        t2 = ParentedTree.fromstring(
            "(S (NP (D the) (N woman)) (AdvP (Adv thoughtfully) (VP (V cooks) (NP meat))))")
        res, trees = compute_num_matching_subtrees_naive(t1, t2)
        self.assertEqual(compute_num_matching_subtrees_dp(t1=t1, t2=t2), res, f"Should be {res}")
        print("\n")
        print("--- Expected matching trees ---")
        print(trees)

    def trees_with_duplicate_structures(self):
        t1 = ParentedTree.fromstring(
            "(S (NP (D the) (N dog)) (VP (V chased) (NP (NP (D the) (N cat)) (NP (Conj and) (NP (D the) (N mouse))))))")
        t2 = ParentedTree.fromstring(
            "(S (NP (D the) (N woman)) (VP (V cooks) (NP (Adj nice) (NP meat))))")
        res, trees = compute_num_matching_subtrees_naive(t1, t2)
        self.assertEqual(compute_num_matching_subtrees_dp(t1=t1, t2=t2), res, f"Should be {res}")
        print("\n")
        print("--- Expected matching trees ---")
        print(trees)

if __name__ == '__main__':
    unittest.main()




