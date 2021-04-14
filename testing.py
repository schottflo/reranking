import unittest
import time
from itertools import combinations
from collections import defaultdict
from nltk import Tree, ParentedTree
from num_matching_subtrees import compute_num_matching_subtrees_dp
import parse_tree_generator

def _extract_all_subtrees(tree):
    """
    Extract all possible subtrees according to the definition in Collins and Duffy (2001).

    :param tree: nltk.Tree
    :return: list of lists of nltk.Trees
    """
    subtrees = []

    if tree.height() == 2: # Base case: pre-terminals (smallest trees), just add the tree itself
        subtrees.append(tree)
        return subtrees, subtrees

    else:
        # Keep track of additional subtrees per iteration to ensure that larger subtrees are connected
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
        left_add, left_all = _extract_all_subtrees(left_child)
        right_add, right_all = _extract_all_subtrees(right_child)

        # Keep track of all subtrees so far (idea: later merge the additional ones into that list as well)
        subtrees.extend(left_all)
        subtrees.extend(right_all)

        # Consider all possible subtrees on the left child while holding the right child fixed
        for lower_subtree_left in left_add:
            add_subtrees.append(Tree(top_node, [lower_subtree_left, right_node]))

        # Consider all possible subtrees on the right child while holding the left child fixed
        for lower_subtree_right in right_add:
            add_subtrees.append(Tree(top_node, [left_node, lower_subtree_right]))

        # Consider all possible combinations of left and right subtrees (which doesn't include just one node per child
        # which is why we accounted for those cases in the two loops before that)
        for lower_subtree_left in left_add:
            for lower_subtree_right in right_add:
                add_subtrees.append(Tree(top_node, [lower_subtree_left, lower_subtree_right]))

        # Merge additional subtrees into all subtrees list
        subtrees.extend(add_subtrees)

        return add_subtrees, subtrees

def extract_all_subtrees(tree):
    """
    Wrapper function for _extract_all_subtrees that only returns the final subtrees list.

    :param tree: nltk.Tree
    :return: list of nltk.Trees
    """
    return _extract_all_subtrees(tree)[1]

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

    # Bit of code for runtime comparisons (since it is leaner and doesn't keep track which subtrees matched)
    # count = 0
    # for s1 in subtrees1:
    #     for s2 in subtrees2:
    #         if s1 == s2:
    #             count += 1
    # matching_subtrees = []

    count = sum(matching_subtrees.values())
    return count, matching_subtrees

class TestNumSubtrees(unittest.TestCase):

    def random_trees(self):
        t1, t2 = parse_tree_generator.generate_random_parse_trees(num_trees=2)

        start_dp = time.time()
        res_dp = compute_num_matching_subtrees_dp(t1=t1, t2=t2)
        end_dp = time.time()

        start_naive = time.time()
        res_naive, trees_naive = compute_num_matching_subtrees_naive(t1=t1, t2=t2)
        end_naive = time.time()

        print("\n")
        print("--- Expected matching trees ---")
        print(trees_naive, "\n")
        print("Naive time:", end_naive - start_naive)
        print("DP time:", end_dp - start_dp)

        self.assertEqual(res_dp, res_naive, f"Should be {res_naive}")

    def trees_of_same_size(self):
        t1 = ParentedTree.fromstring("(S (NP (D the) (N man)) (VP (V eats) (NP fish)))")
        t2 = ParentedTree.fromstring("(S (NP (D the) (N woman)) (VP (V cooks) (NP meat)))")

        start_naive = time.time()
        res_naive, trees_naive = compute_num_matching_subtrees_naive(t1, t2)
        end_naive = time.time()

        start_dp = time.time()
        res_dp = compute_num_matching_subtrees_dp(t1=t1, t2=t2)
        end_dp = time.time()

        print("\n")
        print("--- Expected matching trees ---")
        print(trees_naive, "\n")
        print("Naive time:", end_naive - start_naive)
        print("DP time:", end_dp - start_dp)

        self.assertEqual(res_dp, res_naive, f"Should be {res_naive}")

    def trees_of_different_size(self):
        t1 = ParentedTree.fromstring("(S (NP (D the) (N man)) (VP (V eats) (NP fish)))")
        t2 = ParentedTree.fromstring(
            "(S (NP (D the) (N woman)) (AdvP (Adv thoughtfully) (VP (V cooks) (NP meat))))")

        start_naive = time.time()
        res_naive, trees_naive = compute_num_matching_subtrees_naive(t1, t2)
        end_naive = time.time()

        start_dp = time.time()
        res_dp = compute_num_matching_subtrees_dp(t1=t1, t2=t2)
        end_dp = time.time()

        print("\n")
        print("--- Expected matching trees ---")
        print(trees_naive, "\n")
        print("Naive time:", end_naive - start_naive)
        print("DP time:", end_dp - start_dp)

        self.assertEqual(res_dp, res_naive, f"Should be {res_naive}")


    def trees_with_duplicate_subtrees(self):
        t1 = ParentedTree.fromstring(
                "(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
        t2 = ParentedTree.fromstring(
            "(S (NP (D the) (N woman)) (AdvP (Adv thoughtfully) (VP (V cooks) (NP meat))))")

        start_naive = time.time()
        res_naive, trees_naive = compute_num_matching_subtrees_naive(t1, t2)
        end_naive = time.time()

        start_dp = time.time()
        res_dp = compute_num_matching_subtrees_dp(t1=t1, t2=t2)
        end_dp = time.time()

        print("\n")
        print("--- Expected matching trees ---")
        print(trees_naive, "\n")
        print("Naive time:", end_naive - start_naive)
        print("DP time:", end_dp - start_dp)

        self.assertEqual(res_dp, res_naive, f"Should be {res_naive}")


    def similar_trees(self):
        t1 = ParentedTree.fromstring(
            "(S (NP (D the) (N dog)) (VP (V chased) (NP (NP (D the) (N cat)) (NP (Conj and) (NP (D the) (N mouse))))))")
        t2 = ParentedTree.fromstring(
            "(S (NP (D the) (N dog)) (VP (V chased) (NP (NP (D the) (N mom)) (NP (Conj and) (NP (D the) (N mouse))))))")

        start_naive = time.time()
        res_naive, trees_naive = compute_num_matching_subtrees_naive(t1, t2)
        end_naive = time.time()

        start_dp = time.time()
        res_dp = compute_num_matching_subtrees_dp(t1=t1, t2=t2)
        end_dp = time.time()

        print("\n")
        print("--- Expected matching trees ---")
        print(trees_naive, "\n")
        print("Naive time:", end_naive - start_naive)
        print("DP time:", end_dp - start_dp)

        self.assertEqual(res_dp, res_naive, f"Should be {res_naive}")


if __name__ == '__main__':
    unittest.main()




