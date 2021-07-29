import time, unittest
from nltk import ParentedTree

from parsing.algorithms.cd_const_brute_force import compute_num_matching_subtrees_naive
from parsing.algorithms.cd_const_dynamic_program import compute_num_matching_subtrees_dp

from parsing.const_tree_models.data_generator import transform_nltk_tree


def test_two_trees(t1, t2):
    """
    Compute the number of matching subtrees with the dynamic program proposed by Collins & Duffy (2001) and the brute
    force method and compare the time.

    :param t1: nltk.Tree
    :param t2: nltk.Tree
    :return: tuple of int
    """
    # Compute number of matching trees with the dynamic program and measure time
    t1_custom, t2_custom = transform_nltk_tree(tree=t1), transform_nltk_tree(tree=t2)

    print("\n---Matching trees---")
    start_dp = time.time()
    res_dp = compute_num_matching_subtrees_dp(t1=t1_custom, t2=t2_custom, show_DP_table=True)
    end_dp = time.time()

    # Compute number of matching trees naively and measure time
    print("\n--- Expected matching trees ---")
    start_naive = time.time()
    res_naive = compute_num_matching_subtrees_naive(t1=t1, t2=t2)
    end_naive = time.time()

    print("\n")
    print("Number of trees:", res_dp)
    print("Naive time:", end_naive - start_naive)
    print("DP time:", end_dp - start_dp)

    return res_dp, res_naive


class TestNumSubtrees(unittest.TestCase):

    def random_trees(self):

        from parsing.const_tree_models.helpers.testing_helpers import generate_two_parse_trees

        t1, t2 = generate_two_parse_trees()
        res_dp, res_naive = test_two_trees(t1=t1, t2=t2)

        self.assertEqual(res_dp, res_naive, f"Should be {res_naive}")

    def trees_of_same_size(self):
        t1 = ParentedTree.fromstring("(S (NP (D the) (N man)) (VP (V eats) (NP fish)))")
        t2 = ParentedTree.fromstring("(S (NP (D the) (N woman)) (VP (V cooks) (NP meat)))")
        res_dp, res_naive = test_two_trees(t1=t1, t2=t2)

        self.assertEqual(res_dp, res_naive, f"Should be {res_naive}")

    def trees_of_different_size(self):
        t1 = ParentedTree.fromstring("(S (NP (D the) (N man)) (VP (V eats) (NP fish)))")
        t2 = ParentedTree.fromstring(
            "(S (NP (D the) (N woman)) (AdvP (Adv thoughtfully) (VP (V cooks) (NP meat))))")
        res_dp, res_naive = test_two_trees(t1=t1, t2=t2)

        self.assertEqual(res_dp, res_naive, f"Should be {res_naive}")

    def trees_with_duplicate_subtrees(self):
        t1 = ParentedTree.fromstring(
                "(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
        t2 = ParentedTree.fromstring(
            "(S (NP (D the) (N woman)) (AdvP (Adv thoughtfully) (VP (V cooks) (NP meat))))")
        res_dp, res_naive = test_two_trees(t1=t1, t2=t2)

        self.assertEqual(res_dp, res_naive, f"Should be {res_naive}")

    def similar_trees(self):
        t1 = ParentedTree.fromstring(
            "(S (NP (D the) (N dog)) (VP (V chased) (NP (NP (D the) (N cat)) (NP (Conj and) (NP (D the) (N mouse))))))")
        t2 = ParentedTree.fromstring(
            "(S (NP (D the) (N dog)) (VP (V chased) (NP (NP (D the) (N mom)) (NP (Conj and) (NP (D the) (N mouse))))))")
        res_dp, res_naive = test_two_trees(t1=t1, t2=t2)

        self.assertEqual(res_dp, res_naive, f"Should be {res_naive}")

if __name__ == '__main__':
    unittest.main()




