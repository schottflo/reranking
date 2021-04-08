import unittest
import nltk
from collins_duffy import compute_num_matching_subtrees

class TestNumSubtrees(unittest.TestCase):

    def trees_of_same_size(self):
        t1 = nltk.ParentedTree.fromstring("(S (NP (D the) (N man)) (VP (V eats) (NP fish)))")
        t2 = nltk.ParentedTree.fromstring("(S (NP (D the) (N woman)) (VP (V cooks) (NP meat)))")
        self.assertEqual(compute_num_matching_subtrees(t1=t1, t2=t2), 10, "Should be 10")

    def trees_of_different_size(self):
        t1 = nltk.ParentedTree.fromstring("(S (NP (D the) (N man)) (VP (V eats) (NP fish)))")
        t2 = nltk.ParentedTree.fromstring(
            "(S (NP (D the) (N woman)) (AdvP (Adv thoughtfully) (VP (V cooks) (NP meat))))")
        self.assertEqual(compute_num_matching_subtrees(t1=t1, t2=t2), 4, "Should be 4")

    def trees_with_duplicate_subtrees(self):
        t1 = nltk.ParentedTree.fromstring(
                "(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
        t2 = nltk.ParentedTree.fromstring(
            "(S (NP (D the) (N woman)) (AdvP (Adv thoughtfully) (VP (V cooks) (NP meat))))")
        self.assertEqual(compute_num_matching_subtrees(t1=t1, t2=t2), 7, "Should be 7")

    def trees_with_duplicate_structures(self):
        t1 = nltk.ParentedTree.fromstring(
            "(S (NP (D the) (N dog)) (VP (V chased) (NP (NP (D the) (N cat)) (NP (Conj and) (NP (D the) (N mouse))))))")
        t2 = nltk.ParentedTree.fromstring(
            "(S (NP (D the) (N woman)) (VP (V cooks) (NP (Adj nice) (NP meat))))")
        self.assertEqual(compute_num_matching_subtrees(t1=t1, t2=t2), 13, "Should be 14")

if __name__ == '__main__':
    unittest.main()
    unittest.trees_of_same_size()




