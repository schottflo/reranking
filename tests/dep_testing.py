import time, unittest

from parsing.algorithms.cd_dep_brute_force import compute_num_matching_subgraphs_naive
from parsing.algorithms.cd_dep_dynamic_program import compute_num_matching_subgraphs_dp

from parsing.data_structures.dep_tree import Node, Arc, DependencyTree
from parsing.dep_tree_models.helpers.test_helpers import convert_to_custom_data_structure, parse_sentences


def test_two_dependency_trees(t1, t2):
    """
    Compute the number of matching subgraphs with the dynamic program proposed by Collins & Duffy (2001) and the brute
    force method and compare the time.

    :param t1: nltk.DependencyGraph
    :param t2: nltk.DependencyGraph
    :return: tuple of int
    """
    ## Dynamic Program
    # Convert into the appropriate data structure
    t1_custom, t2_custom = convert_to_custom_data_structure(dep_graph=t1), convert_to_custom_data_structure(dep_graph=t2)

    print("\n---Matching trees---")
    start_dp = time.time()
    res_dp = compute_num_matching_subgraphs_dp(t1=t1_custom, t2=t2_custom)
    end_dp = time.time()

    print(res_dp)

    ## Naive Computation (incl. data structure conversion)
    print("\n--- Expected matching trees ---")
    start_naive = time.time()
    res_naive = compute_num_matching_subgraphs_naive(t1=t1, t2=t2)
    end_naive = time.time()

    print("\n")
    print("Number of trees:", res_dp)
    print("Naive time:", end_naive - start_naive)
    print("DP time:", end_dp - start_dp)

    return res_dp, res_naive


class TestNumSubgraphs(unittest.TestCase):

    def two_trees(self):
        """
        Idea: Hand-write two similar sentences and parse them and then compare the number of subgraphs found by
        the dynamic program and the naive approach. If one would randomly sample sentences, they would most likely
        have no common subgraphs.
        """
        # sentence_1 = "Barack Obama and his two sisters played in Hawaii."
        # sentence_2 = "Barack O'Reilly and his four crazy sisters played in Hawaii."

        #sentence_1 = "Hello my dear friend Mel ."
        #sentence_2 = "Hello my dear friend Ricardo ."

        #sentence_1 = "My girlfriend is swimming in the river, while I do other sports ."
        #sentence_2 = "My girlfriend is swimming in the lake, while I do Parkour ."

        # 1
        # sentence_1 = "Lightning Paradise was the local hangout joint where the group usually ended up spending the night."
        # sentence_2 = "Lightning Paradise was the local hangout joint where the crew usually ended up spending the night."

        # # 2
        # sentence_1 = "After fighting off the alligator, Brian still had to face the anaconda."
        # sentence_2 = "Once he faught off the alligator, Brian still had to face the anaconda."

        # 3
        sentence_1 = "The truth is that you pay for your lifestyle in hours."
        sentence_2 = "The truth is that you pay for your lifestyle in minutes."

        # # 4
        # sentence_1 = "The sun had set and so had his dreams."
        # sentence_2 = "The sun had not set and so had his dreams."

        # # 5
        # sentence_1 = "If you like tuna and tomato sauce- try combining the two."
        # sentence_2 = "If you like tuna and tomato sauce- why not combining the two."

        dep_graph_1, dep_graph_2 = parse_sentences(sent_1=sentence_1, sent_2=sentence_2)

        res_dp, res_naive = test_two_dependency_trees(t1=dep_graph_1, t2=dep_graph_2)

        self.assertEqual(res_dp, res_naive, f"Should be {res_naive}")

if __name__ == '__main__':
    unittest.main()



