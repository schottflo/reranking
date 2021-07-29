from random import sample
from nltk.corpus import treebank


def generate_two_parse_trees():
    """
    Randomly sample two trees from the first 100 sentences of the excerpt of the Penn Treebank integrated into nltk
    that have less than or equal to 15 tokens.

    :return: tuple of nltk.Tree
    """
    # Only work with trees less than 15 tokens
    trees = []
    for tree in treebank.parsed_sents()[:100]:
        if len(tree.leaves()) <= 15:
            trees.append(tree)

    # Sample two trees
    sampled_trees = []
    for sampled_tree in sample(trees, 2):
        sampled_tree.collapse_unary(collapsePOS=True)
        sampled_tree.chomsky_normal_form()
        sampled_trees.append(sampled_tree)

    return sampled_trees

if __name__ == "__main__":

    from random import seed

    seed(42)
    t1, t2 = generate_two_parse_trees()

    t1.pretty_print()
    t2.pretty_print()
