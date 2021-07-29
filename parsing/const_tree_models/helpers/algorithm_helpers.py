from nltk import Tree


def _extract_all_subtrees(tree):
    """
    Extract all possible subtrees according to the definition in Collins and Duffy (2001).

    :param tree: nltk.Tree
    :return: tuple of lists of nltk.Trees
    """
    subtrees = []

    if tree.height() == 2:  # Base case: pre-terminals (smallest trees), just add the tree itself
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
