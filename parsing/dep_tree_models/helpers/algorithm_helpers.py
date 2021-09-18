from itertools import combinations, chain
from nltk import Tree


def _address_tree(dep_graph, i):
    """
    Adapted version of the method "_tree" of a NLTK DependencyGraph to create a nltk.Tree with the node addresses

    :param dep_graph: nltk.DependencyGraph
    :param int: int
    :return: nltk.Tree
    """
    node = dep_graph.get_by_address(i)
    address = str(node["address"])
    deps = sorted(chain.from_iterable(node["deps"].values()))

    if deps:
        return Tree(address, [_address_tree(dep_graph=dep_graph, i=dep) for dep in deps])
    else:
        return address


def address_tree(dep_graph):
    """
    Adapted version of the method "tree" of a NLTK DependencyGraph to create a nltk.Tree with the node addresses
    starting with the ``root`` node.

    :param dep_graph: nltk.DependencyGraph
    :return: nltk.Tree
    """
    node = dep_graph.root

    address = str(node["address"])
    deps = sorted(chain.from_iterable(node["deps"].values()))

    return Tree(str(0), [Tree(address, [_address_tree(dep_graph, dep) for dep in deps])])#Tree(address, [_address_tree(dep_graph, dep) for dep in deps])


def build_pos_to_address(dep_graph, dep_tree):
    """
    Build a dictionary mapping from nltk.Tree positions to addresses (i.e. indices of the words in the sentence plus 1).
    This is done by constructing a nltk.Tree with all the addresses that is of equal shape as the dependency tree.

    :param dep_graph: nltk.DependencyGraph
    :param dep_tree: nltk.Tree
    :return: dict
    """
    ad_tree = address_tree(dep_graph=dep_graph)

    pos_to_address = {}

    for pos in dep_tree.treepositions():

        address_subtree = ad_tree[pos]
        if isinstance(address_subtree, str):
            pos_to_address[pos] = int(address_subtree)
        else:
            pos_to_address[pos] = int(address_subtree.productions()[0].lhs()._symbol)

    return pos_to_address


def extract_top_combinations(node, children, pos_to_address):
    """
    Extract all possible combinations at top level of the tree (i.e. each two-word subgraph AND all possible combinations)

    :return: list of nltk.Trees
    """
    subgraphs = []

    for i in range(len(children)):
        for comb in combinations([str(child) + f":/{pos_to_address[(j,)]}" for j, child in enumerate(children)], i+1):
            subgraphs.append(Tree(str(node) + f":/{pos_to_address[()]}", list(comb)))

    return subgraphs


def adjust_dict(i, pos_to_address):
    """
    Adjust the pos_to_address dictionary based on the level of the subtree

    :param pos_to_address: dict
    :return: dict
    """
    adj_pos_to_address = {}
    for key, val in pos_to_address.items():
        if len(key) > 0 and str(i) in str(key[0]):
            adj_pos_to_address[key[1:]] = val

    return adj_pos_to_address


def _extract_all_subgraphs(dep_tree, pos_to_address):
    """

    :param dep_tree:
    :param pos_to_address:
    :return:
    """
    # Extract the upper arc of the dependency tree
    arc = dep_tree.productions()[0]

    node = arc.lhs()
    children = arc.rhs()

    # extract all possible combinations at this level
    subgraphs = extract_top_combinations(node=node, children=children, pos_to_address=pos_to_address)

    for i in range(len(children)):

        # Extract the nltk position
        ind = (i,)

        # Extract the subtree
        subtree = dep_tree[ind]

        if not isinstance(subtree, str):  # If the child is not terminal, need to go deeper recursively

            desc_node = str(children[i]) + f":/{pos_to_address[ind]}"

            # Depending on where we are give the function a modified version of the dictionary
            adj_pos_to_address = adjust_dict(i=i, pos_to_address=pos_to_address)

            # Extract the subgraphs starting from that nonterminal node
            add_subgraphs = _extract_all_subgraphs(dep_tree=subtree, pos_to_address=adj_pos_to_address)

            # Search through the current add_subgraphs to determine which trees end with that particular word
            comb_graphs = []
            for subgraph in subgraphs:

                # These nodes are the ones that could potentially be extended
                current_children = [str(child) for child in subgraph.productions()[0].rhs()]

                if desc_node in current_children:

                    # Go through all additional subgraphs which start with desc_node
                    add_subgraphs_filt = [add_subgraph for add_subgraph in add_subgraphs if add_subgraph.label() == desc_node]
                    for add_subgraph_filt in add_subgraphs_filt:
                        # Create new tree replacing the desc_node with the new graph
                        comb_graph = Tree(str(node)+ f":/{pos_to_address[()]}",
                                          [child_node if desc_node not in str(child_node) else add_subgraph_filt for
                                           child_node in subgraph])

                        # Append the combined graph
                        comb_graphs.append(comb_graph)

            # Add the additional subgraphs from the level beneath and the resulting combinations to the total subgraphs list
            subgraphs.extend(add_subgraphs)
            subgraphs.extend(comb_graphs)

    return subgraphs


def extract_all_subgraphs(dep_graph):
    """
    Extract all subgraphs from a dependency tree based on the definition in the technical report of Collins & Duffy (2001).

    :param dep_graph: nltk.DependenyGraph
    :return: list of nltk.Tree
    """
    #tree = dep_graph.tree()
    tree = Tree("ROOT", [dep_graph.tree()])


    # Build global dictionary mapping from nltk positions in the tree to addresses
    pos_to_address = build_pos_to_address(dep_graph=dep_graph, dep_tree=tree)

    return _extract_all_subgraphs(dep_tree=tree, pos_to_address=pos_to_address)


def new_triples(dep_graph, node=None):
    """
    Adapted version of the method triples of NLTK DependencyGraphs

    :param dep_graph: nltk.DependencyGraph
    :param node: dict
    :return: 3-tuple
    """
    if not node:
        node = dep_graph.root

    head = (node["word"], node["address"])
    for i in sorted(chain.from_iterable(node["deps"].values())):
        dep = dep_graph.get_by_address(i)
        yield (head, dep["rel"], (dep["word"], dep["address"]))

        for triple in new_triples(dep_graph=dep_graph, node=dep):
            yield triple


def build_dep_dict(dep_graph):
    """
    Build a dictionary mapping arcs to dependency labels.

    :param dep_graph: nltk.DependencyGraph
    :return: dictionary
    """
    arc_to_dep = {}
    for triple in new_triples(dep_graph=dep_graph):
        head, dep_rel, child = triple
        arc_to_dep[(head, child)] = dep_rel

    # Also append the root relation
    node = dep_graph.root
    arc_to_dep[(('ROOT', 0), (node["word"], node["address"]))] = 'root'

    return arc_to_dep


def _extract_edges(arc, split_symbol):
    """
    Extract all edges from a given arc of a dependency graph.

    :param arc: nltk.Production
    :param split_symbol: str
    :return: list of tuples
    """
    arc_edges = []

    head, pos_h = arc.lhs()._symbol.split(sep=split_symbol)

    for child_i in arc.rhs():
        if not isinstance(child_i, str):
            child_i = child_i._symbol
        child, pos_c = child_i.split(sep=split_symbol)
        arc_edges.append(((head, int(pos_h)), (child, int(pos_c))))

    return arc_edges


def extract_edges(subgraph, split_symbol=":/"):
    """
    Extract all edges from a subgraph.

    :param subgraph: nltk.Tree
    :param split_symbol: str
    :return: list of tuples
    """
    edges = []
    for arc in subgraph.productions():
        edges.extend(_extract_edges(arc=arc, split_symbol=split_symbol))

    return edges


def compare_labels(edges1, edges2, arc_to_dep1, arc_to_dep2):
    """
    Compare whether two subgraphs are equivalent.

    :param subgraph1: list of tuples
    :param subgraph2: list of tuples
    :param arc_to_dep: dict
    :return: bool
    """
    if len(edges1) != len(edges2):
        return False

    for edge1, edge2 in zip(edges1, edges2):
        if edge1[0][0] != edge2[0][0] or edge1[1][0] != edge2[1][0] or arc_to_dep1[edge1] != arc_to_dep2[edge2]: # Same head, same child, same label
            return False

    return True