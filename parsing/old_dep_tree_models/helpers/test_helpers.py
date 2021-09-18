from stanza import Pipeline
from nltk.parse.dependencygraph import DependencyGraph
from nltk.tree import Tree

from parsing.data_structures.dep_tree import Node, Arc, DependencyTree
from parsing.old_dep_tree_models.helpers.algorithm_helpers import build_dep_dict, build_pos_to_address


def convert_to_custom_data_structure(dep_graph, conllu_str=None):
    """
    Convert the nltk.DependencyGraph to a (custom) DependencyTree

    :param dep_graph: nltk.DependencyGraph
    :return:
    """
    # dep_tree = dep_graph.tree()

    dep_tree = Tree("ROOT", [dep_graph.tree()])

    # Build dictionary mapping arcs to labels
    arc_to_dep = build_dep_dict(dep_graph=dep_graph)

    # Build dictionary mapping nltk positions to sentences addresses (i.e. indices + 1)
    pos_to_address = build_pos_to_address(dep_graph=dep_graph, dep_tree=dep_tree)

    arcs = []
    for pos in dep_tree.treepositions(order="postorder"):
        subtree = dep_tree[pos]
        if not isinstance(subtree, str):
            prod = subtree.productions()[0]
            children = prod.rhs()  # children

            new_head = (str(prod.lhs()), pos_to_address[pos])

            new_arc = Arc(head=Node(symbol=new_head[0], pos=new_head[1]),
                          label=[arc_to_dep[(new_head, (str(symb), pos_to_address[pos + (ind,)]))] for ind, symb in
                                 enumerate(children)],
                          tail=[Node(symbol=str(symb), pos=pos_to_address[pos + (ind,)]) for ind, symb in
                                enumerate(children)])

            arcs.append(new_arc)

    return DependencyTree(arcs, conllu_str)


def parse_stanza_output(sent):
    """
    Bring a stanza.Sentence into the CoNLL format accepted by nltk.DependencyGraphs

    :param sent: stanza.Sentence
    :return: str
    """
    return '\n'.join([f'{word.id}\t{word.text}\t{word.lemma}\t{word.upos}\t{word.xpos}\t{word.feats}\t{word.head}\t{word.deprel}\t_\t_' for word in sent.words])


def parse_sentences(sent_1, sent_2):
    """
    Create two DependencyGraphs using Stanza given the input sentences.

    :param sent_1: str
    :param sent_2: str
    :return: list of (two) DependencyGraphs
    """
    nlp = Pipeline('en')
    documents = [nlp(sent_1), nlp(sent_2)]

    parsed_sents = [DependencyGraph(parse_stanza_output(sent), top_relation_label="root") for doc in documents for sent in doc.sentences]

    for parsed_sent in parsed_sents:
        Tree("ROOT", [parsed_sent.tree()]).pretty_print()

    return parsed_sents