import numpy as np

from nltk.corpus import treebank
from nltk import Nonterminal, ChartParser
from nltk import induce_pcfg
from nltk.tree import Tree
from nltk.parse import pchart

from sentence_transformers import SentenceTransformer

from tree import Node, Production, ConstituencyTree

def build_grammar_from_treebank():
    """
    Builds a PCFG based on the files in the NLTK treebank

    :return: nltk.grammar
    """
    sentences = []
    gold_standards = []
    productions = []

    for item in treebank.fileids():

        tree_sent_pairs = zip(treebank.sents(item), treebank.parsed_sents(item))

        for sent, tree in tree_sent_pairs:  # iterate through all sentences and trees from each file

            if len(sent) < 15:  # Only take sentences of length < 15 to ensure different parses

                # Keep track of sentence
                sentences.append(sent)

                # Bring trees in CNF
                tree.collapse_unary(collapsePOS=True)  # Remove branches A-B-C into A-B+C
                tree.chomsky_normal_form()  # Remove A->(B,C,D) into A->B,C+D->D

                # Keep track of true tree
                gold_standards.append(tree)

                # Keep track of productions to produce the PCFG
                productions += tree.productions()

    S = Nonterminal("S")
    grammar = induce_pcfg(S, productions)

    return grammar, gold_standards, sentences


def generate_parse_trees_and_embeddings():
    """
    Put the gold standard parse trees in an appropriate form and generate candidate parse trees and embeddings

    :return: 3-tuple of np.arrays
    """
    # Set up the grammar
    grammar, gold_standards, sentences = build_grammar_from_treebank()

    # Set up fastest parser in NLTK (= beam search)
    parser = pchart.InsideChartParser(grammar, beam_size=2000)

    # Set up sentence embedder
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')

    # Number of candidate trees
    c = 5

    # Initialize the lists
    num_sentences = len(sentences)

    true_parses = [] # it can't be known in advance how many true_parses there will be (bec. only the ones with at least 5 cand parses are considered)
    parses = []
    embeddings = []

    for sent_ind in range(num_sentences):

        true_parse = gold_standards[sent_ind]
        cand_parses = []

        count = 0

        # Take the c worst parses
        for ind, parse in enumerate(reversed(list(parser.parse(sentences[sent_ind])))):  # Reversing the list very expensive, maybe can pop
            if count > (c - 1):
                break
            if parse != true_parse:
                cand_parses.append(Tree.convert(tree=parse)) # Convert probabilistic tree into tree
                count += 1

        if len(cand_parses) == c:  # Checks if there are 5 candidate parses (could be more flexible maybe)
            # TO-DO: make the following cells flexible enough to support different lengths of cands

            # Print the current sentence that is actually included into the dataset
            print(sentences[sent_ind])

            # True parses and embeddings
            true_parses.append(true_parse) #true_parses.append(true_parse)
            embeddings.append(model.encode("".join(sentences[sent_ind])))

            # Candidate parses for the given observation
            parses.append(cand_parses)

    return true_parses, parses, embeddings


def set_up_pos_to_span_dict(tree):
    """
    Return a dictionary that maps the nltk positions of each node to the corresponding span

    :param tree: nltk.Tree
    :return: dict
    """
    # Set up position to span dictionary
    pos_to_span = {}

    # Initialize with pre terminals
    for ind in range(len(tree.leaves())):
        pos = tree.leaf_treeposition(ind)
        pos_to_span[pos] = None # Terminals don't have a span
        pos_to_span[pos[:(len(pos) - 1)]] = (ind, ind + 1) # Pre-terminal spans

    # Go through the rest of the tree bottom up to derive the correct span
    for pos in tree.treepositions(order="postorder"):

        subtree = tree[pos] # Extract the subtree at that position
        if not isinstance(subtree, str) and len(subtree.productions()[0].rhs()) > 1:  # excludes terminals (with str condition) and pre terminals

            left = pos + (0,)
            right = pos + (1,)

            if left in pos_to_span and right in pos_to_span:
                start = pos_to_span[left][0]
                end = pos_to_span[right][1]

                pos_to_span[pos] = (start, end)

    return pos_to_span


def transform_nltk_trees(trees):
    """
    Transform a list of nltk trees into ConstituencyTrees.

    :param trees: np.array of nltk trees
    :return: list of ConstituencyTrees
    """
    new_trees = np.empty(shape=len(trees), dtype=object)

    for ind, tree in enumerate(trees):

        pos_to_span = set_up_pos_to_span_dict(tree) # Extract the dictionary mapping nltk positions to spans

        prods = []
        for pos in tree.treepositions(order="postorder"): # Go through productions bottom-up

            subtree = tree[pos]

            if not isinstance(subtree, str): # excludes terminals

                prod = subtree.productions()[0] # to get the actual production

                # Transform into new production
                prod_new = Production(start=Node(symbol=str(prod.lhs()), pos=pos_to_span[pos]),
                                      end=[Node(symbol=str(symb), pos=pos_to_span[pos + (ind,)]) for ind, symb in enumerate(prod.rhs())])

                prods.append(prod_new)

        new_trees[ind] = ConstituencyTree(productions=prods)

    return new_trees


def save_data():
    """
    Convert the gold standard parse trees and the candidate parse trees into the custom data structure and dump them
    into a .npy file

    :return: None
    """
    true_parses, parses, embeddings = generate_parse_trees_and_embeddings()

    print("Generator done")

    new_true_trees = transform_nltk_trees(true_parses)

    new_cands_lists = np.empty(shape=len(true_parses), dtype=object)
    for ind, cands in enumerate(parses):
        new_cands_lists[ind] = transform_nltk_trees(cands)

    np.save("true_parses.npy", new_true_trees)
    np.save("cand_parses.npy", new_cands_lists)
    np.save("embeddings.npy", embeddings)

    print("Files successfully saved")

if __name__ == "__main__":
    save_data()