from io import open
from conllu import parse_tree_incr, parse
import pickle

import numpy as np
from os.path import isfile
from pathlib import Path
from nltk.parse.dependencygraph import DependencyGraph
from parsing.dep_tree_models.helpers.test_helpers import convert_to_custom_data_structure

from spanningtrees.graph import Graph
from spanningtrees.kbest import KBest

LANG = "lt_hse"
BASE_FOLDER = Path.cwd()
K = 10

def load_ud_data(lang, data_part):  # data_part can be train, dev or test
    """
    Load the train, dev or test set of a UD treebank for a given language
    
    :param lang: str
    :param data_part: str
    :return: tuple of lists
    """

    # Set up the path
    path = BASE_FOLDER / f"data/{lang}/{data_part}/{lang}-ud-{data_part}.conllu"
    path_str = str(path)

    # Load the file
    with open(path_str, "r", encoding="utf-8") as f:
        file = f.read()

    gold_annotations = parse(file)

    # Define start signal
    start_signal = "\n1"

    tokens = []
    sentences = []
    dep_graph_strs = []


    count = 0
    for gold_annotation in gold_annotations:

        # Sentences
        #print(gold_annotation.metadata["text"])
        count += 1
        sentences.append(gold_annotation.metadata["text"])

        # Tokens
        sentence_tokens = []
        for token in gold_annotation:
            sentence_tokens.append((token["id"], token["form"]))

        tokens.append(sentence_tokens)

        # Dependency Graph Strings
        if None in [val for val in gold_annotation.metadata.values()]: # then serialize() fails
            gold_annotation.metadata = None
            gold_annotated_str = gold_annotation.serialize()
            dep_graph_strs.append(gold_annotated_str)

        else:
            gold_annotated_str = gold_annotation.serialize()
            dep_graph_strs.append(gold_annotated_str[(gold_annotated_str.find(start_signal) + 1):])

    return sentences, tokens, dep_graph_strs # sonst ohne tokens


def build_true_dep_trees(lang, data_part, dep_graph_strs, path=None, visualize_trees=False):
    """
    Convert a list of UD tables into DependencyTrees and optionally visualize the tree if they are generated.

    :param lang: str
    :param data_part: str
    :param dep_graph_strs: list of str
    :param visualize_trees: bool
    :return: np.array of DependencyTrees
    """
    # Load the saved data if they can be found in the directory
    if path is None:
        path = BASE_FOLDER / f"data/{lang}/{data_part}/{lang}-true_dep_trees_{data_part}.npy"

    if path.is_file():#isfile(str(path)):# and isfile(path_tokens): #str(BASE_FOLDER / f"{lang}-true_dep_trees_{data_part}.npy")
        true_dep_trees = np.load(str(path), allow_pickle=True)
        return true_dep_trees

    # Initialize an array
    true_dep_trees = np.empty(shape=len(dep_graph_strs), dtype=object)

    for ind, dep_graph_str in enumerate(dep_graph_strs):

        dep_graph = DependencyGraph(dep_graph_str, top_relation_label="root", cell_separator="\t")

        if visualize_trees:
            dep_graph.tree().pretty_print()

        # Convert into custom data structure that enables faster computation of the model
        new_dep_tree = convert_to_custom_data_structure(dep_graph, conllu_str=dep_graph_str)

        true_dep_trees[ind] = new_dep_tree

    # Save the array for later use
    np.save(str(path), true_dep_trees)

    return true_dep_trees

def adjust_dep_labels(tokens, heads, labels):

    # Set up from the gold file and then match the prediction in
    new_sent_token_list = []

    ind = 0
    for address, token in tokens:

        head = str(heads[ind])
        label = labels[ind]

        if isinstance(address, tuple):
            address = "".join([str(element) for element in address])
            head = "_"
            label = "_"
            ind -= 1

        new_sent_token_list.append("\t".join([str(address), token, *["_" for _ in range(4)], head, label,
                                    *["_" for _ in range(2)]]))

        ind += 1

    new_conllu_str = "\n".join(new_sent_token_list)

    # print(new_conllu_str)

    return new_conllu_str


def decode_k_best_dep_trees(true_tree, tokens, n_id, adj_mat, label_mat, vocab, k, data_part, dev_gold=False): #stanza_pred):
    """
    Given an adjacency matrix and the corresponding label matrix, generate k-best candidate dependency trees.
    In case there are less than k possible dependency trees, return all candidates.

    :param true_dep_graph_str: str
    :param n_id: int
    :param adj_mat: np.array
    :param label_mat: np.array
    :param vocab: np.array
    :param k: int
    :return: list
    """

    new_adj_mat = np.exp(1 / (np.float64(adj_mat.T) - np.finfo(float).eps))


    g = Graph.build(new_adj_mat)

    k_best_trees = []  # Could be shorter than 5, if there are not enough candidate parses
    for ind, tree in enumerate(KBest(g, True).kbest()):

        if len(k_best_trees) == k:
            break

        tree_heads = tree.to_array()[1:]  # Important: OMIT the -1

        tree_labels = [vocab[label_mat[j + 1][h]] for j, h in enumerate(tree_heads)]
        new_str = adjust_dep_labels(tokens=tokens, heads=tree_heads, labels=tree_labels)

        # Transform into custom data structure
        dep_graph = DependencyGraph(new_str, top_relation_label="root", cell_separator="\t")

        #dep_graph.tree().pretty_print()

        # assign the DependencyTree to the array
        new_tree = convert_to_custom_data_structure(dep_graph=dep_graph, conllu_str=new_str)
        k_best_trees.append(new_tree) # maybe I need a mapping inside there

    return k_best_trees


def load_base_parser_output(lang, data_part, iterator, tokenized):#, sents):
    """
    Load the labeled adjacency matrices and vocabulary from the Stanza model for a list of sentences
    
    :param lang: str
    :param data_part: str
    :param iterator (either sents or tokens): list of str
    :return: 3-tuple of np.arrays
    """
    # Check if the files are already available - otherwise generate them
    # To generate the files with Stanza, the Stanza code needs to be modified

    base_path = BASE_FOLDER / f"data/{lang}"

    scores_path = str(base_path / f"{data_part}/scores.pkl")
    labels_path = str(base_path / f"{data_part}/labels.pkl")
    vocab_path = str(base_path / "vocab.npy")

    if not isfile(scores_path) or not isfile(labels_path) or not isfile(vocab_path):
        # Need to hack into Stanza and extract matrices to directory
        # For simplicity: Just provided at the polybox link in the repo

        from stanza import Pipeline, download

        # Use the stanza model to generate the vocab as well as the adjacency and corresponding label matrix
        lang, package = lang.split(sep="_")

        try:
            download(lang=lang, package=None,
                     processors={"tokenize": package, "pos": package, "lemma": package, "mwt": package,
                                 "depparse": package})

            if tokenized:
                nlp = Pipeline(lang=lang, package=package, processors='tokenize,mwt,pos,lemma,depparse',
                           tokenize_pretokenized=True)
            else:
                nlp = Pipeline(lang=lang, package=package, processors='tokenize,mwt,pos,lemma,depparse',
                           tokenize_nossplit=True)
        except:
            download(lang=lang, package=None,
                     processors={"tokenize": package, "pos": package, "lemma": package, "depparse": package})
            if tokenized:
                nlp = Pipeline(lang=lang, package=package, processors='tokenize,pos,lemma,depparse',
                           tokenize_pretokenized=True)
            else:
                nlp = Pipeline(lang=lang, package=package, processors='tokenize,pos,lemma,depparse',
                           tokenize_nossplit=True)

        if tokenized:
            for ind, sent in enumerate(iterator):
                nlp([sent])
        else:
            for sent in iterator:
                nlp(sent)

    with open(scores_path, 'rb') as infile1:
        scores = pickle.load(infile1)

    with open(labels_path, 'rb') as infile2:
        labels = pickle.load(infile2)

    vocab = np.load(vocab_path, allow_pickle=True)

    return scores, labels, vocab

def extract_single_word_tokens(tokens):

    new_tokens = []
    for sent_tokens in tokens:
        new_sent_tokens = []
        for (address, token) in sent_tokens:
            if not isinstance(address, tuple):
                new_sent_tokens.append((address, token))

        new_tokens.append(new_sent_tokens)

    return new_tokens

def build_candidate_dep_trees(lang, data_part, k, sents, tokens, true_dep_trees, tokenized, path=None,
                              test_with_gold_tree=False, return_gold_tree_ratio=False):
    """

    :param lang:
    :param data_part:
    :param sents:
    :param dep_graph_strs:
    :param true_dep_trees:
    :return:
    """

    pre_path = BASE_FOLDER / f"data/{lang}/{data_part}/cands"
    pre_path.mkdir(parents=True, exist_ok=True)

    if path is None:

    # Check if the desired candidate dependency trees are already saved and if yes, load them
        if data_part == "test" or data_part == "dev":
            # Create a folder if it's not there yet

            path = pre_path / f"{lang}-{k}_best_cand_dep_trees_{data_part}.npy"

            if test_with_gold_tree:
                path = pre_path / f"{lang}-{k}_best_cand_dep_trees_{data_part}-gt{test_with_gold_tree}.npy"

        else:
            # For the training set
            path = BASE_FOLDER / f"data/{lang}/{data_part}/{lang}-{k}_best_cand_dep_trees_{data_part}.npy"

    gtr_path = pre_path / f"{lang}-{k}_gold_tree_ratio_{data_part}.npy"

    if path.is_file() and not return_gold_tree_ratio:

        cand_dep_trees = np.load(str(path), allow_pickle=True)

        #excluded_inds = np.load(f"{lang}-excluded_sentences_{data_part}.npy", allow_pickle=True)
        return cand_dep_trees#, excluded_inds

    if path.is_file() and gtr_path.is_file():

        cand_dep_trees = np.load(str(path), allow_pickle=True)
        gold_tree_ratio = np.load(str(pre_path / f"{lang}-{k}_gold_tree_ratio_{data_part}.npy"), allow_pickle=True)
        print(gold_tree_ratio)

        return cand_dep_trees, gold_tree_ratio


    # Only pass through the single word tokens
    new_tokens_incl_addresses = extract_single_word_tokens(tokens=tokens)
    new_tokens = [[token for address, token in new_sent_tokens] for new_sent_tokens in new_tokens_incl_addresses]

    if tokenized:
        scores, labels, vocab = load_base_parser_output(lang=lang, data_part=data_part, iterator=new_tokens,
                                                        tokenized=tokenized)#, sents=sents) # tokens_list, stanza_preds
    else:
        scores, labels, vocab = load_base_parser_output(lang=lang, data_part=data_part, iterator=sents,
                                                        tokenized=tokenized)

    #cand_parse_lists = []  # it can't be known how many those are in advance (because we are excluding the differently tokenized ones)
    cand_parse_arrays = np.empty(len(tokens), dtype=object)
    #excluded_inds = []

    count = 0

    for ind, tokens_list in enumerate(new_tokens_incl_addresses):


        adj_mat = scores[ind]

        if data_part == "dev" and test_with_gold_tree is True:
            cands = decode_k_best_dep_trees(true_tree=true_dep_trees[ind],
                                        tokens=tokens_list,
                                        n_id=ind,
                                        adj_mat=adj_mat[0],
                                        label_mat=labels[ind][0],
                                        vocab=vocab,
                                        k=k,
                                        data_part=data_part,
                                            dev_gold=True)
        else:
            cands = decode_k_best_dep_trees(true_tree=true_dep_trees[ind],
                                        tokens=tokens_list,
                                        n_id=ind,
                                        adj_mat=adj_mat[0],
                                        label_mat=labels[ind][0],
                                        vocab=vocab,
                                        k=k,
                                        data_part=data_part)

        if (data_part == "test" or data_part == "dev") and not test_with_gold_tree: #or data_part == "dev":

            for cand in cands:

                true = true_dep_trees[ind]

                if cand == true:
                    count += 1


            cand_parse_arrays[ind] = cands
            continue

        else:
            actual_cands = []

            true = true_dep_trees[ind]
            for cand in cands:

                if cand == true:
                    count += 1

                if cand != true:
                    actual_cands.append(cand)

            if len(actual_cands) > (k-1):
                actual_cands = actual_cands[:(k-1)]

            cand_parse_arrays[ind] = actual_cands


    print(f"Ratio of True Trees in the candidates for k:{k}")

    num_sentences = len(tokens)
    gold_tree_ratio = count/num_sentences

    print(gold_tree_ratio)

    np.save(str(path), cand_parse_arrays)
    np.save(str(pre_path / f"{lang}-{k}_gold_tree_ratio_{data_part}.npy"), gold_tree_ratio)

    if return_gold_tree_ratio:
        return cand_parse_arrays, gold_tree_ratio

    return cand_parse_arrays


def build_embeddings(lang, data_part, sents):
    """
    Generate the embeddings of the sentences for a given UD language

    :param lang: str
    :param data_part: str
    :param sents: list of str
    :return: np.array
    """
    # Check if file is already saved
    path = BASE_FOLDER / f"data/{lang}/{data_part}/{lang}-embeddings_{data_part}.npy"
    path_str = str(path)

    if isfile(path_str):
        embeddings = np.load(path_str, allow_pickle=True)
        return embeddings

    from sentence_transformers import SentenceTransformer

    embeddings = np.empty(len(sents), dtype=object)
    model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1') # takes in different languages automatically

    for ind, sent in enumerate(sents):
        embeddings[ind] = model.encode("".join(sent))

    np.save(path_str, embeddings)

    return embeddings


def prepare_dataset(data):#, excluded_inds):
    """
    Set up the data matrix and response vector from the parse trees and embeddings.

    :param data: list of np.arrays
    :return: tuple of np.arrays
    """
    true_parses_ind, parses_ind, embeddings_ind = data

    # Compute the length of the dataset
    dataset_len = 0

    for ind, cand_parse_arr in enumerate(parses_ind):
        dataset_len += len(cand_parse_arr) + 1  # +1 for the true parse

    # Initialize the response vector
    y = np.empty(shape=dataset_len)

    # Initialize the feature matrix
    embedding_length = embeddings_ind[0].shape[0]  # ASSUMPTION: All embeddings have the same length
    X = np.empty(shape=(dataset_len, embedding_length + 1), dtype=object)  # +1 for the trees

    ind = 0
    for sent_ind in range(len(true_parses_ind)):

        embedding = embeddings_ind[sent_ind]

        # True parse
        y[ind] = 1
        X[ind, 0] = true_parses_ind[sent_ind]
        X[ind, 1:] = embedding

        # Candidate parses
        num_cand_parses = len(parses_ind[sent_ind])
        for cand_parse_ind in range(num_cand_parses):
            y[ind + cand_parse_ind + 1] = 0
            X[ind + cand_parse_ind + 1, 0] = parses_ind[sent_ind][cand_parse_ind]
            X[ind + cand_parse_ind + 1, 1:] = embedding  # Embedding is the same for the candidates

        ind += num_cand_parses + 1

    return X, y


def prepare_test_set(data):#, excluded_inds):
    """
    Set up the test data matrix and response vector from the parse trees and embeddings.

    :param data: list of np.arrays
    :return: tuple of np.arrays
    """
    true_parses_ind, parses_ind, embeddings_ind = data

    # Compute the length of the dataset
    dataset_len = 0
    for cand_parse_arr in parses_ind:
        dataset_len += len(cand_parse_arr) # here we don't include the true parse

    # Initialize the response vector
    y = np.empty(shape=dataset_len)

    # Initialize the feature matrix
    embedding_length = embeddings_ind[0].shape[0]  # ASSUMPTION: All embeddings have the same length
    X = np.empty(shape=(dataset_len, embedding_length + 1), dtype=object)  # +1 for the trees

    ind = 0
    for sent_ind in range(len(true_parses_ind)):

        embedding = embeddings_ind[sent_ind]

        # Candidate parses
        num_cand_parses = len(parses_ind[sent_ind])
        for cand_parse_ind in range(num_cand_parses):
            y[ind + cand_parse_ind] = int(parses_ind[sent_ind][cand_parse_ind] == true_parses_ind[sent_ind])
            X[ind + cand_parse_ind, 0] = parses_ind[sent_ind][cand_parse_ind]
            X[ind + cand_parse_ind, 1:] = embedding  # Embedding is the same for the candidates

        ind += num_cand_parses #+ 1

    return X, y



def load_data_set(lang, data_part, k, tokenized):

    sents, tokens, dep_graph_strs = load_ud_data(lang, data_part=data_part)

    # Exceptions resulting from errors with the score matrix
    if lang == "lt_hse" and data_part == "dev":
        tokens = [token for ind, token in enumerate(tokens) if ind not in [16, 28]]
        sents = [sent for ind, sent in enumerate(sents) if ind not in [16, 28]]
        dep_graph_strs = [dep_graph_str for ind, dep_graph_str in enumerate(dep_graph_strs) if ind not in [16, 28]]

    if lang == "ta_ttb" and data_part == "train":
        tokens = [token for ind, token in enumerate(tokens) if ind != 199]
        sents = [sent for ind, sent in enumerate(sents) if ind != 199]
        dep_graph_strs = [dep_graph_str for ind, dep_graph_str in enumerate(dep_graph_strs) if ind != 199]

    true_dep_trees = build_true_dep_trees(lang=lang, data_part=data_part, dep_graph_strs=dep_graph_strs)
    cand_dep_trees = build_candidate_dep_trees(lang=lang, data_part=data_part, tokens=tokens, sents=sents, k=k,
                                                     #dep_graph_strs=dep_graph_strs,
                                                    true_dep_trees=true_dep_trees, tokenized=tokenized) # , excluded_inds

    embeddings = build_embeddings(lang=lang, data_part=data_part, sents=sents)

    data = [true_dep_trees, cand_dep_trees, embeddings]


    if data_part == "train":
        X, y = prepare_dataset(data=data)
        return X,y

    else:
        X, y = prepare_test_set(data=data)
        return X, y, tokens


def load_data_set_incl_gold(lang, k, data_part, tokenized):

    sents, tokens, dep_graph_strs = load_ud_data(lang, data_part=data_part)

    if lang == "lt_hse" and data_part == "dev":
        tokens = [token for ind, token in enumerate(tokens) if ind not in [16, 28]]
        sents = [sent for ind, sent in enumerate(sents) if ind not in [16, 28]]
        dep_graph_strs = [dep_graph_str for ind, dep_graph_str in enumerate(dep_graph_strs) if ind not in [16, 28]]

    true_dep_trees = build_true_dep_trees(lang=lang, data_part=data_part, dep_graph_strs=dep_graph_strs)

    # load the cand_dep_trees differently (they need to be set up like the training ones, i.e. without the true parse)
    cand_dep_trees = build_candidate_dep_trees(lang=lang, data_part=data_part, k=k, tokens=tokens, sents=sents,
                                               true_dep_trees=true_dep_trees, tokenized=tokenized,
                                               test_with_gold_tree=True)

    embeddings = build_embeddings(lang=lang, data_part=data_part, sents=sents)

    data = [true_dep_trees, cand_dep_trees, embeddings]

    X, y = prepare_dataset(data=data)  # Note that this is the function for the training set

    return X, y, tokens


if __name__ == "__main__":
    for k in range(1,16):
        X_train, y_train, tokens = load_data_set(lang="lt_hse", data_part="test", k=k, tokenized=True)

