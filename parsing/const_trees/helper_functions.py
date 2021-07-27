from random import seed

import numpy as np
from eval_metrics import ece, constituent_f1

from sklearn.metrics import roc_auc_score, average_precision_score

seed(42)
np.random.seed(42)


def extract_sentences(data, sent_inds):
    """
    Extract the true parses, candidate parses and embeddings for the given sentences.

    :param data: list of np.arrays
    :param sent_inds: list of int
    :return: 3-tuple of lists
    """
    true_parses, parses, embeddings = data

    return [true_parses[ind] for ind in sent_inds], [parses[ind] for ind in sent_inds], [embeddings[ind] for ind in sent_inds]


def prepare_dataset(data, sent_inds):
    """
    Set up the data matrix and response vector from the parse trees and embeddings.

    :param data: list of np.arrays
    :param sent_inds: list of int
    :return: tuple of np.arrays
    """
    true_parses_ind, parses_ind, embeddings_ind = extract_sentences(data, sent_inds)

    # Compute the length of the dataset
    dataset_len = 0
    for cand_parse_arr in parses_ind:
        dataset_len += cand_parse_arr.shape[0] + 1  # +1 for the true parse

    # Initialize the response vector
    y = np.empty(shape=dataset_len)

    # Initialize the feature matrix
    embedding_length = embeddings_ind[0].shape[0] # ASSUMPTION: All embeddings have the same length
    X = np.empty(shape=(dataset_len, embedding_length + 1), dtype=object) # +1 for the trees

    ind = 0
    for sent_ind in range(len(true_parses_ind)):

        embedding = embeddings_ind[sent_ind]

        # True parse
        y[ind] = 1
        X[ind, 0] = true_parses_ind[sent_ind]
        X[ind, 1:] = embedding

        # Candidate parses
        num_cand_parses = parses_ind[sent_ind].shape[0]
        for cand_parse_ind in range(num_cand_parses):
            y[ind + cand_parse_ind + 1] = 0
            X[ind + cand_parse_ind + 1, 0] = parses_ind[sent_ind][cand_parse_ind]
            X[ind + cand_parse_ind + 1, 1:] = embedding # Embedding is the same for the candidates

        ind += num_cand_parses + 1

    return X, y


def evaluate_classifier(y_test, y_pred_prob, X_test, num_sentences, num_cands):
    """
    Evaluate classifier on AUC, Average Precision, ECE and Constituent F1 score.

    :param y_test: list/np.array
    :param y_pred_prob: list/np.array
    :param X_test: np.array
    :param num_sentences: int
    :param num_cands: int
    :return: 4-tuple of floats
    """
    auc = roc_auc_score(y_true=y_test, y_score=y_pred_prob)
    avg_prec = average_precision_score(y_true=y_test, y_score=y_pred_prob)
    ece_val = ece(y_true=y_test, y_pred_prob=y_pred_prob)
    f1_val = constituent_f1(y_true=y_test, y_pred_prob=y_pred_prob, X_test=X_test,
                            num_sentences=num_sentences, num_cands=num_cands)

    return auc, avg_prec, ece_val, f1_val