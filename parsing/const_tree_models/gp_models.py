from joblib import Parallel, delayed
from os.path import isfile
import time
from random import seed, shuffle, sample

import numpy as np

from parsing.const_tree_models.data_generator import save_data
from parsing.const_tree_models.helpers.modeling_helpers import prepare_dataset, evaluate_classifier
from parsing.const_tree_models import gp

SHUFFLES = 3
NUM_SENTENCES = 400

seed(42)
np.random.seed(42)


def run_dataset(data, sent, num_cands):
    """
    Compute a GP Classification model on a training set and evaluate it on a test set.

    :param sent: list of int
    :param data: list of np.arrays
    :param num_cands: int
    :return: 4-tuple of floats
    """
    # Take 1/2 of the dataset randomly as the training set; the other half as test set
    train_sent = sample(sent, k=int(len(sent) / 2))
    test_sent = [ind for ind in sent if ind not in train_sent]

    # Generate the train and test dataset
    X_train, y_train = prepare_dataset(data=data, sent_inds=train_sent)
    X_test, y_test = prepare_dataset(data=data, sent_inds=test_sent)

    print("Data prepared")

    # GP
    y_pred_prob_gp = gp.fit_and_predict_gp(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    # Evaluate the model
    return evaluate_classifier(y_test=y_test, y_pred_prob=y_pred_prob_gp, X_test=X_test,
                               num_sentences=len(test_sent), num_cands=num_cands)


def main():
    if not (isfile("true_parses.npy") and isfile("cand_parses.npy") and isfile("embeddings.npy")):
        print("--Data is generated--")
        save_data()

    start = time.time()

    # Load the data and adjust the number of sentences used
    true_parses = np.load("true_parses.npy", allow_pickle=True)
    cand_parses = np.load("cand_parses.npy", allow_pickle=True)
    embeddings = np.load("embeddings.npy", allow_pickle=True)

    data = [true_parses[:NUM_SENTENCES], cand_parses[:NUM_SENTENCES], embeddings[:NUM_SENTENCES]]

    # Determine the number of candidate parses in the dataset
    num_cands = data[1][0].shape[0]  # ASSUMPTION: Constant number of candidates

    # Create SHUFFLES many shuffled datasets
    sents = list(range(NUM_SENTENCES))  # Indices of the sentences

    shuffled_sents = []
    for shuffle_i in range(SHUFFLES):
        # Shuffle the order of the sentences
        shuffle(sents)
        sents_n = sents.copy()
        shuffled_sents.append(sents_n)

    # Compute SHUFFLES many GP Classification models
    gp_results = Parallel(n_jobs=SHUFFLES)(
        delayed(run_dataset)(data=data, sent=sent, num_cands=num_cands) for sent in shuffled_sents)

    end = time.time()

    # Compute the needed time
    print(end - start)

    print(gp_results)
    np.save("gp_results.npy", gp_results)


if __name__ == "__main__":
    main()
