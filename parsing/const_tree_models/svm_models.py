from joblib import Parallel, delayed
from os.path import isfile
import time
from random import seed, shuffle, sample, randrange

from parsing.const_tree_models.data_generator import save_data
from parsing.const_tree_models import svm

from parsing.const_tree_models.helpers.modeling_helpers import prepare_dataset, evaluate_classifier
from parsing.const_tree_models.eval_metrics import ece, constituent_f1

import numpy as np
from sklearn.model_selection import ParameterGrid

SHUFFLES = 3
CV = 5
NUM_SENTENCES = 400

seed(42)
np.random.seed(42)


def create_CV_folds(sent_inds, folds=5):
    """
    Split the sentences in "folds" many folds

    :param sent_inds: list of int
    :param folds: int
    :return: list of list
    """
    sent_split = list()
    sent_copy = list(sent_inds)
    fold_size = int(len(sent_inds) / folds)

    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(sent_copy))
            fold.append(sent_copy.pop(index))
        sent_split.append(fold)

    return sent_split


def run_cv_on_hyperparameter_combination(params, data, folds, num_cands):
    """
    Run CV based on the folds with the given data and the parameter combination

    :param params: dict
    :param data: list of lists
    :param folds: list of list
    :param num_cands: int
    :return: int
    """
    # Initialize the results vector for a given parameter combination (is tested on "folds"-many test sets)
    param_res = np.zeros(len(folds))

    # Iterate through the folds
    for ind, current_fold in enumerate(folds):
        # Construct training and validation set
        cv_train_sent = [sent for fold in folds if fold != current_fold for sent in fold]
        val_sent = current_fold

        # Build the dataset
        X_cv_train, y_cv_train = prepare_dataset(data=data, sent_inds=cv_train_sent)
        X_val, y_val = prepare_dataset(data=data, sent_inds=val_sent)

        # Fit a SVM on CV training set and predict on validation set
        y_val_pred_prob = svm.fit_and_predict_svm(X_train=X_cv_train, X_test=X_val, y_train=y_cv_train,
                                                  params=params)

        # Compute and return AUC value
        param_res[ind] = constituent_f1(y_true=y_val, y_pred_prob=y_val_pred_prob, X_test=X_val,
                                        num_sentences=len(val_sent), num_cands=num_cands)

    avg_const_f1 = np.mean(param_res)
    print(f"Combination {params} resulted in {avg_const_f1} average Constituent F1")
    return avg_const_f1


def main():
    if not (isfile("true_parses.npy") and isfile("cand_parses.npy") and isfile("embeddings.npy")):
        print("--Data is generated--")
        save_data()

    start = time.time()

    true_parses = np.load("true_parses.npy", allow_pickle=True)
    cand_parses = np.load("cand_parses.npy", allow_pickle=True)
    embeddings = np.load("embeddings.npy", allow_pickle=True)

    data = [true_parses[:NUM_SENTENCES], cand_parses[:NUM_SENTENCES], embeddings[:NUM_SENTENCES]]

    svm_results = np.zeros((SHUFFLES, 4))  # Shape: num_shuffles x num_evaluation_metrics

    num_cands = data[1][0].shape[0]  # ASSUMPTION: Constant number of candidates
    sent = list(range(NUM_SENTENCES))  # Indices of the sentenences

    for shuffle_i in range(SHUFFLES):
        # Shuffle the order of the sentences
        shuffle(sent)

        # Take 1/2 of the dataset randomly as the training set; the other half as test set
        train_sent = sample(sent, k=int(len(sent) / 2))
        test_sent = [ind for ind in sent if ind not in train_sent]

        # Construct cross validation splits to tune the regularization parameter and the hyperparameters
        folds = create_CV_folds(sent_inds=train_sent, folds=CV)

        # Build a parameter grid
        param_grid = {'C': np.logspace(-5, 0, num=4, endpoint=True), 'lamb': np.linspace(1e-5, 1, num=4, endpoint=True),
                      'e': np.logspace(-5, 0, num=6, endpoint=True)}
        param_comb = list(ParameterGrid(param_grid))

        # Initialize the results vector that will contian the average results for each parameter combination
        params_avg_res = Parallel(n_jobs=48)(
            delayed(run_cv_on_hyperparameter_combination)(params=params, data=data, folds=folds, num_cands=num_cands)
            for params in param_comb)

        # Take the hyperparameter combination with the maximal average result
        max_ind = np.argmax(params_avg_res)
        final_params = param_comb[max_ind]

        max_params_avg_res = np.max(params_avg_res)

        print \
            (
                f"The best performing parameter combination was {final_params} with an average constituent F1 of {max_params_avg_res}")
        print("Cross validation done")

        # Generate the train and test dataset (also to be used by GP later)
        X_train, y_train = prepare_dataset(data=data, sent_inds=train_sent)
        X_test, y_test = prepare_dataset(data=data, sent_inds=test_sent)

        # Fit the SVM
        y_pred_prob_svm = svm.fit_and_predict_svm(X_train=X_train, X_test=X_test, y_train=y_train, params=final_params)

        # Get the metrics
        svm_results[shuffle_i, :] = evaluate_classifier(y_test=y_test, y_pred_prob=y_pred_prob_svm, X_test=X_test,
                                                        num_sentences=len(test_sent), num_cands=num_cands)

        print(svm_results)

    end = time.time()

    # Compute the needed time
    print(end - start)

    print(svm_results)
    np.save("svm_results.npy", svm_results)


if __name__ == "__main__":
    main()
