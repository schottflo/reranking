import numpy as np

def ece(y_true, y_pred_prob, num_bins=10):
    """
    Compute the expected calibration error of a classifier.

    :param y_true: np.array
    :param y_pred_prob: np.array
    :param num_bins: int
    :return: float
    """
    # Set up the intervals
    intervals = [i / num_bins for i in range(1, num_bins + 1)]

    n = len(y_true)

    labels = list(y_true)
    probs = list(y_pred_prob)

    label_bins = []
    prob_bins = []

    for max_prob in intervals:

        # Find the index of the relevant probabilities for each interval
        inds = [ind for ind, prob in enumerate(y_pred_prob) if prob <= max_prob]

        # Add the corresponding labels and probabilities to the overall bin list
        label_bins.append([y_true[ind] for ind in inds])
        prob_bins.append([y_pred_prob[ind] for ind in inds])

        # Update the remaining probabilities
        y_pred_prob = [y_pred_prob[ind] for ind in range(len(y_pred_prob)) if ind not in inds]
        y_true = [y_true[ind] for ind in range(len(y_true)) if ind not in inds]

    # Formula from "On the Calibration of Modern Neural Networks"

    total = 0
    for ind in range(num_bins):

        # Skip empty bins
        if len(label_bins[ind]) == 0:
            continue

        # Implementation of the formula
        accuracy = np.mean(label_bins[ind])
        confidence = np.mean(prob_bins[ind])
        total += len(label_bins[ind]) / n * np.abs(accuracy - confidence)

    return total


def constituent_f1(y_true, y_pred_prob, X_test, num_sentences, num_cands):
    """
    Compute the constituent level F1 score.

    :param y_true: np.array
    :param y_pred_prob: np.array
    :param X_test: np.array
    :param num_sentences: int
    :param num_cands: int
    :return: float
    """
    # Extract the indices of the most probable parse tree according to the model
    max_inds = np.empty(num_sentences, dtype=int)

    ind = 0
    for sent in range(num_sentences):

        cand_pred = np.empty(num_cands + 1)  # Set up an array for each num of candidates
        for cand in range(num_cands + 1):
            cand_pred[cand] = y_pred_prob[ind + cand]

        # Index of most probable parse tree for the given sentence
        max_inds[sent] = ind + np.argmax(
            cand_pred)  # Argmax always picks the last highest values in an array; thus if they are ties it will take the worse one (since the true one is always first in the array)

        # Increment the index
        ind += num_cands + 1

    true_inds = np.where(y_true > 0)[0]

    # Given the indices of the true trees and most probable trees acc. to the model, compute the F1 score
    num_correct_const = 0
    total_true_const = 0
    total_cand_const = 0

    for sent in range(num_sentences):

        # Extract the true and most probable tree for each sentence
        best_cand_tree = X_test[max_inds[sent], 0]
        true_tree = X_test[true_inds[sent], 0]

        # Take all constituents from the cand tree and the true tree
        cand_nts = best_cand_tree.nonterminals()
        true_nts = true_tree.nonterminals()

        # Increment the number of correct constituents by adding the number of correct constituents for each sentence
        for cand_nt in cand_nts:
            for true_nt in true_nts:
                if cand_nt.symbol == true_nt.symbol and cand_nt.pos == true_nt.pos:
                    num_correct_const += 1

        # Increment the total number of true and candidate constituents
        total_true_const += len(true_nts)
        total_cand_const += len(cand_nts)

    # Calculate total precision and total recall
    rec = num_correct_const / total_true_const
    prec = num_correct_const / total_cand_const

    return (2 * prec * rec) / (prec + rec)  # Compute the F1 score


def old_ece(y_true, y_pred, y_pred_conf, num_bins=10):
    """
    Compute the expected calibration error of a classifier.

    :param y_true: list/np.array
    :param pred_conf: list/np.array
    :param y_pred: list/np.array
    :param num_bins: int
    :return: float
    """
    # Convert inputs to lists
    y_true = list(y_true)
    y_pred = list(y_pred)
    y_pred_conf = list(y_pred_conf)

    # Number of samples
    n = len(y_true)

    # Initialize the interval lists
    conf_bins = []
    pred_bins = []
    true_bins = []

    intervals = [i / num_bins for i in range(1, num_bins + 1)]
    for max_prob in intervals:

        # Gather the predictions of a given interval
        interval_ind = [ind for ind, prob in enumerate(y_pred_conf) if prob <= max_prob]

        interval_pred_conf = [y_pred_conf[ind] for ind in interval_ind]
        interval_pred = [y_pred[ind] for ind in interval_ind]
        interval_true = [y_true[ind] for ind in interval_ind]

        # Update the remaining predictions
        y_pred_conf = [y_pred_conf[ind] for ind in range(len(y_pred_conf)) if ind not in interval_ind]
        y_pred = [y_pred[ind] for ind in range(len(y_pred)) if ind not in interval_ind]
        y_true = [y_true[ind] for ind in range(len(y_true)) if ind not in interval_ind]

        # Maintain one list per bin and append to bins list
        conf_bins.append(interval_pred_conf)
        pred_bins.append(interval_pred)
        true_bins.append(interval_true)

    # Compute the ECE
    total = 0
    for ind in range(num_bins):

        accuracy = np.sum(np.array(true_bins[ind]) == np.array(pred_bins[ind]))
        confidence = np.sum(np.array(conf_bins[ind]))
        total += 1 / n * np.abs(accuracy - confidence)

    return total


def old_constituent_f1(y_true, y_pred_prob, X_test, num_sentences, num_cands):
    """
    Compute the constituent level F1 score.

    :param y_true: np.array
    :param y_pred_prob: np.array
    :param X_test: np.array
    :param num_sentences: int
    :param num_cands: int
    :return: float
    """
    # Extract the indices of the most probable parse tree according to the model and the true tree for each sentence
    max_inds = np.empty(num_sentences, dtype=int)

    ind = 0
    for sent in range(num_sentences):
        cand_pred = np.empty(num_cands + 1)  # Set up a list for each num of candidates
        for cand in range(num_cands + 1):
            cand_pred[cand] = y_pred_prob[ind + cand]

        # Most probable parse trees and true trees
        max_inds[sent] = ind + np.argmax(
            cand_pred)  # Argmax always picks the last highest values in an array; thus if they are ties it will take the worse one (since the true one is always first in the array)

        # Increment the index
        ind += num_cands + 1

    true_inds = np.where(y_true > 0)[0]

    # Get the best candidate parse tree (including the true tree)
    f1 = np.empty(num_sentences)
    sent_len = np.empty(num_sentences)
    for sent in range(num_sentences):

        best_cand_tree = X_test[max_inds[sent], 0]
        true_tree = X_test[true_inds[sent], 0]

        # Take all constituents from the cand tree and the true tree
        cand_nts = best_cand_tree.nonterminals()
        true_nts = true_tree.nonterminals()

        # Get the numbers
        num_correct_const = 0
        for cand_nt in cand_nts:
            for true_nt in true_nts:
                if cand_nt.symbol == true_nt.symbol and cand_nt.pos == true_nt.pos:
                    num_correct_const += 1

        total_true_const = len(true_nts)
        total_cand_const = len(cand_nts)

        # Calculate precision and recall
        rec = num_correct_const / total_true_const
        prec = num_correct_const / total_cand_const

        sent_len[sent] = len(
            true_tree.terminals())  # Could implement this in to the data structure, so one just needs to call len(.)
        f1[sent] = (2 * prec * rec) / (prec + rec)

    return np.dot(sent_len, f1) / sum(sent_len)  # Weighted average according to sentence length

if __name__ == "__main__":
    print(ece(y_true=[0,0,1,0,0], y_pred_prob=[0.1,0.3,0.5,0.6,0.4]))
    print(old_ece(y_true=[0,0,1,0,0], y_pred=[0,0,1,1,0], y_pred_conf=[0.9,0.7,0.5,0.6,0.6]))