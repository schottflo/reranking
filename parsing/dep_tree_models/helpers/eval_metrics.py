import numpy as np

def transform_into_conllu(custom_dep_tree):

    conllu_str = []

    for address, token, head, rel in custom_dep_tree.conllu_information():
        conllu_str.append("\t".join([str(address), token, *["_" for _ in range(4)],
                                     str(head), rel, *["_" for _ in range(2)]]))

    return "\n".join(conllu_str)

def extract_the_most_probable_trees(y_pred_prob, cand_trees, true_trees, conf_thresh=0, gold_tree_incl=False):

    num_sentences = len(cand_trees)

    most_probable_trees = []

    ind = 0
    empirical_stanza_count = 0
    for sent in range(num_sentences):

        cands = cand_trees[sent]

        if gold_tree_incl:
            cands.insert(0, true_trees[sent]) # needs to be at first position

        num_cands = len(cands)  # for the true parse - eventuell ändern!

        cand_pred = np.empty(num_cands)  # Set up an array for each num of candidates
        for cand in range(num_cands):
            cand_pred[cand] = y_pred_prob[ind + cand]


        if np.max(cand_pred) >= conf_thresh:
        # Index of most probable parse tree for the given sentence
            max_ind = np.argmax(cand_pred)  # Argmax always picks the last highest values in an array; thus if they are ties it will take the worse one (since the true one is always first in the array)
        else:
            max_ind = 0

        if gold_tree_incl and max_ind == 1:
            empirical_stanza_count += 1

        if not gold_tree_incl and max_ind == 0:
            empirical_stanza_count += 1

        most_probable_trees.append(cands[max_ind])

        ind += num_cands

    assert num_sentences == len(most_probable_trees)

    empirical_stanza_ratio = empirical_stanza_count/num_sentences

    return most_probable_trees, empirical_stanza_ratio

def adjust_the_labels(conllu_str, true_tokens):

    # Set up from the gold file and then match the prediction in
    new_sent_token_list = []
    for address, token in true_tokens:

        if isinstance(address, tuple):
            address = "".join([str(element) for element in address])

        new_sent_token_list.append([str(address), token, *["_" for _ in range(8)]])#, str(new_head), new_label, *["_" for _ in range(2)]]))

    # Split the conllu_str
    tokens = conllu_str.split(sep="\n")

    for token in tokens:

        fields = token.split(sep="\t")

        for new_sent_token in new_sent_token_list:

            if fields[0] == new_sent_token[0] and fields[1] == new_sent_token[1]:

                # Add the head and dependent
                new_sent_token[6] = fields[6]
                new_sent_token[7] = fields[7]

    token_strs = []
    for new_sent_token in new_sent_token_list:
        token_strs.append("\t".join(new_sent_token))

    new_conllu_str = "\n".join(token_strs)

    return new_conllu_str

def dump_predictions(lang, y_pred_prob, cand_trees, true_trees, tokens, name, conf_thresh=0, path=None, gold_tree_incl=False):
    """
    Construct CoNLL-U file for the UD test set.

    :param lang: str
    :param y_pred_prob:
    :param cand_trees:
    :return:
    """
    most_prob_cand_trees, empirical_stanza_ratio = extract_the_most_probable_trees(y_pred_prob, cand_trees, true_trees, conf_thresh, gold_tree_incl)

    if path is None:
        new_path = f"{lang}-{name}-predictions.conllu"
    else:
        new_path = path / f"{lang}-{name}-predictions.conllu"

    with open(str(new_path), "w", encoding="utf-8") as f:

        for ind, tree in enumerate(most_prob_cand_trees):

            output = tree.conllu_str

            new_output = adjust_the_labels(conllu_str=output, true_tokens=tokens[ind]) # Tokens needs to be a list of (address, token) tuples

            f.write(new_output)
            f.write('\n\n')

    return empirical_stanza_ratio

if __name__ == '__main__':
    pass