from stanza import Pipeline, download
from stanza.utils.conll import CoNLL
from pathlib import Path
from parsing.dep_tree_models.helpers.data_preparation import load_ud_data
from parsing.dep_tree_models.helpers.eval_metrics import adjust_the_labels
from parsing.dep_tree_models.helpers.performance_eval import EvaluatorInput
from parsing.dep_tree_models.helpers.conll18_ud_eval import evaluate_wrapper

BASE_FOLDER = Path.cwd()

DATA_PART = "test"
TOKENIZED = True

def get_single_word_tokens_incl_address(tokens):

    new_tokens = []
    for sent_tokens in tokens:
        new_sent_tokens = []
        for (address, token) in sent_tokens:
            if not isinstance(address, tuple):
                new_sent_tokens.append(token)

        new_tokens.append(new_sent_tokens)

    return new_tokens

def get_baseline_values(lang):

    sents, tokens, dep_graph_str = load_ud_data(lang=lang, data_part=DATA_PART)

    new_tokens = get_single_word_tokens_incl_address(tokens=tokens)

    lang_name, package = lang.split(sep="_")

    # Setup the base parser
    try:
        download(lang=lang_name,package=None,processors={"tokenize":package, "pos":package, "lemma":package, "mwt": package, "depparse":package})

        if TOKENIZED:
            nlp = Pipeline(lang=lang_name, package=package, processors='tokenize,mwt,pos,lemma,depparse', tokenize_pretokenized=True)
        else:
            nlp = Pipeline(lang=lang_name, package=package, processors='tokenize,mwt,pos,lemma,depparse', tokenize_nossplit=True)
    except:
        print("No MWT language")
        download(lang=lang_name,package=None,processors={"tokenize":package, "pos":package, "lemma":package, "depparse":package})

        if TOKENIZED:
            nlp = Pipeline(lang=lang_name, package=package, processors='tokenize,pos,lemma,depparse', tokenize_pretokenized=True)
        else:
            nlp = Pipeline(lang=lang_name, package=package, processors='tokenize,pos,lemma,depparse', tokenize_nossplit=True)

    converter = CoNLL()

    path_baseline = BASE_FOLDER / f"data/{lang}/test/{lang}-stanza_tok{int(TOKENIZED)}_predictions.conllu"
    path_treebank = BASE_FOLDER / f"data/{lang}/test/{lang}-ud-test.conllu"

    with open(str(path_baseline), "w", encoding="utf-8") as f:

        if TOKENIZED:

            for ind, new_sent_token_list in enumerate(new_tokens):

                doc = nlp([new_sent_token_list])
                conllu_str = converter.doc2conll_text(doc)

                adj_conllu_str = adjust_the_labels(conllu_str=conllu_str, true_tokens=tokens[ind])

                f.write(adj_conllu_str)
                f.write('\n\n')

        else:

            for sent in sents:
                doc = nlp(sent)
                conllu_str = converter.doc2conll_text(doc)

                f.write(conllu_str)

    evaluator = evaluate_wrapper(EvaluatorInput(str(path_treebank), str(path_baseline)))
    las = 100 * evaluator["LAS"].f1
    uas = 100 * evaluator["UAS"].f1
    clas = 100 * evaluator["CLAS"].f1

    print("\n")
    print(f"Baseline {lang}")
    print(f"LAS: {las}")
    print(f"UAS: {uas}")
    print(f"CLAS: {clas}")
    print("\n")

if __name__ == "__main__":
    get_baseline_values(lang="lt_hse")
    get_baseline_values(lang="be_hse")
    get_baseline_values(lang="mr_ufal")
    get_baseline_values(lang="ta_ttb")

    # import numpy as np
    #
    # res = np.load(str(BASE_FOLDER / "svms/mr_ufal/cd/mr_ufal_svm_hyp_search_goldFalse.npy"), allow_pickle=True)
    # print(res == np.amax(res))
    #
    #
    # las = np.load(str(BASE_FOLDER / "svms/mr_ufal/cd/mr_ufal_emp_stanz_ratios.npy"), allow_pickle=True)
    # print(len(las))
    # print(las)



