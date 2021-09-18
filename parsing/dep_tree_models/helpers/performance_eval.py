from random import seed
import numpy as np
from pathlib import Path
import os

from parsing.dep_tree_models.helpers.data_preparation import load_data_set, load_ud_data, load_data_set_incl_gold
from parsing.dep_tree_models.helpers.eval_metrics import dump_predictions
from parsing.dep_tree_models.helpers.conll18_ud_eval import evaluate_wrapper

from parsing.dep_tree_models.gps.augm_cd_kernel import initialize_kernel as initialize_kernel_augm_cd
from parsing.dep_tree_models.gps.cd_kernel import initialize_kernel as initialize_kernel_cd

from GPy.models import GPClassification
from GPy.inference.latent_function_inference import EP

LANG = "lt_hse"
BASE_FOLDER = Path.cwd()
K = 15

class EvaluatorInput():
    """An object that can be passed into the evaluate function of the official CoNLL18 evaluation script"""
    def __init__(self, gold_file, system_file):
        self.gold_file = gold_file
        self.system_file = system_file


def evaluate_predictions(lang, final_pred, k, eval_set, eval_tokens, kernel, internal_seed, include_gold_tree=False):

    true_trees = np.load(str(BASE_FOLDER / f"data/{lang}/{eval_set}/{lang}-true_dep_trees_{eval_set}.npy"),
                         allow_pickle=True)

    cand_trees = np.load(
        str(BASE_FOLDER / f"data/{lang}/{eval_set}/cands/{lang}-{k}_best_cand_dep_trees_{eval_set}.npy"),
        allow_pickle=True)

    name = f"{kernel}_with_{k}_cand_on_{eval_set}_with_{internal_seed}"

    if include_gold_tree:
        cand_trees = np.load(str(BASE_FOLDER / f"data/{lang}/{eval_set}/cands/{lang}-{k}_best_cand_dep_trees_{eval_set}-gtTrue.npy"),
                         allow_pickle=True)
        name = f"{kernel}_with_{k}_cand_on_{eval_set}_with_{internal_seed}_gold_incl"

    path_pred_folder = BASE_FOLDER / f"data/{lang}/{eval_set}/cands/files"
    path_pred_folder.mkdir(parents=True, exist_ok=True)

    if include_gold_tree:
        empirical_stanza_ratio = dump_predictions(lang=lang, y_pred_prob=final_pred, cand_trees=cand_trees, true_trees=true_trees,
                     tokens=eval_tokens, name=name,
                     path=BASE_FOLDER / f"data/{lang}/{eval_set}/cands/files", gold_tree_incl=True)
    else:
        empirical_stanza_ratio = dump_predictions(lang=lang, y_pred_prob=final_pred, cand_trees=cand_trees, true_trees=true_trees,
                         tokens=eval_tokens, name=name,
                         path=BASE_FOLDER / f"data/{lang}/{eval_set}/cands/files", gold_tree_incl=False)

    path_treebank = BASE_FOLDER / f"data/{lang}/{eval_set}/{lang}-ud-{eval_set}.conllu"
    path_pred = BASE_FOLDER / f"data/{lang}/{eval_set}/cands/files/{lang}-{name}-predictions.conllu"

    if lang == "lt_hse" and eval_set == "dev":

        sents, tokens, dep_graph_strs = load_ud_data(lang=lang, data_part=eval_set)

        # Save a modified version of the treebank file, in order to take right decisions
        path_treebank = BASE_FOLDER / f"data/{lang}/{eval_set}/{lang}-ud-{eval_set}-mod.conllu"

        with open(str(path_treebank), "w", encoding="utf-8") as f:

            for ind, dep_graph_str in enumerate(dep_graph_strs):

                if ind in [16, 28]:
                    continue

                f.write(dep_graph_str)

    evaluator = evaluate_wrapper(EvaluatorInput(str(path_treebank), str(path_pred)))

    las = 100 * evaluator["LAS"].f1
    uas = 100 * evaluator["UAS"].f1
    clas = 100 * evaluator["CLAS"].f1

    return las, uas, clas, empirical_stanza_ratio

def evaluate_model(lang, X_train, y_train, eval_set, k, kernel, internal_seed, save_preds=False, include_gold_tree=False):

    seed(internal_seed)
    np.random.seed(internal_seed)
    os.environ["PYTHONHASHSEED"] = str(internal_seed)

    # Extract the development set
    if include_gold_tree:
        X_eval, y_eval, eval_tokens = load_data_set_incl_gold(lang=lang, data_part=eval_set, k=k, tokenized=True)
    else:
        X_eval, y_eval, eval_tokens = load_data_set(lang=lang, data_part=eval_set, k=k,
                                                tokenized=True)

    # Initialize the kernel
    if kernel == "cd":
        kernel_func = initialize_kernel_cd(num_col=X_train.shape[1], internal_seed=internal_seed)
    elif kernel == "augm_cd":
        kernel_func = initialize_kernel_augm_cd(num_col=X_train.shape[1], internal_seed=internal_seed)

    # Train the model
    k_specific_model = GPClassification(X=X_train, Y=y_train.reshape(-1, 1), kernel=kernel_func,
                                        inference_method=EP(parallel_updates=True))

    print("Model fitted; Sampling starts now")

    # Produce 100 samples from the predictive posterior of the latent function and push them through the link function
    latent_function_samples = k_specific_model.posterior_samples_f(X=X_eval, size=150).reshape((X_eval.shape[0], 150)) # stochastic # k_specific_model
    pred = k_specific_model.likelihood.gp_link.transf(latent_function_samples)# k_specific_model

    final_pred = np.mean(pred, axis=1)

    if save_preds:

        if include_gold_tree:
            pred_path = BASE_FOLDER / f"gps/gold/{lang}/{kernel}/{eval_set}/preds"
        else:
            pred_path = BASE_FOLDER / f"gps/non_gold/{lang}/{kernel}/{eval_set}/preds"

        #pred_path.mkdir(parents=True, exist_ok=True)
        np.save(str(pred_path / f"{lang}_preds_with_{kernel}_{k}_cand_on_{eval_set}_with_{internal_seed}_gold{include_gold_tree}.npy"), final_pred)

    return evaluate_predictions(lang=lang, final_pred=final_pred, k=k, eval_set=eval_set, eval_tokens=eval_tokens,
                               kernel=kernel, internal_seed=internal_seed, include_gold_tree=include_gold_tree)


def compare_performance(lang, X_train, y_train, eval_set, max_k, kernel, internal_seed, include_gold_tree=False):

    uas_scores = np.empty(max_k)
    las_scores = np.empty(max_k)
    clas_scores = np.empty(max_k)
    empirical_stanza_ratios = np.empty(max_k)

    for k in range(1,max_k+1):

        if include_gold_tree:
            las, uas, clas, emp_stanz = evaluate_model(lang=lang, X_train=X_train, y_train=y_train, eval_set=eval_set,
                                                       k=k, kernel=kernel, internal_seed=internal_seed, save_preds=True,
                                                       include_gold_tree=True)
        else:
            las, uas, clas, emp_stanz = evaluate_model(lang=lang, X_train=X_train, y_train=y_train, eval_set=eval_set,
                                                        k=k, kernel=kernel, internal_seed=internal_seed,
                                                        save_preds=True,include_gold_tree=False)

        print("Seed:", internal_seed)

        print(f"LAS for {lang} with k:{k}: ", las)
        print(f"UAS for {lang} with k:{k}: ", uas)
        print(f"CLAS for {lang} with k:{k}: ", clas)
        print(f"Empirical Stanza Ratio {lang} with k:{k}: ", emp_stanz)

        las_scores[k-1] = las
        uas_scores[k-1] = uas
        clas_scores[k-1] = clas
        empirical_stanza_ratios[k-1] = emp_stanz

        #Save to be safe
        if include_gold_tree:
            np.save(str(BASE_FOLDER / f"gps/gold/{lang}/{kernel}/{eval_set}/{lang}_las_scores_{eval_set}_{internal_seed}_gold.npy"), las_scores)
            np.save(str(BASE_FOLDER / f"gps/gold/{lang}/{kernel}/{eval_set}/{lang}_uas_scores_{eval_set}_{internal_seed}_gold.npy"), uas_scores)
            np.save(str(BASE_FOLDER / f"gps/gold/{lang}/{kernel}/{eval_set}/{lang}_clas_scores_{eval_set}_{internal_seed}_gold.npy"),clas_scores)
            np.save(str(BASE_FOLDER / f"gps/gold/{lang}/{kernel}/{eval_set}/{lang}_emp_stanz_ratios_{eval_set}_{internal_seed}_gold.npy"),
                    empirical_stanza_ratios)

        else:
            np.save(str(BASE_FOLDER / f"gps/non_gold/{lang}/{kernel}/{eval_set}/{lang}_las_scores_{eval_set}_{internal_seed}.npy"), las_scores)
            np.save(str(BASE_FOLDER / f"gps/non_gold/{lang}/{kernel}/{eval_set}/{lang}_uas_scores_{eval_set}_{internal_seed}.npy"), uas_scores)
            np.save(str(BASE_FOLDER / f"gps/non_gold/{lang}/{kernel}/{eval_set}/{lang}_clas_scores_{eval_set}_{internal_seed}.npy"),clas_scores)
            np.save(str(BASE_FOLDER / f"gps/non_gold/{lang}/{kernel}/{eval_set}/{lang}_emp_stanz_ratios_{eval_set}_{internal_seed}.npy"),
                    empirical_stanza_ratios)

    return las_scores, uas_scores, clas_scores, empirical_stanza_ratios