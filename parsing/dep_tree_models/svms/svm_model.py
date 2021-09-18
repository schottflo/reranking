from pathlib import Path
from parsing.dep_tree_models.helpers.data_preparation import load_data_set, load_data_set_incl_gold, load_ud_data
import numpy as np
from sklearn.svm import SVC

from scipy.stats import norm
from parsing.dep_tree_models.helpers.performance_eval import EvaluatorInput
from parsing.dep_tree_models.helpers.conll18_ud_eval import evaluate_wrapper
from parsing.dep_tree_models.helpers.eval_metrics import dump_predictions

BASE_FOLDER = Path.cwd()

def get_best_param_comb(dev_las):

    avg_dict = {key: val[0] for key, val in dev_las.items()}
    max_c = max(avg_dict, key=avg_dict.get)

    return max_c

def tune_reg_and_k(lang, X_train, y_train, gram_train, kernel, incl_gold=False):

    if kernel == "cd":
        from parsing.dep_tree_models.svms.cd_svm import CustomKernel
    elif kernel == "augm_cd":
        from parsing.dep_tree_models.svms.augm_cd_svm import CustomKernel

    print(kernel)

    eval_set = "dev"
    model_type= f"SVM_{kernel}"
    internal_seed= "none"
    conf_threshs = [0] # actually not needed

    # Set up the parameter grid
    k_grid = np.array([k for k in range(1,16)])
    c_grid = np.logspace(-5, 2.3, num=4, endpoint=True)

    res = np.empty((len(c_grid), len(conf_threshs), len(k_grid)))

    for c_ind, c in enumerate(c_grid):

        print("Current c:", c)

        model = SVC(C=c, kernel="precomputed", random_state=42)
        model.fit(gram_train, y_train)

        for ind_conf, conf_thresh in enumerate(conf_threshs):

            for k_ind, k in enumerate(k_grid):

                print("Current k:", k)

                if incl_gold:
                    X_eval, y_eval, eval_tokens = load_data_set_incl_gold(lang=lang, k=k, data_part="dev", tokenized=True)
                else:
                    X_eval, y_eval, eval_tokens = load_data_set(lang=lang, data_part="dev", k=k, tokenized=True)

                if kernel == "cd":
                    gram_eval = CustomKernel(X=X_eval, X2=X_train, lamb=0.7012455615069223, emb_scale=0.8847221539035289)
                elif kernel == "augm_cd":
                    gram_eval = CustomKernel(X=X_eval, X2=X_train, lamb=0.7012455615069223)

                # Predict on dev set
                y_dec = model.decision_function(gram_eval)

                final_pred = norm.cdf(y_dec)

                true_trees = np.load(str(BASE_FOLDER / f"data/{lang}/{eval_set}/{lang}-true_dep_trees_{eval_set}.npy"),
                                     allow_pickle=True)

                cand_trees = np.load(
                    str(BASE_FOLDER / f"data/{lang}/{eval_set}/cands/{lang}-{k}_best_cand_dep_trees_{eval_set}.npy"),
                    allow_pickle=True)

                name = f"{kernel}_kern_{model_type}_with_{k}_cand_on_{eval_set}_with_{internal_seed}"

                if incl_gold:
                    cand_trees = np.load(str(
                        BASE_FOLDER / f"data/{lang}/{eval_set}/cands/{lang}-{k}_best_cand_dep_trees_{eval_set}-gtTrue.npy"),
                                         allow_pickle=True)
                    name = f"{model_type}_with_{k}_cand_on_{eval_set}_with_{internal_seed}_gold_incl"

                path_pred_folder = BASE_FOLDER / f"data/{lang}/{eval_set}/cands/files"
                path_pred_folder.mkdir(parents=True, exist_ok=True)

                if incl_gold:
                    empirical_stanza_ratio = dump_predictions(lang=lang, y_pred_prob=final_pred, cand_trees=cand_trees,
                                                              true_trees=true_trees,
                                                              tokens=eval_tokens, name=name, conf_thresh=conf_thresh,
                                                              path=BASE_FOLDER / f"data/{lang}/{eval_set}/cands/files",
                                                              gold_tree_incl=True)
                else:
                    empirical_stanza_ratio = dump_predictions(lang=lang, y_pred_prob=final_pred, cand_trees=cand_trees,
                                                              true_trees=true_trees,
                                                              tokens=eval_tokens, name=name, conf_thresh=conf_thresh,
                                                              path=BASE_FOLDER / f"data/{lang}/{eval_set}/cands/files",
                                                              gold_tree_incl=False)

                path_treebank = BASE_FOLDER / f"data/{lang}/{eval_set}/{lang}-ud-{eval_set}.conllu"
                path_pred = BASE_FOLDER / f"data/{lang}/{eval_set}/cands/files/{lang}-{name}-predictions.conllu"

                if lang == "lt_hse" and eval_set == "dev":

                    sents, tokens, dep_graph_strs = load_ud_data(lang=lang, data_part=eval_set)

                    # Save a modified version of the treebank file, in order to take right decisions
                    path_treebank = BASE_FOLDER / f"data/{lang}/{eval_set}/{lang}-ud-{eval_set}-mod.conllu"

                    with open(str(path_treebank), "w", encoding="utf-8") as f:

                        for ind, dep_graph_str in enumerate(dep_graph_strs):

                            # output = transform_into_conllu(tree)
                            if ind in [16, 28]:
                                continue

                            f.write(dep_graph_str)

                evaluator = evaluate_wrapper(EvaluatorInput(str(path_treebank), str(path_pred)))

                las = 100 * evaluator["LAS"].f1

                res[c_ind, ind_conf, k_ind] = las

    print(res)
    np.save(BASE_FOLDER / f"svms/{lang}/{kernel}/{lang}_svm_hyp_search_gold{incl_gold}.npy", res)

    return res

def run_svm(lang, X_train, y_train, gram_train, max_c, max_thresh, max_k, kernel, incl_gold=False):

    if kernel == "cd":
        from parsing.dep_tree_models.svms.cd_svm import CustomKernel
    elif kernel == "augm_cd":
        from parsing.dep_tree_models.svms.augm_cd_svm import CustomKernel

    # Fit the best model
    final_model = SVC(C=float(max_c), kernel="precomputed", random_state=42)
    final_model.fit(gram_train, y_train)

    k = max_k
    X_eval, y_eval, eval_tokens = load_data_set(lang=lang, data_part="test", k=k, tokenized=True)

    if kernel == "cd":
        gram_eval = CustomKernel(X=X_eval, X2=X_train, lamb=0.7012455615069223)
    elif kernel == "augm_cd":
        gram_eval = CustomKernel(X=X_eval, X2=X_train, lamb=0.7012455615069223, emb_scale=0.8847221539035289)

    # Predict on dev set
    y_dec_final = final_model.decision_function(gram_eval)

    final_pred = norm.cdf(y_dec_final)

    eval_set = "test"
    model_type = f"SVM_{kernel}"
    internal_seed = "none"

    conf_thresh = max_thresh

    true_trees = np.load(str(BASE_FOLDER / f"data/{lang}/{eval_set}/{lang}-true_dep_trees_{eval_set}.npy"),
                         allow_pickle=True)

    cand_trees = np.load(
        str(BASE_FOLDER / f"data/{lang}/{eval_set}/cands/{lang}-{k}_best_cand_dep_trees_{eval_set}.npy"),
        allow_pickle=True)

    name = f"easy_kern_{model_type}_with_{k}_cand_on_{eval_set}_with_{internal_seed}"

    if incl_gold:
        cand_trees = np.load(str(
            BASE_FOLDER / f"data/{lang}/{eval_set}/cands/{lang}-{k}_best_cand_dep_trees_{eval_set}-gtTrue.npy"),
            allow_pickle=True)
        name = f"{model_type}_with_{k}_cand_on_{eval_set}_with_{internal_seed}_gold_incl"

    path_pred_folder = BASE_FOLDER / f"data/{lang}/{eval_set}/cands/files"
    path_pred_folder.mkdir(parents=True, exist_ok=True)

    if incl_gold:
        empirical_stanza_ratio = dump_predictions(lang=lang, y_pred_prob=final_pred, cand_trees=cand_trees,
                                                  true_trees=true_trees,
                                                  tokens=eval_tokens, name=name, conf_thresh=conf_thresh,
                                                  path=BASE_FOLDER / f"data/{lang}/{eval_set}/cands/files",
                                                  gold_tree_incl=True)
    else:
        empirical_stanza_ratio = dump_predictions(lang=lang, y_pred_prob=final_pred, cand_trees=cand_trees,
                                                  true_trees=true_trees,
                                                  tokens=eval_tokens, name=name, conf_thresh=conf_thresh,
                                                  path=BASE_FOLDER / f"data/{lang}/{eval_set}/cands/files",
                                                  gold_tree_incl=False)

    path_treebank = BASE_FOLDER / f"data/{lang}/{eval_set}/{lang}-ud-{eval_set}.conllu"
    path_pred = BASE_FOLDER / f"data/{lang}/{eval_set}/cands/files/{lang}-{name}-predictions.conllu"

    evaluator = evaluate_wrapper(EvaluatorInput(str(path_treebank), str(path_pred)))

    las = 100 * evaluator["LAS"].f1
    uas = 100 * evaluator["UAS"].f1
    clas = 100 * evaluator["CLAS"].f1

    return las, uas, clas, empirical_stanza_ratio

def optimize_and_predict_with_svm(lang, kernel, incl_gold, predict_for_every_k=False):

    print(f"SVM fitting for {lang} and {kernel}_kernel starts")

    if kernel == "cd":
        from parsing.dep_tree_models.svms.cd_svm import CustomKernel
    elif kernel == "augm_cd":
        from parsing.dep_tree_models.svms.augm_cd_svm import CustomKernel

    # Get training data
    X_train, y_train = load_data_set(lang=lang, data_part="train", k=10, tokenized=True)

    # Construct gram matrix
    if kernel == "cd":
        gram_train = CustomKernel(X=X_train, lamb=0.7012455615069223)
    elif kernel == "augm_cd":
        gram_train = CustomKernel(X=X_train, lamb=0.7012455615069223, emb_scale=0.8847221539035289)

    path_folder = BASE_FOLDER / f"{lang}/{kernel}"
    path_folder.mkdir(parents=True, exist_ok=True)

    path = path_folder / f"{lang}_svm_hyp_search_gold{incl_gold}.npy"

    if path.is_file():
        res = np.load(str(path), allow_pickle=True)

    else:
        res = tune_reg_and_k(lang=lang, X_train=X_train, y_train=y_train,
                                                      gram_train=gram_train, kernel=kernel, incl_gold=incl_gold)

    res = res.reshape(4, 15) # could be more flexible here (c_grid, k_grid)

    print(res)

    c_grid = np.logspace(-5, 2.3, num=4, endpoint=True)
    vals = res == np.amax(res)

    row_indices, col_indices = vals.nonzero()

    max_reg_ind = (len(c_grid) - 1)
    max_k_ind = None
    for reg_ind, k_ind in zip(row_indices, col_indices):
        if reg_ind <= max_reg_ind:
            max_reg_ind = reg_ind
            max_k_ind = k_ind

    print(max_reg_ind)
    print(max_k_ind)

    best_c = c_grid[max_reg_ind]  # np.array([(i + 1, c_grid[ind], max(res[:, i])) for i, ind in enumerate(best_c_ind)])
    best_k = max_k_ind + 1

    print(c_grid)

    print(best_c)
    print(best_k)


    # Either with or without gold_tree
    return run_svm(lang=lang, X_train=X_train, y_train=y_train, gram_train=gram_train, max_c=best_c, max_k=best_k,
            max_thresh=0, kernel=kernel, incl_gold=False)

if __name__ == "__main__":
    optimize_and_predict_with_svm(lang="lt_hse", kernel="augm_cd", incl_gold=False)