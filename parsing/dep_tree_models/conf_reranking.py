from pathlib import Path
from parsing.old_dep_tree_models.data_preparation import load_data_set, load_ud_data
from parsing.dep_tree_models.helpers.conll18_ud_eval import evaluate_wrapper
from parsing.dep_tree_models.helpers.eval_metrics import dump_predictions
from parsing.dep_tree_models.helpers.performance_eval import EvaluatorInput
import numpy as np

BASE_FOLDER = Path.cwd()

def compute_pred_with_thresh(lang, eval_set, all_k=False, kernel="augm_cd", gold=False): #conf_thresh,

    num_k = 15
    num_seeds = 5
    conf_threshs = [0, 0.3, 0.7, 1] #[0]

    print(conf_threshs)

    if eval_set == "dev":
        start_k = 2
        num_k -= 1
    else:
        start_k = 1

    true_trees = np.load(str(BASE_FOLDER / f"data/{lang}/{eval_set}/{lang}-true_dep_trees_{eval_set}.npy"),
                         allow_pickle=True)

    las_mat = np.empty((len(conf_threshs), num_k, num_seeds))

    for thresh_ind, conf_thresh in enumerate(conf_threshs):

        for k in range(start_k, num_k + start_k):

            print(f"k: {k}")

            cand_trees = np.load(
                str(BASE_FOLDER / f"data/{lang}/{eval_set}/cands/{lang}-{k}_best_cand_dep_trees_{eval_set}.npy"),
                allow_pickle=True)

            X_eval, y_eval, eval_tokens = load_data_set(lang=lang, data_part=eval_set, k=k, tokenized=True)

            path_treebank = BASE_FOLDER / f"data/{lang}/{eval_set}/{lang}-ud-{eval_set}.conllu"

            for internal_seed in range(1, num_seeds + 1):

                name = f"experimental_{k}_{internal_seed}"

                # if kernel == "augm":
                #     preds_path = BASE_FOLDER / f"gp/non_gold/{lang}/{eval_set}/preds/{lang}_preds_with_{k}_cand_on_{eval_set}_with_{internal_seed}_gold{gold}.npy"
                # elif kernel == "cd":
                preds_path = BASE_FOLDER / f"gps/non_gold/{lang}/{kernel}/{eval_set}/preds/{lang}_preds_with_{k}_cand_on_{eval_set}_with_{internal_seed}_gold{gold}.npy"

                final_preds = np.load(str(preds_path), allow_pickle=True)

                emp_stanz = dump_predictions(lang=lang, y_pred_prob=final_preds, cand_trees=cand_trees,
                                             true_trees=true_trees, tokens=eval_tokens,
                                             name=name, path=BASE_FOLDER / f"data/{lang}/{eval_set}/cands/files",
                                             conf_thresh=conf_thresh, gold_tree_incl=False)

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

                las_mat[thresh_ind, (k - start_k), internal_seed - 1] = 100 * evaluator["LAS"].f1

    mean_las_per_k_and_per_thres = np.median(las_mat, axis=2) - np.std(las_mat, axis=2)#   #

    if not all_k:

        print(mean_las_per_k_and_per_thres)

        vals = mean_las_per_k_and_per_thres == np.amax(mean_las_per_k_and_per_thres)

        row_indices, col_indices = vals.nonzero()

        print(row_indices)
        print(col_indices)

        max_conf_ind = 0
        max_k_ind = None
        for conf_ind, k_ind in zip(row_indices, col_indices):
            if conf_ind >= max_conf_ind:
                max_conf_ind = conf_ind
                max_k_ind = k_ind

        if max_k_ind is None and max_conf_ind == 0:
            max_k_ind = min([k for thresh, k in zip(row_indices, col_indices) if thresh==0])


        print("FINAL")
        print(max_conf_ind)
        print(vals)

        print("Start K")

        thresh_ind = max_conf_ind

        threshs = [conf_threshs[thresh_ind]]
        max_ks = [(max_k_ind+start_k)]

    else:

        highest_threshold_inds_per_k = np.argmax(mean_las_per_k_and_per_thres, axis=0)
        print(highest_threshold_inds_per_k)

        threshs = [conf_threshs[ind] for ind in highest_threshold_inds_per_k]
        max_ks = [i for i in range(2, 16)]

        print(threshs)
        print(max_ks)

    return threshs, max_ks


def main(lang, eval_set, num_seeds, kernel, gold=False, all_k=False):

    true_trees = np.load(str(BASE_FOLDER / f"data/{lang}/{eval_set}/{lang}-true_dep_trees_{eval_set}.npy"),
                         allow_pickle=True)

    # if optimize:
    max_conf_thresh, max_ks = compute_pred_with_thresh(lang=lang, eval_set="dev", kernel=kernel, all_k=all_k)

    if not all_k:
        print(f"Final thres:{max_conf_thresh[0]}, k:{max_ks[0]}")

    iterator = zip(max_conf_thresh, max_ks)

    las = []
    uas = []
    clas = []
    emp_stanz_ratios = []

    for conf_thresh, k in iterator:

        cand_trees = np.load(
                    str(BASE_FOLDER / f"data/{lang}/{eval_set}/cands/{lang}-{k}_best_cand_dep_trees_{eval_set}.npy"),
                    allow_pickle=True)

        X_eval, y_eval, eval_tokens = load_data_set(lang=lang, data_part=eval_set, k=k, tokenized=True)

        path_treebank = BASE_FOLDER / f"data/{lang}/{eval_set}/{lang}-ud-{eval_set}.conllu"

        las_res = np.empty(num_seeds)
        uas_res = np.empty(num_seeds)
        clas_res = np.empty(num_seeds)
        emp_stanz_ratios_res = np.empty(num_seeds)

        for internal_seed in range(1, num_seeds + 1):

                name = f"experimental_{kernel}_{k}_{internal_seed}"

                path_folder = BASE_FOLDER / f"data/{lang}/{eval_set}/cands/files"
                path_folder.mkdir(exist_ok=True,parents=True)
                # if kernel == "augm_cd":
                #     preds_path = BASE_FOLDER / f"gps/non_gold/{lang}/{kernel}/{eval_set}/preds/{lang}_preds_with_{k}_cand_on_{eval_set}_with_{internal_seed}_gold{gold}.npy"
                # elif kernel == "cd":
                preds_path = BASE_FOLDER / f"gps/non_gold/{lang}/{kernel}/{eval_set}/preds/{lang}_preds_with_{k}_cand_on_{eval_set}_with_{internal_seed}_gold{gold}.npy"

                final_preds = np.load(str(preds_path), allow_pickle=True)

                emp_stanz = dump_predictions(lang=lang, y_pred_prob=final_preds, cand_trees=cand_trees,
                                     true_trees=true_trees, tokens=eval_tokens,
                                     name=name, path=BASE_FOLDER / f"data/{lang}/{eval_set}/cands/files",
                                     conf_thresh=conf_thresh, gold_tree_incl=False)

                path_pred = BASE_FOLDER / f"data/{lang}/{eval_set}/cands/files/{lang}-{name}-predictions.conllu"

                evaluator = evaluate_wrapper(EvaluatorInput(str(path_treebank), str(path_pred)))

                las_res[internal_seed - 1] = 100 * evaluator["LAS"].f1
                uas_res[internal_seed - 1] = 100 * evaluator["UAS"].f1
                clas_res[internal_seed-1] = 100 * evaluator["CLAS"].f1
                emp_stanz_ratios_res = emp_stanz

        las.append(np.mean(las_res))
        uas.append(np.mean(uas_res))
        clas.append(np.mean(clas_res))
        emp_stanz_ratios.append(np.mean(emp_stanz_ratios_res))

        if all_k:

            np.save(str(BASE_FOLDER / f"gps/non_gold/{lang}/{kernel}/{eval_set}/{kernel}_thresh_las_all.npy"), las)
            np.save(str(BASE_FOLDER / f"gps/non_gold/{lang}/{kernel}/{eval_set}/{kernel}_thresh_uas_all.npy"), uas)
            np.save(str(BASE_FOLDER / f"gps/non_gold/{lang}/{kernel}/{eval_set}/{kernel}_thresh_clas_all.npy"), clas)
            np.save(str(BASE_FOLDER / f"gps/non_gold/{lang}/{kernel}/{eval_set}/{kernel}_thresh_emp_stanz_ratios_all.npy"), emp_stanz_ratios)

        print(las_res)
        print(np.std(las_res, ddof=1))

        print(f"Mean LAS for k: {k} on test", np.mean(las_res))
        print(f"Median LAS for k: {k} on test", np.median(las_res))
        print(f"Std LAS for k: {k} on test", np.std(las_res, ddof=1))

        print(f"Mean UAS for k: {k} on test", np.mean(uas_res))
        print(f"Median UAS for k: {k} on test", np.median(uas_res))
        print(f"Std UAS for k: {k} on test", np.std(uas_res, ddof=1))

        print(f"Mean CLAS for k: {k} on test", np.mean(clas_res))
        print(f"Median CLAS for k: {k} on test", np.median(clas_res))
        print(f"Std CLAS for k: {k} on test", np.std(clas_res, ddof=1))

if __name__ == "__main__":
    main(lang="lt_hse", eval_set="test", num_seeds=5, kernel="cd")

