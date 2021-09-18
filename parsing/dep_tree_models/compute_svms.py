from parsing.dep_tree_models.svms.svm_model import optimize_and_predict_with_svm, tune_reg_and_k, run_svm
from parsing.dep_tree_models.helpers.data_preparation import load_data_set
from joblib import Parallel, delayed
from pathlib import Path
from itertools import product
import numpy as np

def compute_svms():

    combinations = list(product(["augm_cd", "cd"], ["lt_hse", "be_hse", "mr_ufal", "ta_ttb"]))

    # Set up folder structure
    for kernel, lang in combinations:
        path_folder = Path.cwd() / f"{lang}/{kernel}"
        path_folder.mkdir(parents=True, exist_ok=True)

    res = Parallel(n_jobs=len(combinations))(
            delayed(optimize_and_predict_with_svm)(lang=lang, kernel=kernel, incl_gold=False)
            for kernel, lang in combinations)

    print(res)

def get_prediction_for_every_k(lang, kernel, incl_gold=False):

    BASE_FOLDER = Path.cwd()

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

    path_folder = BASE_FOLDER / f"svms/{lang}/{kernel}"
    path_folder.mkdir(parents=True, exist_ok=True)

    path = path_folder / f"{lang}_svm_hyp_search_gold{incl_gold}.npy"

    print(path)

    if path.is_file():
        res = np.load(str(path), allow_pickle=True)

    else:
        res = tune_reg_and_k(lang=lang, X_train=X_train, y_train=y_train,
                             gram_train=gram_train, kernel=kernel, incl_gold=incl_gold)

    res = res.reshape(4, 15)

    print(res)
    print(res[:,1:])
    # For every column take the max
    max_c_inds = np.argmax(res[:,1:], axis=0) # for k=1, it is clear that it is 1

    print(max_c_inds)

    c_grid = np.logspace(-5, 2.3, num=4, endpoint=True)

    las_vals = np.empty(len(max_c_inds))
    uas_vals = np.empty(len(max_c_inds))
    clas_vals = np.empty(len(max_c_inds))
    empirical_stanza_ratios = np.empty(len(max_c_inds))

    for ind, max_c_ind in enumerate(max_c_inds):

        max_c = c_grid[max_c_ind]
        k = ind + 2

        las, uas, clas, emp_stanza_ratio = run_svm(lang=lang, X_train=X_train, y_train=y_train, gram_train=gram_train,
                                                   max_c=max_c, max_k=k, max_thresh=0, kernel=kernel, incl_gold=False)

        las_vals[ind] = las
        uas_vals[ind] = uas
        clas_vals[ind] = clas
        empirical_stanza_ratios[ind] = emp_stanza_ratio

        np.save(str(path_folder / f"{lang}_las_scores_test.npy"), las_vals)
        np.save(str(path_folder / f"{lang}_uas_scores_test.npy"), uas_vals)
        np.save(str(path_folder / f"{lang}_clas_scores_test.npy"), clas_vals)
        np.save(str(path_folder / f"{lang}_emp_stanz_ratios_test.npy"), empirical_stanza_ratios)

    np.save(str(path_folder / f"{lang}_las_scores_test.npy"), las_vals)
    np.save(str(path_folder / f"{lang}_uas_scores_test.npy"), uas_vals)
    np.save(str(path_folder / f"{lang}_clas_scores_test.npy"), clas_vals)
    np.save(str(path_folder / f"{lang}_emp_stanz_ratios_test.npy"), empirical_stanza_ratios)


if __name__ == "__main__":
    compute_svms()
    for lang in ["mr_ufal", "lt_hse"]:
        get_prediction_for_every_k(lang=lang, kernel="cd", incl_gold=False)