from pathlib import Path
from joblib import Parallel, delayed

from parsing.dep_tree_models.helpers.data_preparation import load_data_set
from parsing.dep_tree_models.helpers.performance_eval import compare_performance

BASE_FOLDER = Path.cwd()
MAX_K = 15

def compute_gp_model(lang, kernel, num_seeds):

    path_test_folder = Path.cwd() / f"gps/non_gold/{lang}/{kernel}/test/preds"
    path_test_folder.mkdir(parents=True, exist_ok=True)

    path_dev_folder = Path.cwd() / f"gps/non_gold/{lang}/{kernel}/dev/preds"
    path_dev_folder.mkdir(parents=True, exist_ok=True)


    for internal_seed in range(1, num_seeds+1):
        # Extract the training set (hyperparameters: k=9)
        X_train, y_train = load_data_set(lang=lang, data_part="train", k=10, tokenized=True)

        test_results, dev_results = Parallel(n_jobs=2)(
            delayed(compare_performance)(lang=lang, X_train=X_train, y_train=y_train, eval_set=eval_set, kernel=kernel,
                                         internal_seed=internal_seed, max_k=MAX_K) for eval_set in ["test", "dev"])

        print(f"Dev for {lang} with {kernel}:", dev_results)
        print(f"Test for {lang} with {kernel}:", test_results)

def compute_gp_models():

    for lang in ["ta_ttb", "lt_hse", "be_hse", "mr_ufal"]:
        for kernel in ["cd", "augm_cd"]:
            compute_gp_model(lang=lang, kernel=kernel, num_seeds=5)

if __name__ == "__main__":
    compute_gp_models()