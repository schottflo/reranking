from pathlib import Path
from joblib import Parallel, delayed
from parsing.dep_tree_models.helpers.data_preparation import load_data_set
from parsing.dep_tree_models.helpers.performance_eval import compare_performance

BASE_FOLDER = Path.cwd()

def compute_gold_performance(lang):

    for kernel in ["cd", "augm_cd"]:

        path_test_folder = Path.cwd() / f"gps/gold/{lang}/{kernel}/test/preds"
        path_test_folder.mkdir(parents=True, exist_ok=True)

        # Extract the training set (hyperparameters: k=10)
        X_train, y_train = load_data_set(lang=lang, data_part="train", k=10, tokenized=True)

        gold_results = Parallel(n_jobs=5)(
            delayed(compare_performance)(lang=lang, X_train=X_train, y_train=y_train, eval_set="test",
                                        max_k=15, include_gold_tree=True, kernel=kernel,
                                        internal_seed=internal_seed) for internal_seed in range(1, 6))

        print(f"{lang} gold results:", gold_results)

if __name__ == "__main__":
    compute_gold_performance(lang="lt_hse")