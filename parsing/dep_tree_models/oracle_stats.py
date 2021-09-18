from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

from parsing.dep_tree_models.helpers.data_preparation import load_ud_data, build_candidate_dep_trees
from parsing.dep_tree_models.helpers.eval_metrics import adjust_the_labels
from parsing.dep_tree_models.helpers.performance_eval import EvaluatorInput
from parsing.dep_tree_models.helpers.conll18_ud_eval import evaluate_wrapper

WORKING_DIR = Path.cwd()
BASE_FOLDER = WORKING_DIR / "oracle_computations"
DATA_PART = "test"
CRITERION = "LAS"
K = 15

def create_individual_true_tree_conllu_files(lang, true_trees, tokens):

    path_true_trees = BASE_FOLDER / f"{lang}/{DATA_PART}/true_files"

    if path_true_trees.is_dir():
        print(f"The true files had already been created for {lang}")
        return path_true_trees

    else:
        path_true_trees.mkdir(parents=True, exist_ok=True)

    for ind, true_tree in enumerate(true_trees):

        with open(str(path_true_trees / f"{ind}.conllu"), "w", encoding="utf-8") as t:

            output = true_tree.conllu_str
            new_output = adjust_the_labels(conllu_str=output, true_tokens=tokens[ind]) # Tokens needs to be a list of (address, token) tuples

            t.write(new_output)
            t.write('\n\n')

    print(f"The true files were created for {lang}")

    return path_true_trees


def create_individual_cand_tree_conllu_files(lang, k, cand_trees, tokens):

    path_cand_trees = BASE_FOLDER / f"{lang}/{DATA_PART}/cands/{k}/files"

    if path_cand_trees.is_dir():
        print(f"The candidate files had already been created for {lang}")
        return path_cand_trees

    else:
        path_cand_trees.mkdir(parents=True, exist_ok=True)

    for ind, cand_tree_list in enumerate(cand_trees):
        for cand_ind, cand_tree in enumerate(cand_tree_list):
            with open(str(path_cand_trees / f"{ind}_{cand_ind}.conllu"), "w", encoding="utf-8") as c:

                output = cand_tree.conllu_str
                new_output = adjust_the_labels(conllu_str=output,
                                               true_tokens=tokens[ind])  # Tokens needs to be a list of (address, token) tuples

                c.write(new_output)
                c.write('\n\n')

    print(f"The candidate files were created for {lang}")
    return path_cand_trees


def is_tie(max_ind, score_per_cand_list):

    max_score = score_per_cand_list[max_ind]

    count = 0
    for sc in score_per_cand_list:

        if max_score == sc:
            count += 1

        if count > 1:
            return 1

    return 0

def get_index_of_best_cand_trees(num_sentences, path_true_trees, path_cand_trees, cand_trees, crit):

    max_inds = np.empty(num_sentences)

    tie_count = 0


    for ind in range(num_sentences):

        path_true_tree = str(path_true_trees / f"{ind}.conllu")

        score_per_cand_list = []
        for cand_ind in range(len(cand_trees[ind])):
            path_cand = str(path_cand_trees / f"{ind}_{cand_ind}.conllu")

            evaluator = evaluate_wrapper(EvaluatorInput(path_true_tree, path_cand))
            score = 100 * evaluator[crit].f1

            score_per_cand_list.append(score)  # 100 * evaluator["UAS"].f1,

        # Check if there is several with a maximal value
        ind_max_las = np.argmax(score_per_cand_list)

        tie_count += is_tie(max_ind=ind_max_las, score_per_cand_list=score_per_cand_list)

        max_inds[ind] = ind_max_las

    print("Ratio of the Stanza prediction being the best tree")
    stanza_ratio = len(max_inds[max_inds == 0]) / num_sentences
    print(stanza_ratio)

    print("Tie ratio")
    tie_ratio = tie_count / num_sentences
    print(tie_ratio)

    return max_inds, stanza_ratio, tie_ratio

def construct_best_cand_conllu_file(lang, path_k, path_true_trees, path_cand_trees, cand_trees, tokens, crit):

    # Construct a file that takes in the maximal candidates
    path_best_cands = path_k / "oracle-candidates.conllu"

    num_sentences = len(cand_trees)
    max_inds, stanza_ratio, tie_ratio = get_index_of_best_cand_trees(num_sentences, path_true_trees, path_cand_trees, cand_trees,
                                                          crit)

    if path_best_cands.is_file():
        print(f"The best candidates file had already been created for {lang}")
        return path_best_cands, stanza_ratio, tie_ratio

    with open(path_best_cands, "w", encoding="utf-8") as f:

        for ind, cand_tree_list in enumerate(cand_trees):

            best_tree = cand_tree_list[int(max_inds[ind])]
            output = best_tree.conllu_str

            new_output = adjust_the_labels(conllu_str=output,
                                           true_tokens=tokens[ind]) # Tokens needs to be a list of (address, token) tuples

            f.write(new_output)
            f.write('\n\n')

    print(f"The best candidates file were created for {lang}")
    return path_best_cands, stanza_ratio, tie_ratio

def construct_cand_score_file(lang, k, path_true_trees, path_cand_trees, true_trees, cand_trees, crit):

    # Construct a file that takes in the maximal candidates
    path_cand_score_file = BASE_FOLDER / f"{lang}/{lang}_{k}_cand_scores_{crit}.npy"
    path_num_nodes_file = BASE_FOLDER / f"{lang}/{lang}_num_nodes.npy"

    num_sentences = len(cand_trees)

    cand_scores = np.empty(num_sentences, dtype=object)
    num_nodes = np.empty(num_sentences, dtype=object)

    if path_cand_score_file.is_file():
        print(f"The candidate score file had already been created for {lang}")
        #cand_scores = np.load(str(path_cand_score_file), allow_pickle=True)
        return path_cand_score_file, path_num_nodes_file

    for ind in range(num_sentences):

        path_true_tree = str(path_true_trees / f"{ind}.conllu")

        score_per_cand_list = [100]

        print("Number of words")
        print(len(true_trees[ind].words()))

        num_nodes[ind] = len(true_trees[ind].words())

        try:
            cand_tree = cand_trees[ind][0]
        except:
            print("Skipping")
            cand_scores[ind] = score_per_cand_list
            continue

        print("Tree")
        print(cand_tree)

        print("Number of words")
        print(len(cand_tree.words()))

        print("Number of candidates")
        print(len(cand_trees[ind]))

        for cand_ind in range(len(cand_trees[ind])):

            path_cand = str(path_cand_trees / f"{ind}_{cand_ind}.conllu")

            evaluator = evaluate_wrapper(EvaluatorInput(path_true_tree, path_cand))
            score = 100 * evaluator[crit].f1

            score_per_cand_list.append(score)  # 100 * evaluator["UAS"].f1,

        cand_scores[ind] = score_per_cand_list

    print(path_cand_score_file)

    np.save(str(path_cand_score_file), cand_scores)
    np.save(str(path_num_nodes_file), num_nodes)

    return path_cand_score_file, path_num_nodes_file

def compute_oracle_scores(lang, crit):

    # Load the data
    sents, tokens, dep_graph_strs = load_ud_data(lang=lang, data_part=DATA_PART)
    true_trees = np.load(str(WORKING_DIR / f"data/{lang}/{DATA_PART}/{lang}-true_dep_trees_{DATA_PART}.npy"),
                         allow_pickle=True)
    path_true_trees = create_individual_true_tree_conllu_files(lang=lang, true_trees=true_trees, tokens=tokens)

    oracle_scores = np.empty(K)
    gold_tree_ratios = np.empty(K)
    stanza_ratios = np.empty(K)
    tie_ratios = np.empty(K)

    # Generate the true trees and candidate trees for a given k
    for k in range(1, K+1):

        path_k = BASE_FOLDER / f"{lang}/{DATA_PART}/cands/{k}"

        # Create the folder if it doesn't exist
        path_k.mkdir(parents=True, exist_ok=True)

        # Save the candidate tree file
        cand_trees, gold_tree_ratio = build_candidate_dep_trees(lang=lang, data_part=DATA_PART, tokens=tokens,
                                                                sents=sents,true_dep_trees=true_trees, k=k,
                                                                path=path_k / f"{lang}-cand_dep_trees_{DATA_PART}.npy",
                                                                tokenized=True, test_with_gold_tree=False,
                                                                return_gold_tree_ratio=True)

        gold_tree_ratios[k-1] = gold_tree_ratio

        path_cand_trees = create_individual_cand_tree_conllu_files(lang=lang, k=k, cand_trees=cand_trees, tokens=tokens)

        # Construct a file with the best candidate parses (in terms of LAS)
        path_oracle_cands, stanza_ratio, tie_ratio = construct_best_cand_conllu_file(lang=lang, path_k=path_k,
                                                                                      path_true_trees=path_true_trees,
                                                                                      path_cand_trees=path_cand_trees,
                                                                                      cand_trees=cand_trees,
                                                                                      tokens=tokens,
                                                                                      crit=crit)

        stanza_ratios[k - 1] = stanza_ratio
        tie_ratios[k - 1] = tie_ratio

        # Get the treebank file
        path_treebank = WORKING_DIR / f"data/{lang}/{DATA_PART}/{lang}-ud-{DATA_PART}.conllu"

        final_evaluator = evaluate_wrapper(EvaluatorInput(str(path_treebank), str(path_oracle_cands)))

        score = 100 * final_evaluator[crit].f1

        print(f"{k} candidates led to an oracle {crit} of {score}")

        oracle_scores[k - 1] = score

        # Independent from criterion
        np.save(BASE_FOLDER / f"{lang}/{DATA_PART}/{lang}_gold_tree_ratios.npy", gold_tree_ratios)

        # Dependent on criterion
        np.save(BASE_FOLDER / f"{lang}/{DATA_PART}/{lang}_oracle_{crit}.npy", oracle_scores)
        np.save(BASE_FOLDER / f"{lang}/{DATA_PART}/{lang}_stanza_ratios_{crit}.npy", stanza_ratios)
        np.save(BASE_FOLDER / f"{lang}/{DATA_PART}/{lang}_tie_ratios_{crit}.npy", tie_ratios)

    # Independent from criterion
    np.save(BASE_FOLDER / f"{lang}/{DATA_PART}/{lang}_gold_tree_ratios.npy", gold_tree_ratios)

    # Dependent on criterion
    np.save(BASE_FOLDER / f"{lang}/{DATA_PART}/{lang}_oracle_{crit}.npy", oracle_scores)
    np.save(BASE_FOLDER / f"{lang}/{DATA_PART}/{lang}_stanza_ratios_{crit}.npy", stanza_ratios)
    np.save(BASE_FOLDER / f"{lang}/{DATA_PART}/{lang}_tie_ratios_{crit}.npy", tie_ratios)

def compute_random_baseline(lang, max_k, crit):

    sents, tokens, dep_graph_strs = load_ud_data(lang=lang, data_part=DATA_PART)

    true_trees = np.load(str(WORKING_DIR / f"data/{lang}/{DATA_PART}/{lang}-true_dep_trees_{DATA_PART}.npy"),
                         allow_pickle=True)

    path_true_trees = create_individual_true_tree_conllu_files(lang=lang, true_trees=true_trees, tokens=tokens)

    random_baseline = np.empty(max_k)

    for k in range(1, max_k+1):

        print("k")

        path_k = BASE_FOLDER / f"{lang}/{DATA_PART}/cands/{k}"

        # Create the folder if it doesn't exist
        path_k.mkdir(parents=True, exist_ok=True)

        cand_trees = build_candidate_dep_trees(lang=lang, data_part=DATA_PART, tokens=tokens,
                                               sents=sents,true_dep_trees=true_trees, k=k,
                                               path=path_k / f"{lang}-cand_dep_trees_{DATA_PART}.npy",
                                               tokenized=True, test_with_gold_tree=True,
                                               return_gold_tree_ratio=False)

        path_cand_trees = create_individual_cand_tree_conllu_files(lang=lang, k=k, cand_trees=cand_trees, tokens=tokens)

        path_cand_scores, path_num_nodes = construct_cand_score_file(lang=lang, k=k, path_true_trees=path_true_trees,
                                                                      path_cand_trees=path_cand_trees, true_trees=true_trees,
                                                                     cand_trees=cand_trees, crit=crit)

        cand_scores = np.load(str(path_cand_scores), allow_pickle=True)
        num_nodes = np.load(str(path_num_nodes), allow_pickle=True)

        avg_cand_scores = np.empty(len(sents))
        for ind, cand_score_list in enumerate(cand_scores):
            #print(cand_score_list)

            #print(np.mean(cand_score_list))
            avg_cand_scores[ind] = np.mean(cand_score_list)

        # print("RANDOM SCORE")
        # print(np.sum(num_nodes/np.sum(num_nodes)))
        # print(avg_cand_scores)
        # print(np.dot(num_nodes/np.sum(num_nodes), avg_cand_scores))

        random_baseline[k-1] = np.dot(num_nodes/np.sum(num_nodes), avg_cand_scores)

    print(random_baseline)
    np.save(str(BASE_FOLDER / f"{lang}/test/{lang}_random_baseline.npy"), random_baseline)

    return random_baseline

def visualize_oracle_scores(lang, crit):

    path_lang_gold_tree_ratios = BASE_FOLDER / f"{lang}/{DATA_PART}/{lang}_gold_tree_ratios.npy"
    path_lang_scores = BASE_FOLDER / f"{lang}/{DATA_PART}/{lang}_oracle_{crit}.npy"
    path_lang_stanza_ratios = BASE_FOLDER / f"{lang}/{DATA_PART}/{lang}_stanza_ratios_{crit}.npy"
    path_lang_tie_ratios = BASE_FOLDER / f"{lang}/{DATA_PART}/{lang}_tie_ratios_{crit}.npy"

    if not path_lang_scores.is_file():
        compute_oracle_scores(lang=lang, crit=crit)

    gold_tree_ratios = np.load(str(path_lang_gold_tree_ratios), allow_pickle=True)
    oracle_scores = np.load(str(path_lang_scores), allow_pickle=True)
    stanza_ratios = np.load(str(path_lang_stanza_ratios), allow_pickle=True)
    tie_ratios = np.load(str(path_lang_tie_ratios), allow_pickle=True)

    print(gold_tree_ratios)
    print(oracle_scores)

    k_vec = np.array([k for k in range(1,K+1)])

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    title = f"Oracle performance and gold tree ratios in {lang}"
    fig.suptitle(title, fontsize=16)

    xticks = [1, *[num for num in range(5,K+1,5)]]

    p1 = sb.lineplot(x=k_vec, y=oracle_scores, ax=ax1)
    p1.set(ylabel=f'Oracle {crit}')
    p1.tick_params(labelbottom=False)

    p2 = sb.lineplot(x=k_vec, y=stanza_ratios, ax=ax2)
    p2.set(ylabel=f'Stanza ratio')
    p2.tick_params(labelbottom=False)

    p3 = sb.lineplot(x=k_vec, y=tie_ratios, ax=ax3)
    p3.set(ylabel=f'Tie ratio')
    p3.tick_params(labelbottom=False)

    p4 = sb.lineplot(x=k_vec, y=gold_tree_ratios, ax=ax4)
    p4.set(xlabel='k', ylabel=f'Gold Tree ratio', xticks=xticks)

    # Stanza ratio is the number of times that the Stanza prediction was the best performing according to the criterion (normalized by the total number of sentences)
    # Tie ratio is the number of times that the maximal score was tied (i.e. multiple candidate trees attended the max

    plt.savefig(str(BASE_FOLDER / f"oracle_perf_{lang}.jpeg"))


if __name__ == "__main__":
    for lang in ["lt_hse", "be_hse", "mr_ufal", "ta_ttb"]:
        compute_oracle_scores(lang=lang, crit=CRITERION)
    compute_random_baseline(lang="lt_hse", max_k=15, crit=CRITERION)

