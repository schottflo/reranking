import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from pathlib import Path
import pandas as pd

BASE_FOLDER = Path.cwd()

def load_results(lang, eval_set, num_seeds, max_k, kernel, gold=False):

    las = []
    uas = []
    clas = []
    emp_stanz_ratios = []

    if gold:
        type_g = "gold"
    else:
        type_g = "non_gold"

    for i in range(1, num_seeds + 1):

        if gold:
            path_las = BASE_FOLDER / f"gps/{type_g}/{lang}/{kernel}/{eval_set}/{lang}_las_scores_{eval_set}_{i}_gold.npy"
            path_uas = BASE_FOLDER / f"gps/{type_g}/{lang}/{kernel}/{eval_set}/{lang}_uas_scores_{eval_set}_{i}_gold.npy"
            path_clas = BASE_FOLDER / f"gps/{type_g}/{lang}/{kernel}/{eval_set}/{lang}_clas_scores_{eval_set}_{i}_gold.npy"
            path_emp_stanza = BASE_FOLDER / f"gps/{type_g}/{lang}/{kernel}/{eval_set}/{lang}_emp_stanz_ratios_{eval_set}_{i}_gold.npy"

        else:
            path_las = BASE_FOLDER / f"gps/{type_g}/{lang}/{kernel}/{eval_set}/{lang}_las_scores_{eval_set}_{i}.npy"
            path_uas = BASE_FOLDER / f"gps/{type_g}/{lang}/{kernel}/{eval_set}/{lang}_uas_scores_{eval_set}_{i}.npy"
            path_clas = BASE_FOLDER / f"gps/{type_g}/{lang}/{kernel}/{eval_set}/{lang}_clas_scores_{eval_set}_{i}.npy"
            path_emp_stanza = BASE_FOLDER / f"gps/{type_g}/{lang}/{kernel}/{eval_set}/{lang}_emp_stanz_ratios_{eval_set}_{i}.npy"

        add_las = np.load(str(path_las),allow_pickle=True)[:max_k]
        add_uas = np.load(str(path_uas),allow_pickle=True)[:max_k]
        add_clas = np.load(str(path_clas),allow_pickle=True)[:max_k]
        add_emp_stanza = np.load(str(path_emp_stanza),allow_pickle=True)[:max_k]

        las.append(add_las)
        uas.append(add_uas)
        clas.append(add_clas)
        emp_stanz_ratios.append(add_emp_stanza)

    k_vec = np.array([k for k in range(1, max_k + 1)])

    las_mat = np.stack(las, axis=1)
    uas_mat = np.stack(uas, axis=1)
    clas_mat = np.stack(clas, axis=1)
    emp_stanza_ratio_mat = np.stack(emp_stanz_ratios, axis=1)

    return k_vec, las_mat, uas_mat, clas_mat, emp_stanza_ratio_mat


def transform_into_df(mat, k, num_seeds, col_name):

    stacked = pd.DataFrame(mat)
    stacked["k"] = k
    stacked.columns = [*[f"{col_name}{i}" for i in range(1, num_seeds+1)], "k"]

    return stacked


def experiment_1(lang, eval_set, num_seeds, max_k):

    k_vec, gold_las_mat, _, _, _ = load_results(lang=lang, eval_set=eval_set, num_seeds=num_seeds, max_k=max_k, gold=True, kernel="augm_cd")

    _, other_gold_las_mat, _, _, _ = load_results(lang=lang, eval_set=eval_set, num_seeds=num_seeds, max_k=max_k,
                                                kernel="cd", gold=True)

    # Random baseline
    random_baseline = np.load(str(BASE_FOLDER / f"oracle_computations/{lang}/test/{lang}_random_baseline.npy"), allow_pickle=True)

    # LAS
    stacked_gold_las = transform_into_df(mat=gold_las_mat, k=k_vec, num_seeds=num_seeds, col_name="gold_las")
    stacked_other_gold_las = transform_into_df(mat=other_gold_las_mat, k=k_vec, num_seeds=num_seeds, col_name="other_gold_las")

    wide_gold_las = pd.wide_to_long(stacked_gold_las, stubnames="gold_las", i="k", j="seed")
    wide_other_gold_las = pd.wide_to_long(stacked_other_gold_las, stubnames="other_gold_las", i="k", j="seed")

    oracle_las = np.load(str(BASE_FOLDER / f"oracle_computations/{lang}/{eval_set}/{lang}_oracle_LAS.npy"))[:(max_k-1)]
    oracle_las = np.insert(oracle_las, 0, 0)

    baseline = oracle_las[1]

    xticks = [num for num in range(1, max_k + 1, 1)]

    sb.set_style("darkgrid")
    s2 = sb.lineplot(x="k", y="other_gold_las", data=wide_other_gold_las, color="orange")
    sb.lineplot(x="k", y="gold_las", data=wide_gold_las, color="green")
    sb.lineplot(x=k_vec, y=oracle_las)
    sb.lineplot(x=k_vec, y=random_baseline, color="violet")
    s2.axhline(baseline, ls='--', color="red")
    s2.set(ylabel="LAS", ylim=(-1, 101), xlabel="Test k")
    s2.set_xticks(xticks)
    s2.legend(labels=["GP RR C&D Kernel incl gold", "GP RR Augm C&D Kernel incl gold", "Optimal predictions excl. gold candidate", "Random Reranker", "Base Parser"])

    plt.show()

def experiment_2(eval_set, kernel, num_seeds, max_k):

    data = []
    plot_limits = []

    for lang in ["lt_hse", "be_hse", "mr_ufal", "ta_ttb"]:

        k_vec, las_mat, uas_mat, clas_mat, _ = load_results(lang=lang, eval_set=eval_set, num_seeds=num_seeds, max_k=max_k, kernel=kernel)
        print(np.mean(las_mat, axis=1))
        print(np.std(las_mat, axis=1, ddof=1))

        stacked_las = transform_into_df(mat=las_mat, k=k_vec, num_seeds=num_seeds, col_name="las")
        wide_las = pd.wide_to_long(stacked_las, stubnames="las", i="k", j="seed")
        data.append(wide_las)

        plot_limits.append((np.amin(las_mat)+0.1, np.amax(las_mat)+0.15))

    xticks = [num for num in range(1, max_k + 1, 1)]

    fig = plt.figure()
    with sb.axes_style("darkgrid"):

        lb, ub = plot_limits[0]

        ax1 = fig.add_subplot(411)
        s1 = sb.lineplot(x="k", y="las", data=data[0], legend="full", ax=ax1)
        s1.set(ylabel="LAS", ylim=(lb+0.15, ub+0.1))
        s1.set_xticklabels([])
        s1.set_xlabel("")
        s1.set_title(label="Lithuanian HSE", fontsize=10)
        s1.legend(labels=["Mean Reranker LAS"])

        ax2 = fig.add_subplot(412)
        s2 = sb.lineplot(x="k", y="las", data=data[1], legend="full", ax=ax2)
        s2.set(ylabel="LAS", ylim=plot_limits[1])
        s2.set_xticklabels([])
        s2.set_xlabel("")
        s2.set_title(label="Belarussian HSE", fontsize=10)
        s2.legend(labels=["Mean Reranker LAS"])

        ax3 = fig.add_subplot(413)
        s3 = sb.lineplot(x="k", y="las", data=data[2], legend="full", ax=ax3)
        s3.set(ylabel="LAS", ylim=plot_limits[2])
        s3.set_xticklabels([])
        s3.set_xlabel("")
        s3.set_title(label="Marathi UFAL", fontsize=10)
        s3.legend(labels=["Mean Reranker LAS"])

        ax4 = fig.add_subplot(414)
        s4 = sb.lineplot(x="k", y="las", data=data[3], legend="full", ax=ax4)
        s4.set(ylabel="LAS", ylim=plot_limits[3], xlabel="Development k")
        s4.set_title(label="Tamil TTB", fontsize=10)
        s4.set_xticks(xticks)
        s4.legend(labels=["Mean Reranker LAS"])

    plt.show()


def experiment_3_1(lang, eval_set, num_seeds, max_k): # conf_thresh=False

    cd_gp_emp_stanza_ratios = load_results(lang, eval_set, num_seeds, max_k, kernel="cd")[4]
    augm_cd_gp_emp_stanza_ratios = load_results(lang, eval_set, num_seeds, max_k, kernel="augm_cd")[4]

    avg_cd_gp_emp_stanza_ratios = np.mean(cd_gp_emp_stanza_ratios, axis=1)
    avg_augm_cd_gp_emp_stanza_ratios = np.mean(augm_cd_gp_emp_stanza_ratios, axis=1)

    print(avg_cd_gp_emp_stanza_ratios)
    print(avg_augm_cd_gp_emp_stanza_ratios)

    k_vec = [k for k in range(1, max_k+1)]

    true_stanza_ratio = np.load(str(BASE_FOLDER / f"oracle_computations/{lang}/{eval_set}/{lang}_stanza_ratios_LAS.npy"))[:max_k]

    svm_emp_stanza_ratios = np.load(str(BASE_FOLDER / f"svms/{lang}/cd/{lang}_emp_stanz_ratios_test.npy"), allow_pickle=True)
    svm_emp_stanza_ratios = np.insert(svm_emp_stanza_ratios, 0, 1)

    print(true_stanza_ratio)

    fig = plt.figure()
    with sb.axes_style("darkgrid"):

        sb.set_style("darkgrid")
        s1 = sb.lineplot(x=k_vec, y=true_stanza_ratio, color="red")
        sb.lineplot(x=k_vec, y=avg_cd_gp_emp_stanza_ratios, color="orange")
        sb.lineplot(x=k_vec, y=avg_augm_cd_gp_emp_stanza_ratios, color="green")
        sb.lineplot(x=k_vec, y=svm_emp_stanza_ratios, color="violet")
        s1.set_xticks(k_vec)
        s1.set_ylabel("Base parser ratio")
        s1.set_xlabel("k")
        s1.legend(labels=["True BPR", "Emp BPR of GP with C&D kernel",
                          "Emp BPR of GP with Augm C&D kernel",
                          "Emp BPR of SVM with C&D kernel"])

    plt.show()

def experiment_3_2(max_k):

    langs = ["lt_hse", "be_hse", "mr_ufal", "ta_ttb"]
    eval_set = "test"

    oracles = []
    gtr = []
    for lang in langs:

        oracle_las = np.load(str(BASE_FOLDER / f"oracle_computations/{lang}/{eval_set}/{lang}_oracle_LAS.npy"))[:max_k]
        cum_growth_oracle = 100 * (oracle_las - oracle_las[0]) / oracle_las[0]
        oracles.append(cum_growth_oracle)

        gtr.append(100 * np.load(str(BASE_FOLDER / f"oracle_computations/{lang}/{eval_set}/{lang}_gold_tree_ratios.npy"))[:max_k])

    k_vec = np.array([k for k in range(1, max_k + 1)])

    print(k_vec)

    stacked_oracles = pd.DataFrame(np.stack(oracles, axis=1))
    stacked_oracles.set_index(k_vec, inplace=True)

    stacked_gtr = pd.DataFrame(np.stack(gtr, axis=1))
    stacked_gtr.set_index(k_vec, inplace=True)

    fig = plt.figure()
    with sb.axes_style("darkgrid"):

        ax1 = fig.add_subplot(121)
        g1 = sb.lineplot(data=stacked_oracles, ax=ax1)
        g1.set_xticks(k_vec)
        g1.set_xlabel("Test k")
        g1.set_ylabel("Growth of oracle LAS compared to k=1 in %")
        g1.legend(labels=langs)

        ax2 = fig.add_subplot(122)
        g = sb.lineplot(data=stacked_gtr, ax=ax2)
        g.set_xticks(k_vec)
        g.set_xlabel("Test k")
        g.set_ylabel("Gold tree ratio in %")
        g.legend(labels=langs)

    plt.show()

def experiment_3_3(max_k):

    # Not shown in the thesis
    langs = ["lt_hse", "be_hse", "mr_ufal", "ta_ttb"]
    eval_set = "test"

    tie_ratios = []
    for lang in langs:
        tie_ratios.append(np.load(str(BASE_FOLDER / f"oracle_computations/{lang}/{eval_set}/{lang}_tie_ratios_LAS.npy"))[:max_k])

    k_vec = np.array([k for k in range(1, max_k + 1)])

    print(k_vec)

    stacked_ties = pd.DataFrame(np.stack(tie_ratios, axis=1))
    stacked_ties.set_index(k_vec, inplace=True)

    fig = plt.figure()
    with sb.axes_style("darkgrid"):
        ax1 = fig.add_subplot(111)
        g1 = sb.lineplot(data=stacked_ties, ax=ax1)
        g1.set_xticks(k_vec)
        g1.set_ylabel("Max Score Tie Ratio")
        g1.legend(labels=langs)

    plt.show()


if __name__ == "__main__":
    experiment_1(lang="lt_hse", eval_set="test", num_seeds=5, max_k=15)
    experiment_2(eval_set="dev", kernel="augm_cd", num_seeds=5, max_k=15)
    experiment_3_1(lang="lt_hse", eval_set="test", num_seeds=5, max_k=15)
    experiment_3_1(lang="mr_ufal", eval_set="test", num_seeds=5, max_k=15)
    experiment_3_2(max_k=15)

    experiment_2(eval_set="dev", kernel="cd", num_seeds=5, max_k=15)
