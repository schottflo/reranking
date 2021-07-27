from os.path import isfile
import numpy as np
from matplotlib import pyplot as plt

def extract_metrics(svm_res, gp_res):
    """
    Extract the AUC, average Precision, ECE and Constituent F1 vector from the arrays

    :param gp: np.array
    :param svm: np.array
    :return:
    """
    model = np.array(["SVM" for _ in range(3)] + ["GP" for _ in range(3)])
    auc = np.concatenate([svm_res[:,0], gp_res[:,0]])
    prec = np.concatenate([svm_res[:,1], gp_res[:,1]])
    ece = np.concatenate([svm_res[:,2], gp_res[:,2]])
    const_f1 = np.concatenate([svm_res[:,3], gp_res[:,3]])

    return model, auc, prec, ece, const_f1

def main():

    if not isfile("svm_results.npy"):

        print("--The results of the SVM model need to be computed first.--\n This is recommended to be done on a cluster\n")
        print("Alternatively, run the svm_models.py file first")
        return FileNotFoundError

    if not isfile("gp_results.npy"):

        print("--Computing the results of the GP model now. This likely takes approximately 2.5 hours if the data"
              "is already generated--")

    svm = np.load("svm_results.npy",allow_pickle=True)
    gp = np.load("gp_results.npy", allow_pickle=True)

    print("GP")
    print(gp)
    print("\n")
    print("SVM")
    print(svm)

    model, auc, prec, ece, const_f1 = extract_metrics(svm_res=svm, gp_res=gp)

    svm_avg = np.mean(svm, axis=0)
    gp_avg = np.mean(gp, axis=0)

    # General plot set up
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    title = "Collins Duffy + Embeddings Dot Product (200 sentences)"
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(pad=2.0)

    # AUC plot
    ax1.scatter(x=model, y=auc)
    ax1.scatter(np.array(["SVM", "GP"]), np.array([svm_avg[0], gp_avg[0]]), c="red")
    ax1.axhline(svm_avg[0], c="green", linewidth=0.3, linestyle="dashed")
    ax1.axhline(gp_avg[0], c="green", linewidth=0.3, linestyle="dashed")
    ax1.set_ylabel("AUC")
    ax1.set_ylim([0.4, 1])
    ax1.set_title(f"Average AUC  SVM:{round(svm_avg[0],4)}, GP:{round(gp_avg[0],4)}")

    # Average Prec plot
    ax2.scatter(x=model, y=prec)
    ax2.scatter(np.array(["SVM", "GP"]), np.array([svm_avg[1], gp_avg[1]]), c="red")
    ax2.axhline(svm_avg[1], c="green", linewidth=0.3, linestyle="dashed")
    ax2.axhline(gp_avg[1], c="green", linewidth=0.3, linestyle="dashed")
    ax2.set_ylabel("Average Precision")
    ax2.set_ylim([0.166, 1])
    ax2.set_title(f"Average Precision  SVM:{round(svm_avg[1],4)}, GP:{round(gp_avg[1],4)}")

    # ECE plot
    ax3.scatter(x=model, y=ece)
    ax3.scatter(np.array(["SVM", "GP"]), np.array([svm_avg[2], gp_avg[2]]), c="red")
    ax3.axhline(svm_avg[2], c="green", linewidth=0.3, linestyle="dashed")
    ax3.axhline(gp_avg[2], c="green", linewidth=0.3, linestyle="dashed")
    ax3.set_ylabel("Expected Calibration Error")
    ax3.set_ylim([0, 1])
    ax3.set_title(f"Average ECE  SVM:{round(svm_avg[2],4)}, GP:{round(gp_avg[2],4)}")

    # Constituent F1 plot
    ax4.scatter(x=model, y=const_f1)
    ax4.scatter(np.array(["SVM", "GP"]), np.array([svm_avg[3], gp_avg[3]]), c="red")
    ax4.axhline(svm_avg[3], c="green", linewidth=0.3, linestyle="dashed")
    ax4.axhline(gp_avg[3], c="green", linewidth=0.3, linestyle="dashed")
    ax4.set_ylabel("Constituent F1 Score")
    ax4.set_ylim([0.8, 1])
    ax4.set_title(f"Average Constituent F1 Score  SVM:{round(svm_avg[3],4)}, GP:{round(gp_avg[3],4)}", wrap=True)

    plt.show()

if __name__ == "__main__":
    main()