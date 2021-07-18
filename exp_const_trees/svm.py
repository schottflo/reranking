from cd_dynamic_program import compute_num_matching_subtrees_dp
from tree import Node, Production, ConstituencyTree
from scipy.spatial.distance import squareform

from random import seed
import numpy as np

from sklearn.svm import SVC

seed(42)
np.random.seed(42)

def CustomKernel(X, X2=None, lamb=1, emb_scale=0.01):
    """
    Custom kernel function

    :param X: np.array
    :param X2: np.array
    :param lamb: float
    :param emb_scale: float
    :return: np.array
    """

    n = X.shape[0]

    if X2 is None:

        # Collins & Duffy
        # Compute upper triangular matrix
        vec = np.empty(shape=n * (n - 1) // 2)
        k = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                vec[k] = compute_num_matching_subtrees_dp(t1=X[i, 0], t2=X[j, 0], lamb=lamb)
                k += 1

        # Use symmetry of the matrix: fill in the lower triangle of the matrix; but: diagonal still zeros
        CD = squareform(vec)

        # Fill the diagonal
        CD_diag = np.empty(n)
        for i in range(n):
            CD_diag[i] = compute_num_matching_subtrees_dp(t1=X[i, 0], t2=X[i, 0], lamb=lamb)

        # Add the main diagonal to get the final Collins & Duffy matrix
        CD = CD + np.diag(CD_diag)

        # Embeddings
        E = emb_scale * np.array(np.matmul(X[:, 1:], X[:, 1:].T), dtype=np.float32)

    else:

        # Collins & Duffy
        n2 = X2.shape[0]
        CD = np.empty((n, n2))

        for i in range(n):
            for j in range(n2):
                CD[i, j] = compute_num_matching_subtrees_dp(t1=X[i, 0], t2=X2[j, 0], lamb=lamb)

        # Embeddings
        E = emb_scale * np.array(np.matmul(X[:, 1:], X2[:, 1:].T), dtype=np.float32)

    K = CD + E

    return K


def fit_and_predict_svm(X_train, X_test, y_train, params):
    """
    Fit a SVM model with the custom kernel and return predicted probabilities on the test set.

    :param X_train: np.array
    :param X_test: np.array
    :param y_train: np.array
    :param params: dict
    :return: np.array
    """
    # Construct gram matrix
    gram_train = CustomKernel(X=X_train, lamb=params["lamb"], emb_scale=params["e"])
    gram_test = CustomKernel(X=X_test, X2=X_train, lamb=params["lamb"], emb_scale=params["e"])

    # Fit the model
    model = SVC(C=params["C"], kernel="precomputed")
    model.fit(gram_train, y_train)

    # Predict on the test set
    y_dec = model.decision_function(gram_test)

    return 1 / (1 + np.exp(-y_dec)) # Turn into a probabilitiy

