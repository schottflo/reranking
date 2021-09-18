from parsing.algorithms.cd_dep_dynamic_program import compute_num_matching_subgraphs_dp as compute_num_matching_subgraphs_dp
import numpy as np
from scipy.spatial.distance import squareform
from scipy.stats import norm
from sklearn.svm import SVC

def convert_into_corr_mat(cov_mat):
    """
    Convert a covariance matrix into a correlation matrix

    :param sent_inds: np.array
    :return: np.array
    """
    n = cov_mat.shape[0]
    upper_triang = np.empty(shape=n * (n - 1) // 2)
    k = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            upper_triang[k] = cov_mat[i, j] / (np.sqrt(cov_mat[i, i]) * np.sqrt(cov_mat[j, j]))
            k += 1

    # Create a symmetric matrix from the upper triangular values (still has 0's on the main diagonal)
    corr = squareform(upper_triang)
    corr = corr + np.diag(np.ones(n))

    return corr


def CustomKernel(X, X2=None, lamb=1, emb_scale=1):
    """
    Compute the full kernel matrix

    :param X: np.array
    :param X2: np.array
    :return: np.array
    """
    n = X.shape[0]

    if X2 is None:  # Symmetric case

        # Compute the upper triangular values
        vec = np.empty(shape=n * (n - 1) // 2)
        k = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                vec[k] = compute_num_matching_subgraphs_dp(t1=X[i, 0], t2=X[j, 0], lamb=lamb)
                k += 1

        # Create a symmetric matrix from the upper triangular values (still has 0's on the main diagonal)
        CD = squareform(vec)

        # Compute the main diagonal
        CD_diag = np.empty(n)
        for i in range(n):
            CD_diag[i] = compute_num_matching_subgraphs_dp(t1=X[i, 0], t2=X[i, 0], lamb=lamb)

        # Add the main diagonal to get the final Collins & Duffy matrix
        CD = CD + np.diag(CD_diag)

        CD = convert_into_corr_mat(CD)

        # Combine the C&D matrix with the embeddings matrix
        K = CD

        print("Eig Vals under 0")
        eigval = np.linalg.eigvals(K)
        print(eigval[eigval < 0])

        return K

    else:  # Non-symmetric case

        n2 = X2.shape[0]

        # Calculate the Collins & Duffy matrix
        CD = np.empty((n, n2))

        for i in range(n):
            for j in range(n2):
                CD[i, j] = compute_num_matching_subgraphs_dp(t1=X[i, 0], t2=X2[j, 0], lamb=lamb)

        # Combine the C&D matrix with the embeddings matrix
        K = CD

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
    gram_train = CustomKernel(X=X_train, lamb=0.7012455615069223)
    gram_test = CustomKernel(X=X_test, X2=X_train, lamb=0.7012455615069223)

    # Fit the model
    model = SVC(C=params["C"], kernel="precomputed")
    model.fit(gram_train, y_train)

    # Predict on the test set
    y_dec = model.decision_function(gram_test)

    return norm.cdf(y_dec)  # Turn into a probabilitiy