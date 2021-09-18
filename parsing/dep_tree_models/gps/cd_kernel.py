from parsing.algorithms.cd_dep_dynamic_program import compute_num_matching_subgraphs_dp as compute_num_matching_subgraphs_dp

import autograd.numpy as np
from autograd import grad

from scipy.spatial.distance import squareform

from GPy.kern.src.kern import Kern
from GPy.core import Param


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

def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a - a.T) < tol)

def is_pos_sem_def(x):
    return np.all(np.linalg.eigvals(x) >= 0)


class Custom_GPY(Kern):

    def __init__(self, input_dim, lamb, active_dims=None): #e,
        super(Custom_GPY, self).__init__(input_dim, active_dims, 'custom')

        self.lamb = Param('lamb', lamb)
        self.lamb.constrain_bounded(lower=0, upper=1)

        # self.emb_scale = Param('emb_scale', e)
        # self.emb_scale.constrain_bounded(lower=0, upper=1)

        self.link_parameters(self.lamb)#, self.emb_scale)

    def K(self, X, X2):
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
                    vec[k] = compute_num_matching_subgraphs_dp(t1=X[i, 0], t2=X[j, 0], lamb=self.lamb)
                    k += 1

            # Create a symmetric matrix from the upper triangular values (still has 0's on the main diagonal)
            CD = squareform(vec)

            # Compute the main diagonal
            CD_diag = np.empty(n)
            for i in range(n):
                CD_diag[i] = compute_num_matching_subgraphs_dp(t1=X[i, 0], t2=X[i, 0], lamb=self.lamb)

            # Add the main diagonal to get the final Collins & Duffy matrix
            CD = CD + np.diag(CD_diag)

            CD = convert_into_corr_mat(CD) # not strictly necessary

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
                    CD[i, j] = compute_num_matching_subgraphs_dp(t1=X[i, 0], t2=X2[j, 0], lamb=self.lamb)

            # Combine the C&D matrix with the embeddings matrix
            K = CD

        return K

    def Kdiag(self, X):
        """
        Compute the main diagonal of the kernel matrix

        :param X: np.array
        :return: np.array (1-dim)
        """
        n = X.shape[0]
        CD_diag = np.empty(n)
        for i in range(n):
            CD_diag[i] = compute_num_matching_subgraphs_dp(t1=X[i, 0], t2=X[i, 0], lamb=self.lamb)

        return CD_diag

    def update_gradients_full(self, dL_dK, X, X2):
        """
        Compute the gradient of the loss function w.r.t the hyperparameters.

        :param dL_dK: np.array
        :param X: np.array
        :param X2: np.array
        :return: None
        """
        n = X.shape[0]

        # Set up the derivative of the C&D dynamic program w.r.t lambda
        part_deriv = grad(compute_num_matching_subgraphs_dp, 2)

        if X2 is None:

            # Compute the upper triangular values of the dK/dlambda
            vec = np.empty(shape=n * (n - 1) // 2)
            k = 0
            for i in range(n - 1):
                for j in range(i + 1, n):
                    vec[k] = part_deriv(X[i, 0], X[j, 0], self.lamb*1)# * 1)
                    k += 1

            # Create a symmetric matrix from the upper triangular values (still has 0's on the main diagonal)
            dlambda = squareform(vec)

            # Compute the main diagonal
            dlambda_diag = np.empty(n)
            for i in range(n):
                dlambda_diag[i] = part_deriv(X[i, 0], X[i, 0], self.lamb*1)

            # Add the main diagonal to get the final derivative
            dlambda = dlambda + np.diag(dlambda_diag)

        else:

            n2 = X2.shape[0]

            # Evaluate the derivative at every element of the kernel matrix
            dlambda = np.empty((n, n2))
            for i in range(n):
                for j in range(n2):
                    dlambda[i, j] = part_deriv(X[i, 0], X2[j, 0], self.lamb*1)

        # Compute dL/dlambda
        self.lamb.gradient = np.sum(dlambda * dL_dK)

def initialize_kernel(num_col, internal_seed):

    return Custom_GPY(input_dim=num_col, lamb=0.7012455615069223)