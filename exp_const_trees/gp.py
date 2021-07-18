import autograd.numpy as np
from autograd import grad

from random import seed
from scipy.spatial.distance import squareform

from tree import Node, Production, ConstituencyTree
from cd_dynamic_program import compute_num_matching_subtrees_dp

from GPy.kern.src.kern import Kern
from GPy.core import Param
from GPy.models import GPClassification

seed(42)
np.random.seed(42)


class Custom_GPY(Kern):

    def __init__(self, input_dim, lamb, e, active_dims=None):
        super(Custom_GPY, self).__init__(input_dim, active_dims, 'custom')

        self.lamb = Param('lamb', lamb)
        self.lamb.constrain_positive()

        self.emb_scale = Param('emb_scale', e)
        self.emb_scale.constrain_positive()

        self.link_parameters(self.lamb, self.emb_scale)

    def K(self, X, X2):
        """
        Compute the full kernel matrix

        :param X: np.array
        :param X2: np.array
        :return: np.array
        """
        n = X.shape[0]

        if X2 is None: # Symmetric case

            # Compute the upper triangular values
            vec = np.empty(shape=n * (n - 1) // 2)
            k = 0
            for i in range(n - 1):
                for j in range(i + 1, n):
                    vec[k] = compute_num_matching_subtrees_dp(t1=X[i, 0], t2=X[j, 0], lamb=self.lamb)
                    k += 1

            # Create a symmetric matrix from the upper triangular values (still has 0's on the main diagonal)
            CD = squareform(vec)

            # Compute the main diagonal
            CD_diag = np.empty(n)
            for i in range(n):
                CD_diag[i] = compute_num_matching_subtrees_dp(t1=X[i, 0], t2=X[i, 0], lamb=self.lamb)

            # Add the main diagonal to get the final Collins & Duffy matrix
            CD = CD + np.diag(CD_diag)

            # Calculate the embeddings matrix
            self.E_symm = self.emb_scale * np.array(np.matmul(X[:, 1:], X[:, 1:].T), dtype=np.float32)

            # Combine the C&D matrix with the embeddings matrix
            K = CD + self.E_symm

        else: # Non-symmetric case

            n2 = X2.shape[0]

            # Calculate the Collins & Duffy matrix
            CD = np.empty((n, n2))

            for i in range(n):
                for j in range(n2):
                    CD[i, j] = compute_num_matching_subtrees_dp(t1=X[i, 0], t2=X2[j, 0], lamb=self.lamb)

            # Calculate the embeddings matrix
            self.E_asymm = self.emb_scale * np.array(np.matmul(X[:, 1:], X2[:, 1:].T), dtype=np.float32)

            # Combine the C&D matrix with the embeddings matrix
            K = CD + self.E_asymm

        return K

    def Kdiag(self, X):
        """
        Compute the main diagonal of the kernel matrix

        :param X: np.array
        :return: np.array (1-dim)
        """
        n = X.shape[0]
        CD_diag = np.empty(n)
        E_diag = np.empty(n)
        for i in range(n):
            CD_diag[i] = compute_num_matching_subtrees_dp(t1=X[i, 0], t2=X[i, 0], lamb=self.lamb)
            E_diag[i] = self.emb_scale * np.dot(X[i, 1:], X[i, 1:])

        return CD_diag + E_diag

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
        part_deriv = grad(compute_num_matching_subtrees_dp, 2)

        if X2 is None:

            # Compute the upper triangular values of the dK/dlambda
            vec = np.empty(shape=n * (n - 1) // 2)
            k = 0
            for i in range(n - 1):
                for j in range(i + 1, n):
                    vec[k] = part_deriv(X[i, 0], X[j, 0], self.lamb*1)
                    k += 1

            # Create a symmetric matrix from the upper triangular values (still has 0's on the main diagonal)
            dlambda = squareform(vec)

            # Compute the main diagonal
            dlambda_diag = np.empty(n)
            for i in range(n):
                dlambda_diag[i] = part_deriv(X[i, 0], X[i, 0], self.lamb*1)

            # Add the main diagonal to get the final derivative
            dlambda = dlambda + np.diag(dlambda_diag)

            # Compute dL/demb_scale
            dE = self.E_symm / self.emb_scale
            self.emb_scale.gradient = np.sum(dE * dL_dK)

        else:

            n2 = X2.shape[0]

            # Evaluate the derivative at every element of the kernel matrix
            dlambda = np.empty((n, n2))
            for i in range(n):
                for j in range(n2):
                    dlambda[i, j] = part_deriv(X[i, 0], X2[j, 0], self.lamb*1)

            # Compute dL/demb_scale
            dE = self.E_asymm / self.emb_scale
            self.emb_scale.gradient = np.sum(dE * dL_dK)

        # Compute dL/dlambda
        self.lamb.gradient = np.sum(dlambda * dL_dK)


def fit_and_predict_gp(X_train, X_test, y_train, y_test):
    """
    Fit a GP model with the custom kernel and return the sampled predictions.

    :param X_train: np.array
    :param X_test: np.array
    :param y_train: np.array
    :param y_test: np.array
    :param index: int
    :return: tuple of float
    """
    # Set up the kernel and fit the model with the best parameters from an optimized model on 50 samples
    kernel = Custom_GPY(input_dim=X_train.shape[1],lamb=0.5122493885645022, e=0.019017003926375585) # average values from optimized model on 50 observations
    gp_model = GPClassification(X=X_train, Y=y_train.reshape(-1, 1), kernel=kernel)

    print("Model fitted; Sampling starts now")

    # Produce 100 samples from the predictive posterior of the latent function and push them through the link function
    latent_function_samples = gp_model.posterior_samples_f(X=X_test, size=100).reshape((X_test.shape[0], 100))
    pred = gp_model.likelihood.gp_link.transf(latent_function_samples)

    return np.mean(pred, axis=1) # compute the empirical average of these probabilities