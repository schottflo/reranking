import numpy as np

# Implementation from Ran Zmigrod

def gen_cfg(N, n):
    # Initialize the grammar (would need to be defined/learned in practice)
    R_A = np.exp(np.random.random((N, N, N))) # N is the number of non-terminals
    R_w = np.exp(np.random.random((N, n))) # n is the length of the sentence (and the number of terminals)
    return R_w, R_A

def inside(R_w, R_A):
    # equivalent to CKY algorithm (finds the probability of deriving w_i+1:j from non-terminal X for every X)
    N, W = R_w.shape
    beta = np.zeros((N, W+1, W+1)) # W+1 instead of W because we are considering spans
    for k in range(W):
        for A in range(N):
            beta[A, k, k+1] += R_w[A, k] # R_w is only concerned with terminals (constituents)
    for width in range(2, W+1):
        for i in range(W - width + 1):
            k = i + width
            for j in range(i+1, k):
                for A in range(N):
                    for B in range(N):
                        for C in range(N):
                            beta[A, i, k] += R_A[A, B, C] * beta[B, i, j] * beta[C, j, k] # R_A are non-term const
    Z = beta[0, 0, W]
    return Z, beta

def inside_outside(R_w, R_A):
    # outside probabilities: prob of having a phrase of type X covering span i+1:j with the exterior context w_1:i and
    # w_j+1:M (where M is the whole sentence length)
    N, W = R_w.shape
    Z, beta = inside(R_w, R_A)
    alpha1 = np.zeros((N, W+1, W+1)) # for constituents
    alpha_A = np.zeros((N, N, N)) # for non-terminal productions
    alpha_w = np.zeros((N, W)) # for terminal productions
    alpha1[0, 0, W] += 1
    for width in range(W, 1, -1):
        for i in range(W - width + 1):
            k = i + width
            for j in range(i+1, k):
                for A in range(N):
                    for B in range(N):
                        for C in range(N):
                            # Go over all positions and add the probabilities of outside at that position times
                            # prob of two constituents given that A is the head
                            alpha_A[A, B, C] += alpha1[A, i, k] * beta[B, i, j] * beta[C, j, k]

                            # R_A is decisive, it needs to be initialized correctly for this to work
                            # Left and right sum cannot be directly seen from this
                            alpha1[B, i, j] += R_A[A, B, C] * alpha1[A, i, k] * beta[C, j, k]

                            # R_A is decisive, it needs to be initialized correctly for this to work
                            # Left and right sum cannot be directly seen from this
                            alpha1[C, j, k] += R_A[A, B, C] * beta[B, i, j] * alpha1[A, i, k]
    for k in range(W):
        for A in range(N):
            alpha_w[A, k] += alpha1[A, k, k+1] # terminal constituents
    return alpha_w, alpha_A, alpha1

if __name__ == '__main__':
    sent = "The man eats fish"
    n = len(sent)
    N = 3
    R_w, R_A = gen_cfg(N, n)
    print(inside_outside(R_w, R_A))