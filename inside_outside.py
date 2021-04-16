import numpy as np

# Implementation from Ran Zmigrod

def gen_cfg(N, n):
    # Initialize the grammar (would need to be defined/learned in practice)
    R_A = np.exp(np.random.random((N, N, N))) # N is the number of non-terminals
    R_w = np.exp(np.random.random((N, n))) # n is the length of the sentence (and the number of terminals)
    return R_w, R_A

def inside(R_w, R_A):
    # equivalent to CKY algorithm
    N, W = R_w.shape
    beta = np.zeros((N, W+1, W+1)) # W+1 instead of W because we are considering spans
    for k in range(W):
        for A in range(N):
            beta[A, k, k+1] += R_w[A, k] # R_w is only concerned with terminal productions
    for width in range(2, W+1):
        for i in range(W - width + 1):
            k = i + width
            for j in range(i+1, k):
                for A in range(N):
                    for B in range(N):
                        for C in range(N):
                            beta[A, i, k] += R_A[A, B, C] * beta[B, i, j] * beta[C, j, k] # R_A are the non-term. prod.
    Z = beta[0, 0, W]
    return Z, beta

def inside_outside(R_w, R_A):
    N, W = R_w.shape
    Z, beta = inside(R_w, R_A)
    alpha1 = np.zeros((N, W+1, W+1))
    alpha_A = np.zeros((N, N, N))
    alpha_w = np.zeros((N, W))
    alpha1[0, 0, W] += 1
    for width in range(W, 1, -1):
        for i in range(W - width + 1):
            k = i + width
            for j in range(i+1, k):
                for A in range(N):
                    for B in range(N):
                        for C in range(N):
                            alpha_A[A, B, C] += alpha1[A, i, k] * beta[B, i, j] * beta[C, j, k]
                            alpha1[B, i, j] += R_A[A, B, C] * alpha1[A, i, k] * beta[C, j, k]
                            alpha1[C, j, k] += R_A[A, B, C] * beta[B, i, j] * alpha1[A, i, k]
    for k in range(W):
        for A in range(N):
            alpha_w[A, k] += alpha1[A, k, k+1]
    return alpha_w, alpha_A, alpha1

if __name__ == '__main__':
    sent = "The man eats fish"
    n = len(sent)
    N = 3
    R_w, R_A = gen_cfg(N, n)
    print(inside_outside(R_w, R_A))














# G = {""}

# N = 10 # Number of non-terminals in a given tree
#
# T = len(sent) # Number of terminals in a given tree
#
# W_N = np.zeros((N, N, N))
# W_T = np.zeros((N, T))
#
#
#
# G = {}
#
# for k in range(sent_len):
#     for terminal_rule in terminal_rules:
#         W_T[i, i+1, pre_terminal] = G[A -> ]
#
# for span in range(2,sent_len+1):
#     for i in range(n-span+1):
#         k = i + span
#         for j in range(i+1, k):
#             for non_terminal_rule in non_terminal_rules:
#                 X = non_terminal_rule.lhs()
#                 Y, Z = non_terminal_rule.rhs()
#                 W_N[i,k,X] += W_



