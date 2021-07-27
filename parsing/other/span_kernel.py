
import numpy as np
import inside_outside

sent1 = "The man eats fish"
sent2 = "The woman knows meat"

def span_kernel(sent1, sent2):

    n_1 = len(sent1)
    n_2 = len(sent2)

    N = 10

    # Have to make sure that the grammar is the same
    R_w_1, R_A_1 = inside_outside.gen_cfg(N, n_1)
    R_w_2, R_A_2 = inside_outside.gen_cfg(N, n_2)

    betas1, alphas1 = inside_outside.inside_outside(R_w_1, R_A_1)
    betas2, alphas2 = inside_outside.inside_outside(R_w_2, R_A_2)

    alphas1 = alphas1[1]
    alphas2 = alphas2[1]

    print(betas1.shape)

    total = 0

    C_A = np.zeros((N, N, N)) # N is the number of non-terminals


    n = len(sent1)

    for l in range(2, n):
        for i in range(1, n-l):
            k = i + l
            for j in range(i+1, k):
                for A in range(1, N):
                    for B in range(1, N):
                        for C in range(1, N):
                            # Prob. of the whole constituent
                            print(i)
                            print(k)
                            C_A[A, i, k] += betas1[A, i, k] * betas2[A, i, k]

                            # Left
                            C_A[A, i, k] += C_A[B, i, j] * betas1[C, j, k] * betas2[C, j, k] * R_A_1[A, B, C] * R_A_2[A, B, C]

                            # Right
                            C_A[A, i, k] += C_A[C, j, k] * betas1[B, i, j] * betas2[B, i, j] * R_A_1[A, B, C] * R_A_2[A, B, C]

                            # Both
                            C_A[A, i, k] += C_A[B, i, j] * C_A[C, j, k] * R_A_1[A, B, C] * R_A_2[A, B, C]

    for i in range(1, n+1):
        for k in range(i+2, n):
            for X in range(N):
                total += alphas1[X, i, k] * alphas2[X, i, k] * C_A[X, i, k]

    return total

print(span_kernel(sent1, sent2))

