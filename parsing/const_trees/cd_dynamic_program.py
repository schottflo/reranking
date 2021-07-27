from tree import Node, Production, ConstituencyTree
import autograd.numpy as np
import time

def compute_num_matching_subtrees_dp(t1, t2, lamb=1, show_DP_table=False):
    """
    Given two trees (in CNF), this function returns the number of matching subtrees.

    :param t1: ConstituencyTree
    :param t2: ConstituencyTree
    :return: int
    """
    # Initialize the DP table: it needs specification by the position of the match in each tree (ow: you overcount)
    # General remark on the DP_table: Always need to cast the first element of the tuple key as a str

    DP_table = {} # array (Cython)

    # Initialize the terminal nodes
    for terminal_t1 in t1.terminals():
        for terminal_t2 in t2.terminals():
            DP_table[(terminal_t1, terminal_t1.pos, terminal_t2.pos)] = 0.0

    # Initialize the non-terminal nodes
    for nonterminal_t1, _ in t1:
        for nonterminal_t2, _ in t2:
            DP_table[(nonterminal_t1, nonterminal_t1.pos, nonterminal_t2.pos)] = 0.0

    # Run the algorithm of Collins & Duffy (2001)
    for nonterminal_t1, prod_t1 in t1:
        for nonterminal_t2, prod_t2 in t2:
            if nonterminal_t1 == nonterminal_t2 and prod_t1 == prod_t2: # Node symbols (not positions!) and productions need to match
                factors = [(1 + DP_table[(prod_t1[i], prod_t1[i].pos, prod_t2[i].pos)]) for i in range(len(prod_t1))]
                DP_table[(nonterminal_t1, nonterminal_t1.pos, nonterminal_t2.pos)] = \
                    DP_table[(nonterminal_t1, nonterminal_t1.pos, nonterminal_t2.pos)] + lamb * np.prod(factors) # Doesn't use += notation because of autograd

    if show_DP_table:
        print(DP_table)

    return sum(DP_table.values())


if __name__ == "__main__":

    np.random.seed(42)

    # Load true parses
    trees = np.load("true_parses.npy", allow_pickle=True)

    vals = []
    times = []

    for _ in range(1):

        # Choose two random trees
        t1, t2 = np.random.choice(trees, size=2)

        # Compute the needed time for one comparison
        start = time.time()
        t = compute_num_matching_subtrees_dp(t1, t2, lamb=1)
        end = time.time()

        times.append(end - start)
        vals.append(t)

    print(vals)
    print(times)

