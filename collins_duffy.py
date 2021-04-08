import nltk
from nltk.grammar import Production
from nltk import Tree, Nonterminal

#t1 = nltk.ParentedTree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP( (NP (D the) (N cat)) (NP (Adj and) (NP (D the) (N mouse)))))))")
# t1 = nltk.ParentedTree.fromstring("(S (NP (D the) (N dog)) (V barks))") #
# t2 = nltk.ParentedTree.fromstring("(S (NP (D the) (N cat)) (V eats))") #


t1 = nltk.ParentedTree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (VP (D the) (N cat)) (NP (Adj and) (NP (D the) (N mouse))))))")

#t1 = nltk.ParentedTree.fromstring("(S (NP (D the) (N man)) (VP (V eats) (NP fish)))")
# t2 = nltk.ParentedTree.fromstring("(S (NP (D the) (N woman)) (VP (V cooks) (NP meat)))")
#t2 = nltk.ParentedTree.fromstring("(S (NP (D the) (N woman)) (AdvP (Adv thoughtfully) (VP (V cooks) (NP meat))))")
#t2 = nltk.ParentedTree.fromstring("(S (NP (D the) (N woman)) (AdvP (Adv thoughtfully) (VP (V cooks) (NP meat))))")

t2 = nltk.ParentedTree.fromstring("(S (NP (D the) (N woman)) (AdvP (Adv thoughtfully) (VP (V cooks) (NP meat))))")

# print(t1[(0,)].productions()[0])
# print(t2[(1,)].productions()[0])
#
# print(t2[(0,)].productions()[0])
# print(t2[(1,)].productions()[0])

# Bottom up:

for pos in t1.treepositions(order="postorder"):
    tr = t1[pos]
    if type(tr) != str and len(tr.productions()) > 1:
        print(tr.productions()[0])
        print("Subtree Anfang")
        print(tr)
        print("Subtree Ende")
        count = 0

        # direct descendants
        print(tr[(0,)].productions()[0])
        print(tr[(1,)].productions()[0])

        #
        # for neu in tr.treepositions(order="preorder")[1:]:
        #
        #     desc = tr[neu]
        #     if type(desc) != str:
        #         print(tr[neu].productions()[0])
        #         count += 1
        #         print(count)



            # if count == 2:
            #     break
        #print("new round")


t1.pretty_print()
t2.pretty_print()



def extract_production(t, start, end):
    """
    Returns the (highest) production of a given span.

    :param t: nltk.Tree
    :param start: int
    :param end: int
    :param span: int
    :return: nltk.Production
    """

    if t[t.treeposition_spanning_leaves(start=start, end=end)].height() > ((end-start) + 1):
        return 0

    # The if statement is needed to avoid that a str (i.e. the terminal) is returned. We are interested in the terminal production
    if type(t[t.treeposition_spanning_leaves(start=start, end=end)]) == str:

        # A little hacky
        bord = len(t.treeposition_spanning_leaves(start=start, end=end)) - 1
        return t[t.treeposition_spanning_leaves(start=start, end=end)[:bord]].productions()[0]

    # if t[t.treeposition_spanning_leaves(start=start, end=end)].height() == ((end - start) + 1):
    #     for prod in prods:

    return t[t.treeposition_spanning_leaves(start=start, end=end)].productions()[0]

def production_not_valid(prod, t, span, sent_length):
    """
    Returns True if there is no production at a given span of a given tree. The original sentence length is needed
    because of the implementation of the nltk.Tree method treeposition_spanning_leaves.

    :param prod: nltk.Production
    :param t: nltk.Tree
    :param span: int
    :param sent_length: int
    :return: bool
    """
    return prod == t.productions()[0] and span != sent_length






# t1[t1.treeposition_spanning_leaves(start=3, end=7)].pretty_print()
# print(extract_production(t=t1, start=3, end=7))



def compute_num_matching_subtrees(t1, t2):
    """
    Given two trees (in CNF) returns the number of matching subtrees.

    :param t1: nltk.Tree
    :param t2: nltk.Tree
    :return: int
    """
    # Extract all productions
    prods_1 = t1.productions()
    prods_2 = t2.productions()

    # Initialize the DP table (still a dictionary)
    DP_table = {prod: 0 for prod in set(prods_1 + prods_2)}

    for prod_1 in prods_1:
        for prod_2 in prods_2:
            if prod_1.is_lexical() and prod_1 == prod_2:
                DP_table[prod_1] = 1

    # Algorithm
    for pos1 in t1.treepositions(order="postorder"):
        subtree1 = t1[pos1]
        if type(subtree1) != str and len(subtree1.productions()) > 1: # first condition exclude terminals, the second pre-terminals
            prod_1 = subtree1.productions()[0]

            for pos2 in t2.treepositions(order="postorder"):
                subtree2 = t2[pos2]
                if type(subtree2) != str and len(subtree2.productions()) > 1:
                    prod_2 = subtree2.productions()[0]

                    if prod_1 == prod_2:

                        prev_prod_1_1 = subtree1[(0,)].productions()[0]
                        prev_prod_1_2 = subtree1[(1,)].productions()[0]

                        prev_prod_2_1 = subtree2[(0,)].productions()[0]
                        prev_prod_2_2 = subtree2[(1,)].productions()[0]

                        if prev_prod_1_1 == prev_prod_2_1 and prev_prod_1_2 == prev_prod_2_2:

                            # if DP_table[prev_prod_1_1] > 1:
                            #
                            #     print("INTERESTING1")
                            #     print(prev_prod_1_1)
                            #     print(DP_table[prev_prod_1_1])
                            #     print("INTERESTING1")
                            #
                            # if DP_table[prev_prod_1_2] > 1:
                            #
                            #     print("INTERESTING2")
                            #     print(prev_prod_1_2)
                            #     print(DP_table[prev_prod_1_2])
                            #     print("INTERESTING2")

                            DP_table[prod_1] += (1 + DP_table[prev_prod_1_1]) * (1 + DP_table[prev_prod_1_2])
                            #res += (1 + DP_table[prev_prod_1_1]) * (1 + DP_table[prev_prod_1_2])
                            continue

                        if prev_prod_1_1 == prev_prod_2_1:
                            DP_table[prod_1] += (1 + DP_table[prev_prod_1_1])

                        elif prev_prod_1_2 == prev_prod_2_2:
                            DP_table[prod_1] += (1 + DP_table[prev_prod_1_2])

                        else:
                            DP_table[prod_1] += 1

    # Accounting for terminal productions that occur more than once
    counts = {prod: 0 for prod in set(prods_1 + prods_2)}
    for prod_1 in prods_1:
        for prod_2 in prods_2:
            if prod_1.is_lexical() and prod_1 == prod_2:
                counts[prod_1] += 1

    for key, value in counts.items():
        if value > 1:
            DP_table[key] *= value

    print(DP_table)
    print(sum(DP_table.values()))



















    # for span1 in range(2, max_sent_length+1):
    #     for i in range(max_sent_length+1-span1):
    #
    #         k = i+span1
    #         if k > sent1_length: # To account for cases where sent1_length < max_sent_length
    #             continue
    #
    #         print(i)
    #         print(k)
    #
    #         prod_1 = extract_production(t=t1, start=i, end=k)#, span=span1)
    #
    #         if production_not_valid(prod=prod_1, t=t1, span=span1, sent_length=sent1_length) or prod_1 == 0:
    #             continue
    #
    #         print(prod_1)
    #         for span2 in range(2, max_sent_length+1):
    #             for j in range(max_sent_length+1-span2):
    #
    #                 l = j+span2
    #                 if l > sent2_length: # To account for cases where sent2_length < max_sent_length
    #                     continue
    #
    #                 prod_2 = extract_production(t=t2, start=j, end=l, span=span2)
    #
    #                 if production_not_valid(prod=prod_2, t=t2, span=span2, sent_length=sent2_length) or prod_2 == 0:
    #                     continue
    #
    #                 if prod_1 == prod_2:
    #
    #                     # I NEED TO KNOW WHERE THE TERMINALS WERE AND WHERE NOT
    #                     # KEY: FIND A WAY TO EXTRACT THE TERMINAL PRODUCTIONS TO SAVE THE SPAN==2 PART
    #
    #                     # if span == 2: # pre-terminals
    #                     #
    #                     #     # PROBLEM 1: We should not go over any non terminal in the DP table (but only the relevant ones)
    #                     #     # PROBLEM 2: We get wrong numbers if our non-terminal production is not preterminal
    #                     #
    #                     #
    #                     #     non_terms = prod_1.rhs()
    #                     #
    #                     #     res = 1
    #                     #     for key, value in DP_table.items():
    #                     #         for non_term in non_terms:
    #                     #             if key.is_lexical() and key.lhs() == non_term:
    #                     #                 if value > 1:
    #                     #                     res *= value
    #                     #                 else:
    #                     #                     res *= (1+value)
    #                     #
    #                     #     DP_table[prod_1] += res
    #                     #
    #                     # if span > 2: # non-terminals (except for pre-terminals)
    #
    #                     for m in range(i+1, k):
    #
    #                         yields = prod_1.rhs()
    #
    #                         prev_prod_1_1 = extract_production(t=t1, start=i, end=m, span=span1) # need to be position sensitive
    #                         prev_prod_1_2 = extract_production(t=t1, start=m, end=k, span=span1)
    #
    #                         # if prev_prod_1_1 == t1.productions()[0] or prev_prod_1_2 == t1.productions()[0]:
    #                         #     continue
    #
    #                         if prev_prod_1_1 == 0 or prev_prod_1_2 ==0:
    #                             continue
    #
    #                         for n in range(j+1, l):
    #
    #                             prev_prod_2_1 = extract_production(t=t2, start=j, end=n, span=span2)
    #                             prev_prod_2_2 = extract_production(t=t2, start=n, end=l, span=span2)
    #
    #                             # if prev_prod_2_1 == t2.productions()[0] or prev_prod_2_2 == t2.productions()[0]:
    #                             #     continue
    #
    #                             if prev_prod_2_1 == 0 or prev_prod_2_2 == 0:
    #                                 continue
    #
    #                             #DP_table[prod_1] += (1 + DP_table[prev_prod_1_1]) * (1 + DP_table[prev_prod_1_2])
    #
    #                             if prev_prod_1_1 == prev_prod_2_1 and prev_prod_1_2 == prev_prod_2_2:
    #                                 DP_table[prod_1] += (1 + DP_table[prev_prod_1_1]) * (1 + DP_table[prev_prod_1_2])
    #                                 continue
    #
    #                             if prev_prod_1_1 == prev_prod_2_1:
    #                                 DP_table[prod_1] += (1 + DP_table[prev_prod_1_1])
    #                             elif prev_prod_1_2 == prev_prod_2_2:
    #                                 DP_table[prod_1] += (1 + DP_table[prev_prod_1_2])
    #                             else:
    #                                 DP_table[prod_1] += 1
    #
    #
    # print(DP_table)
    # print(sum(DP_table.values()))

compute_num_matching_subtrees(t1=t2, t2=t1)










# CKY-like algorithm
#     for span1 in range(2, max_sent_length+1):
#         for i in range(max_sent_length+1-span1):
#
#             k = i+span1
#             if k > sent1_length: # To account for cases where sent1_length < max_sent_length
#                 continue
#
#             print(i)
#             print(k)
#
#             prod_1 = extract_production(t=t1, start=i, end=k)#, span=span1)
#
#             if production_not_valid(prod=prod_1, t=t1, span=span1, sent_length=sent1_length) or prod_1 == 0:
#                 continue
#
#             print(prod_1)
#             for span2 in range(2, max_sent_length+1):
#                 for j in range(max_sent_length+1-span2):
#
#                     l = j+span2
#                     if l > sent2_length: # To account for cases where sent2_length < max_sent_length
#                         continue
#
#                     prod_2 = extract_production(t=t2, start=j, end=l, span=span2)
#
#                     if production_not_valid(prod=prod_2, t=t2, span=span2, sent_length=sent2_length) or prod_2 == 0:
#                         continue
#
#                     if prod_1 == prod_2:
#
#                         # I NEED TO KNOW WHERE THE TERMINALS WERE AND WHERE NOT
#                         # KEY: FIND A WAY TO EXTRACT THE TERMINAL PRODUCTIONS TO SAVE THE SPAN==2 PART
#
#                         # if span == 2: # pre-terminals
#                         #
#                         #     # PROBLEM 1: We should not go over any non terminal in the DP table (but only the relevant ones)
#                         #     # PROBLEM 2: We get wrong numbers if our non-terminal production is not preterminal
#                         #
#                         #
#                         #     non_terms = prod_1.rhs()
#                         #
#                         #     res = 1
#                         #     for key, value in DP_table.items():
#                         #         for non_term in non_terms:
#                         #             if key.is_lexical() and key.lhs() == non_term:
#                         #                 if value > 1:
#                         #                     res *= value
#                         #                 else:
#                         #                     res *= (1+value)
#                         #
#                         #     DP_table[prod_1] += res
#                         #
#                         # if span > 2: # non-terminals (except for pre-terminals)
#
#                         for m in range(i+1, k):
#
#                             yields = prod_1.rhs()
#
#                             prev_prod_1_1 = extract_production(t=t1, start=i, end=m, span=span1) # need to be position sensitive
#                             prev_prod_1_2 = extract_production(t=t1, start=m, end=k, span=span1)
#
#                             # if prev_prod_1_1 == t1.productions()[0] or prev_prod_1_2 == t1.productions()[0]:
#                             #     continue
#
#                             if prev_prod_1_1 == 0 or prev_prod_1_2 ==0:
#                                 continue
#
#                             for n in range(j+1, l):
#
#                                 prev_prod_2_1 = extract_production(t=t2, start=j, end=n, span=span2)
#                                 prev_prod_2_2 = extract_production(t=t2, start=n, end=l, span=span2)
#
#                                 # if prev_prod_2_1 == t2.productions()[0] or prev_prod_2_2 == t2.productions()[0]:
#                                 #     continue
#
#                                 if prev_prod_2_1 == 0 or prev_prod_2_2 == 0:
#                                     continue
#
#                                 #DP_table[prod_1] += (1 + DP_table[prev_prod_1_1]) * (1 + DP_table[prev_prod_1_2])
#
#                                 if prev_prod_1_1 == prev_prod_2_1 and prev_prod_1_2 == prev_prod_2_2:
#                                     DP_table[prod_1] += (1 + DP_table[prev_prod_1_1]) * (1 + DP_table[prev_prod_1_2])
#                                     continue
#
#                                 if prev_prod_1_1 == prev_prod_2_1:
#                                     DP_table[prod_1] += (1 + DP_table[prev_prod_1_1])
#                                 elif prev_prod_1_2 == prev_prod_2_2:
#                                     DP_table[prod_1] += (1 + DP_table[prev_prod_1_2])
#                                 else:
#                                     DP_table[prod_1] += 1






















# CKY-like algorithm
#     for span in range(2, max_sent_length+1):
#         for i in range(max_sent_length+1-span):
#
#             k = i+span
#             if k > sent1_length: # To account for cases where sent1_length < max_sent_length
#                 continue
#
#             prod_1 = extract_production(t=t1, start=i, end=k)
#
#             if production_not_valid(prod=prod_1, t=t1, span=span, sent_length=sent1_length):
#                 continue
#
#             for j in range(max_sent_length+1-span):
#
#                 l = j+span
#                 if l > sent2_length: # To account for cases where sent2_length < max_sent_length
#                     continue
#
#                 prod_2 = extract_production(t=t2, start=j, end=l)
#
#                 if production_not_valid(prod=prod_2, t=t2, span=span, sent_length=sent2_length):
#                     continue
#
#                 if prod_1 == prod_2:
#
#                     # I NEED TO KNOW WHERE THE TERMINALS WERE AND WHERE NOT
#                     # KEY: FIND A WAY TO EXTRACT THE TERMINAL PRODUCTIONS TO SAVE THE SPAN==2 PART
#
#                     # if span == 2: # pre-terminals
#                     #
#                     #     # PROBLEM 1: We should not go over any non terminal in the DP table (but only the relevant ones)
#                     #     # PROBLEM 2: We get wrong numbers if our non-terminal production is not preterminal
#                     #
#                     #
#                     #     non_terms = prod_1.rhs()
#                     #
#                     #     res = 1
#                     #     for key, value in DP_table.items():
#                     #         for non_term in non_terms:
#                     #             if key.is_lexical() and key.lhs() == non_term:
#                     #                 if value > 1:
#                     #                     res *= value
#                     #                 else:
#                     #                     res *= (1+value)
#                     #
#                     #     DP_table[prod_1] += res
#                     #
#                     # if span > 2: # non-terminals (except for pre-terminals)
#
#                     for m in range(i+1, k):
#
#                         yields = prod_1.rhs()
#
#                         prev_prod_1_1 = extract_production(t=t1, start=i, end=m) # need to be position sensitive
#                         prev_prod_1_2 = extract_production(t=t1, start=m, end=k)
#
#                         if prev_prod_1_1 == t1.productions()[0] or prev_prod_1_2 == t1.productions()[0]:
#                             continue
#
#                         for n in range(j+1, l):
#
#                             prev_prod_2_1 = extract_production(t=t2, start=j, end=n)
#                             prev_prod_2_2 = extract_production(t=t2, start=n, end=l)
#
#                             if prev_prod_2_1 == t2.productions()[0] or prev_prod_2_2 == t2.productions()[0]:
#                                 continue
#
#                             #DP_table[prod_1] += (1 + DP_table[prev_prod_1_1]) * (1 + DP_table[prev_prod_1_2])
#
#                             if prev_prod_1_1 == prev_prod_2_1 and prev_prod_1_2 == prev_prod_2_2:
#                                 DP_table[prod_1] += (1 + DP_table[prev_prod_1_1]) * (1 + DP_table[prev_prod_1_2])
#                                 continue
#
#                             if prev_prod_1_1 == prev_prod_2_1:
#                                 DP_table[prod_1] += (1 + DP_table[prev_prod_1_1])
#                             elif prev_prod_1_2 == prev_prod_2_2:
#                                 DP_table[prod_1] += (1 + DP_table[prev_prod_1_2])
#
#
#     print(DP_table)
#     print(sum(DP_table.values()))





















# You cannot go with the "siblings" into other productions

# print(t1.productions()[::-1])
#
# prods_1 = t1.productions()[::-1]
#
#
# prods_2 = t2.productions()[::-1]
#
# C = {}
#
# # I need a mapping that tells me given a productions what are the descendents and the current "count"
# for prod_1 in prods_1:
#     for prod_2 in prods_2:
#         C[prod_1] = 0
#         C[prod_2] = 0
#
# print("Initialized")
# print(C)
#
# for prod_1 in prods_1:
#     for prod_2 in prods_2:
#         if prod_1.is_lexical() and prod_1 == prod_2:
#             C[prod_1] += 1
#
# print("terminals done")
# print(C)

# for prod, value in C.items:
#     if value > 0:
#
#
#
#
#
#
#
#         if prod_1.is_nonlexical() and prod_1 == prod_2:
#             # is there any production in C where one of the terms of the RHS from prod of interest is the LHS and has a positive value
#             lookup = {k.lhs():v for k, v in C.items()}
#             print(lookup)
#             print(prod_1.rhs())
#
#             for non_term in prod_1.rhs():
#                 print(non_term in lookup)
#                 if non_term in lookup:
#                     C[prod_1] += lookup[non_term] + 1


            #if C[prod_1] > 0:

# print(C)


# if prod_1.is_nonlexical() and prod_1 == prod_2:
#
#
#
#             for child in t1:
#                 for _ in range(t1.height()):
#                     for chil in child:
#                         print(chil)
#
#             # prod_indices = [pos for pos in t1.treepositions() if type(t1[pos]) != str]
#             # for prod_ind in prod_indices:
#             #     print(t1[prod_ind])
#             #     print(t1[prod_ind] == prod_1)
#
#             C[prod_1] = 2
#
#
#
# print(C)





#t = t1
#for index in
#
# num_levels2 = t2.height()
# num_levels1 = t1.height()
#
#
#
# prod_indices = [pos for pos in t1.treepositions() if type(t1[pos]) != str]

# print(prod_indices)
# print(t[t1.leaf_treeposition(1)])
# #print(t[t1.treepositions()[2]].label())
# print(len(t))
#print(t.treeposition_spanning_leaves(start=0, end=1))

# We really can't go through the subtrees, because these are actually exponentially many!

# for pos in t1.treepositions():
#     prod = t1[pos]
#     if type(prod) == str:
#         print(prod)
#     else:
#         print(prod.label())



#for index in t.lea


# def extract_subtree(tree, start, end):
#     if type(tree[tree.treeposition_spanning_leaves(start=start, end=end)]) == str:
#         return tree[tree.treeposition_spanning_leaves(start=start, end=end)]
#
#     return tree[tree.treeposition_spanning_leaves(start=start, end=end)].productions()










# This gives us all the nodes (we need to work with them)

#for level in range(num_levels2):
# tmp = 0
#
# mark = 0
# mult = 0
# for level in range(max(num_levels1, num_levels2)):
#     mult += 1
#
#     for _ in range(2):
#
#         for prod_index in prod_indices:
#
#             print("---------")
#             print("What we check")
#             print(t1[prod_index])
#             print("---------")
#
#             if mark == 0:
#                 n1 = t2[t2.leaf_treeposition(0)[:(num_levels2-2)]]
#
#             if n1 == t1[prod_index]:
#                 tmp += 1 * (level+1)
#
#             print("What is true")
#             print(n1)
#
#         if n1.right_sibling() is None:
#             continue
#
#         else:
#             n1 = n1.right_sibling()
#             print("one round done, let's check the next one")
#             mark += 1
#             print(tmp)
#
#     if n1.parent() is None:
#         print(tmp)
#
#     else:
#         n1 = n1.parent()

    #     print(n2)
    #     n1 = n2.parent()
    #     mark += 1
    #
    # n1 = n1.right_sibling()

#print(tmp)
    # t1[prod_index] ==


















#
# levels = t.height()
# print(levels)
# leaves_index = levels - 1
#
# prod_indices = [pos for pos in t.treepositions() if len(pos) < leaves_index]
#
# for position in prod_indices:#t.treepositions():
#     print(position)
#     print(t[position])
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# print(t.leaves())
# print(t.leaf_treeposition(0))
#
# print(levels)
# print(t.leaf_treeposition(0)[:levels-2])
# print(t[t.leaf_treeposition(0)[:levels-2]].right_sibling())


# for level in range(levels):
#     for i in range(len(t.leaves())):
#         print(t[t.leaf_treeposition(i)])

#hello = t[t.leaf_treeposition(0)[:2]].parent_index()
#print(t[t.leaf_treeposition(0)[:2]].right_sibling()) # gets the parent of a given node , the indexing depends on how many levels the tree has
#print(t[hello].right_sibling())




# print(len(t.leaves()))
# print(t.leaf_treeposition(3))
# print(t[t.leaf_treeposition(2)].left_sibling())

# for tree in t.subtrees():
#     print(tree)


def rejoin_sentence(sent_list):
    """
    :param sent_list: list of str
    :return: str
    """
    return " ".join(sent_list)


# print(t.pretty_print())
#
# print(t[0, 0])
# print(t[0, 1])
# print(t[1, 0])
#
# print(t[0])
# print(t[1])
# print(t[1][1][1] == nltk.Tree.fromstring("(N cat)"))

#
# grammar = nltk.CFG.fromstring("""
#   S -> NP VP
#   VP -> V NP | V NP PP
#   PP -> P NP
#   V -> "saw" | "ate" | "walked"
#   NP -> "John" | "Mary" | "Bob" | Det N | Det N PP
#   Det -> "a" | "an" | "the" | "my"
#   N -> "man" | "dog" | "cat" | "telescope" | "park"
#   P -> "in" | "on" | "by" | "with"
#   """)
#
# s1 = "Mary saw Bob"
# sent = s1.split()
# rd_parser = nltk.RecursiveDescentParser(grammar)
#
# for tree in rd_parser.parse(sent):
#     print(tree.subtrees()) # g
#     print(tree.leaves())


# s1 = "Mary saw Bob"
# s2 = "John ate Bob"
# s3 = "Mary walked the man"
#
# sent = [s1.split(), s2.split(), s3.split()]
# rd_parser = nltk.RecursiveDescentParser(grammar)
#
# mapping = {}
#
# for s in sent:
#     for tree in rd_parser.parse(s):
#         mapping[rejoin_sentence(s)] = tree
#
# print(mapping)