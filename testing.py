from collins_duffy import extract_production
import nltk

t1 = nltk.ParentedTree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP( (NP (D the) (N cat)) (NP (Adj and) (NP (D the) (N mouse)))))))")
# t1 = nltk.ParentedTree.fromstring("(S (NP (D the) (N dog)) (V barks))") #
# t2 = nltk.ParentedTree.fromstring("(S (NP (D the) (N cat)) (V eats))") #

#t1 = nltk.ParentedTree.fromstring("(S (NP (D the) (N man)) (VP (V eats) (NP fish)))")
# t2 = nltk.ParentedTree.fromstring("(S (NP (D the) (N woman)) (VP (V cooks) (NP meat)))")
#t2 = nltk.ParentedTree.fromstring("(S (NP (D the) (N woman)) (AdvP (Adv thoughtfully) (VP (V cooks) (NP meat))))")
t2 = nltk.ParentedTree.fromstring("(S (NP (D the) (N woman)) (AdvP (Adv thoughtfully) (VP (V cooks) (NP meat))))")

t1.pretty_print()

print(extract_production(tree=t1, start=1, end=2))