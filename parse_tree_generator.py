import os
import string
from copy import deepcopy
from nltk.treetransforms import collapse_unary, chomsky_normal_form
from cfgen import GrammarModel, clean_output_text
from stat_parser import Parser

PATH = os.getcwd() + "\corpora"

def generate_random_sentence(grammar, ref_sentence):
    """
    Given a cfgen.GrammarModel generate a random sentence from the inferred grammar.

    :param grammar: cfgen.GrammarModel
    :param ref_sentence: str
    :return: str
    """
    sentence = grammar.make_sentence(fixed_grammar=True, sample_sentence=ref_sentence, do_markov=True)

    # Correct the sentence and remove punctuation
    corrected_sentence = clean_output_text(sentence).translate(str.maketrans('', '', string.punctuation))

    return corrected_sentence

def parse_sentence(sentence):
    """
    Parse a sentence and transform the resulting tree into CNF.

    :param sentence: str
    :return: nltk.Tree
    """
    parser = Parser()

    original_parse_tree = parser.parse(sentence)
    parse_tree = deepcopy(original_parse_tree)

    collapse_unary(parse_tree)
    chomsky_normal_form(parse_tree)

    return parse_tree

def generate_random_parse_trees(num_trees=2, corpus_file="war_and_peace.txt"):
    """
    Generate num_trees parse trees based on a grammar induced from corpus_file.
    
    :param num_trees: int
    :param corpus_file: str
    :return: list of nltk.Trees
    """
    grammar = GrammarModel(PATH + "\\" + corpus_file, 3)
    print("Grammar created\n")
    ref_sentence = 'The dog chased the cat and the mouse.'

    parse_trees = []
    for num in range(num_trees):
        sentence = generate_random_sentence(grammar=grammar, ref_sentence=ref_sentence)
        parse_tree = parse_sentence(sentence=sentence)
        parse_trees.append(parse_tree)
        print(f"Parse Tree {num+1}/{num_trees} created")

    return parse_trees

if __name__ == "__main__":
    trees = generate_random_parse_trees(num_trees=2)
    for tree in trees:
        tree.pretty_print()