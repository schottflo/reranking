import os
import string
from cfgen import GrammarModel, clean_output_text
from stat_parser import parser

PATH = os.getcwd()

def generate_random_sentence(grammar, ref_sentence):
    """
    Given a cfgen.GrammarModel generate a random sentence from the inferred grammar.

    :param grammar: cfgen.GrammarModel
    :param ref_sentence: str
    :return: str
    """
    # Generate sentence based on a reference sentence
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
    # Initialize parser
    parser_model = parser.Parser()

    # Use CKY parsing to parse the sentence (need this specific function to make sure that parse is in CNF)
    parse = parser_model.norm_parse(sentence)

    # Transform parse into NLTK tree
    parse_tree = parser.nltk_tree(parse)
    return parse_tree

def generate_random_parse_trees(num_trees=2, corpus_file="war_and_peace_short.txt"):
    """
    Generate num_trees parse trees based on a grammar induced from corpus_file.
    
    :param num_trees: int
    :param corpus_file: str
    :return: list of nltk.Trees
    """
    # Construct the path to the data relative to the current location
    dir_path_list = PATH.split(sep="\\")
    dir_path = "\\".join(dir_path_list[:(len(dir_path_list) - 1)])

    # Build the a grammar based on the corpus
    grammar = GrammarModel(dir_path + "\\data\\" + corpus_file, 3)
    print("Grammar created\n")

    # Define reference sentence for sentence generator
    ref_sentence = 'The dog chased the cat and the mouse.'

    parse_trees = []
    for num in range(num_trees):
        sentence = generate_random_sentence(grammar=grammar, ref_sentence=ref_sentence) # Generate sentence
        parse_tree = parse_sentence(sentence=sentence) # Parse it
        parse_trees.append(parse_tree)
        print(f"Parse Tree {num+1}/{num_trees} created")

    return parse_trees

if __name__ == "__main__":
    trees = generate_random_parse_trees()
    for tree in trees:
        tree.pretty_print()