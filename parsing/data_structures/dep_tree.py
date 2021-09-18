class Node():

    def __init__(self, symbol, pos):
        self.symbol = symbol
        self.pos = pos # We will use the nltk positions here, because the notion of a "span" doesn't make sense for dep trees

    def __repr__(self):
        return f"{self.symbol} at pos {self.pos}"

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.symbol == other.symbol #and self.pos == other.pos # this is a method solely for the C&D constituency tree algorithm (bec. positions don't need to be equal)

    def __hash__(self):
        return hash((self.symbol, self.pos))

class Arc():

    def __init__(self, head, label, tail):
        self.head = head
        self.label = label # a list (because for each element in the tail list, we need a distinct label
        self.tail = tail # a list

    def __repr__(self):

        outp = ""
        for ind, node in enumerate(self.tail):
            outp += f"{self.head}--{self.label[ind]}-->{node}\n"

        return outp

    def __eq__(self, other):

        if self.head != other.head: # since label is a list, they need to be equivalent
            return False
        elif sum(x == y for x, y in zip(self.label, other.label)) == len(self.label) == len(other.label): # Check that labels list are equivalent
            self_tail, other_tail = self.tail, other.tail
            return sum(x == y for x, y in zip(self_tail, other_tail)) == len(self_tail) == len(other_tail) # Check that everything is equal (also ordered)
        else:
            return False

    def __len__(self):
        return len(self.tail)

    def __getitem__(self, item):
        return self.tail[item]

    def __iter__(self):

        for ind, (label, child) in enumerate(zip(self.label, self.tail)):
            yield (ind, label, child)

    def compare_arcs(self, other):
        """
        Build a list of common dependencies of two arcs.

        :param other: Arc
        :return: list
        """

        common_dep = []
        for ind, label, child in self:
            for ind_other, label_other, child_other in other:
                if label == label_other and child == child_other:
                    common_dep.append((ind, ind_other, child))

        return common_dep

    def compare_arcs_new(self, other):
        """
        Build a list of common dependencies of two arcs.

        :param other: Arc
        :return: list
        """

        common_dep = []
        for ind, label, child in self:
            for ind_other, label_other, child_other in other:
                if child == child_other:
                    common_dep.append((ind, ind_other, child))

        return common_dep

class DependencyTree(list):#list):

    def __init__(self, arcs, conllu_str):#root=None, eps=None):
        list.__init__(self, arcs)
        self.arcs_bottom_up = arcs
        self.conllu_str = conllu_str

    def __eq__(self, other):
        if isinstance(other, DependencyTree):
            for ind in range(len(self.arcs_bottom_up)):
                if self.arcs_bottom_up[ind] != other.arcs_bottom_up[ind]:
                    return False
            return True
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return str(self.arcs_bottom_up)

    def __str__(self):
        return str(self.arcs_bottom_up)

    def __iter__(self):
        """
        When iterating a tree, non-terminal nodes and the corresponding productions will be yielded.

        :return: tuple of Node and Production
        """
        for arc in self.arcs_bottom_up:
            yield (arc.head, arc)

    def __hash__(self):
        return hash(self.arcs_bottom_up)

    def words(self):
        """
        Extract all words (i.e. nodes) of the DependencyTree.

        :return: list of Nodes
        """
        words = []
        for arc in self.arcs_bottom_up:#full_arcs:
            for word in arc.tail:
                words.append(word)

        return words

    def nodes(self):
        """

        :return:
        """
        nodes = self.words()
        nodes.append(Node("ROOT", 0))

        return nodes

    def conllu_information(self):
        """

        :return:
        """
        tokens = []

        for arc in self.arcs_bottom_up:  # full_arcs:
            for ind, word in enumerate(arc.tail):
                tokens.append((word.pos, word.symbol, arc.head.pos, arc.label[ind].lower()))

        # Sort according to the addresses
        tokens.sort(key=lambda tup: tup[0])

        return tokens