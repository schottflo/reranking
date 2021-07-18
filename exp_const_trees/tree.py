
class Node:

    def __init__(self, symbol, pos):
        self.symbol = symbol
        self.pos = pos

    def __repr__(self):
        return f"{self.symbol} with span {self.pos}"

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.symbol == other.symbol # this is a method solely for the C&D constituency tree algorithm (bec. positions don't need to be equal)

    def __hash__(self):
        return hash((self.symbol, self.pos))


class Production:

    def __init__(self, start, end):
        self.head = start
        self.tail = end

    def __repr__(self):
        return f"{self.head}->{self.tail}"

    def __eq__(self, other):

        if self.head != other.head:
            return False
        else:
            self_tail, other_tail = self.tail, other.tail
            return sum(x == y for x, y in zip(self_tail, other_tail)) == len(self_tail) == len(other_tail) # Check that everything is equal (also ordered)

    def __len__(self):
        return len(self.tail)

    def __getitem__(self, item):
        return self.tail[item]


class ConstituencyTree(list):#list):

    def __init__(self, productions):#root=None, eps=None):
        list.__init__(self, productions)
        self.productions_bottom_up = productions

    def __eq__(self, other):
        if isinstance(other, ConstituencyTree):
            for ind in range(len(self.productions_bottom_up)):
                if self.productions_bottom_up[ind] != other.productions_bottom_up[ind]:
                    return False
                return True
        else:
            return False

    def __repr__(self):
        return self.productions_bottom_up

    def __str__(self):
        return self.productions_bottom_up

    def __iter__(self):
        """
        When iterating a tree, non-terminal nodes and the corresponding productions will be yielded.

        :return: tuple of Node and Production
        """
        for prod in self.productions_bottom_up:
            yield (prod.head, prod)

    def nonterminals(self):
        """
        Extract the nonterminals.

        :return: list of Nodes
        """
        return [prod.head for prod in self.productions_bottom_up]

    def terminals(self):
        """
        Extract the terminals.

        :return: list of Nodes
        """
        terminals = []

        for prod in self.productions_bottom_up:

            if len(prod) == 1:
                terminals.append(prod.tail[0])

        return terminals


