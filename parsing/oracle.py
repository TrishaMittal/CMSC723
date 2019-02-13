import sys

import nltk
from nltk.corpus import dependency_treebank
from nltk.classify.maxent import MaxentClassifier
from nltk.classify.util import accuracy

VALID_TYPES = set(['s', 'l', 'r'])

class Transition:
    def __init__(self, type, edge=None):
        self._type = type
        self._edge = edge
        assert self._type in VALID_TYPES

    def pretty_print(self, sentence):
        if self._edge:
            a, b = self._edge
            return "%s\t(%s, %s)" % (self._type,
                                     sentence.get_by_address(a)['word'],
                                     sentence.get_by_address(b)['word'])
        else:
            return self._type

def transition_sequence(sentence):
    """
    Return the sequence of shift-reduce actions that reconstructs the input sentence.
    """

    sentence_length = len(sentence.nodes)

    ##### Defining stack and buffer #####
    buffer = []
    stack = [sentence.nodes[0]['address']]
    for ii in range(1, sentence_length):
        buffer.append(sentence.nodes[ii]['address'])

    ##### Helper Functions #####
    def shift_change():
        stack.append(buffer[0])
        buffer.pop(0)
        return stack, buffer

    def left_change():
        edge = (buffer[0], stack[-1])
        stack.pop()
        return stack, edge

    def right_change():
        edge = (stack[-1], buffer[0])
        a = stack.pop()
        buffer.pop(0)
        buffer.insert(0, a)
        return stack, buffer, edge

    def flat_list(list):
        flat_list = [item for sublist in list for item in sublist]
        return flat_list

    def check_one_liner(a, b):
        return any(i in b for i in a)

    ##### Main Logic #####
    while (stack != [] and buffer != [0]):#and len(buffer)!=1 and buffer[0] != 0):
        if stack[-1] in flat_list(sentence.nodes[buffer[0]]['deps'].values()):
            stack, edge = left_change()
            yield Transition('l', (edge[0], edge[1]))
        elif (buffer[0] in flat_list(sentence.nodes[stack[-1]]['deps'].values())) and not check_one_liner(flat_list(sentence.nodes[buffer[0]]['deps'].values()), buffer):
            stack, buffer, edge = right_change()
            yield Transition('r', (edge[0], edge[1]))
        else:
            yield Transition('s')
            stack, buffer = shift_change()
