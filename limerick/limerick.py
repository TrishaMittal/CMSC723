# Author: Trisha Mittal
# Date: September 6, 2018

# Use word_tokenize to split raw text into words
import string
from string import punctuation
import nltk
from nltk.tokenize import word_tokenize
import re

class LimerickDetector:

    def __init__(self):
        """
        Initializes the object to have a pronunciation dictionary available
        """
        self._pronunciations = nltk.corpus.cmudict.dict()

    def num_syllables(self, word):
        """
        Returns the number of syllables in a word.  If there's more than one
        pronunciation, take the shorter one.  If there is no entry in the
        dictionary, return 1.
        """
        try:
                return min([len(list(y for y in x if y[-1].isdigit())) for x in self._pronunciations[word.lower()]])
        except KeyError:
                #if word not found in cmudict
                return 1

    def apostrophe_tokenize(self, line):
        """
	    Extra Credit
	    """
        line = line.strip(" ")
        word = line.split(' ')[-1]
        word = re.sub('\W+','', word)
        return word


    def guess_syllables(self, word):
        """
    	Returns the syllables, if the word is unknown
    	Source: https://www.howmanysyllables.com/howtocountsyllables
    	"""
        count = 0
        word = word.lower()
        for i in word:
            if i in "aeiou":
                count+=1
        y_rule = re.compile('\w*y')
        if y_rule.findall(word):
            count+=1
        e_rule = re.compile('\w*e')
        if e_rule.findall(word):
            count-=1
        two_vowel_rule = re.compile('r\w*(au|oy|oo)\w*')
        if two_vowel_rule.findall(word):
            count-=1
        three_vowel_rule = re.compile('r\w*(iou)\w*')
        if three_vowel_rule.findall(word):
            count-=2
        return count

    def remove_punctuation(self, value):
        result = ''
        for c in value:
            if c not in string.punctuation:
                result += c
        return result

    def remove_first_consonant_sounds(self, word):

        pos = []
        p = self._pronunciations[word]
        for i in range(len(p)):
            for j in range(len(p[i])):
                if p[i][j][-1].isdigit() == True:
                    pos.append(j)
                    break
        p_cut = []
        for i in range(len(p)):
            p_cut.append(p[i][pos[i]:])
        return p_cut

    def join_strings(self, cut_phenomes):
        """
        Joins the pronunciations stored as lists into strings.
        Makes it easier for suffix check.
        """
        string = []
        for i in range(len(cut_phenomes)):
            string.append(''.join(cut_phenomes[i]))
        return string

    def compare_syllables(self, a,b):
        """
        Checks the number of syllables in two words
        """
        num_syllables_a = self.num_syllables(a)
        num_syllables_b = self.num_syllables(b)

        if num_syllables_a == num_syllables_b:
            return 0
        elif num_syllables_a < num_syllables_b:
            return 1
        else:
            return 2

    def rhymes(self, a, b):
        """
        Returns True if two words (represented as lower-case strings) rhyme,
        False otherwise.
        """
        p_a = self._pronunciations[a]
        p_b = self._pronunciations[b]
        comp = self.compare_syllables(a, b)

        if comp == 0:
            # Now, check if phenomes_a_cut and phenomes_b_cut match
            p_a_cut = self.remove_first_consonant_sounds(a)
            p_b_cut = self.remove_first_consonant_sounds(b)
            p_a_string = self.join_strings(p_a_cut)
            p_b_string = self.join_strings(p_b_cut)
            if set(p_a_string).intersection(set(p_b_string)):
                return True
            else:
                return False
        if comp == 1:
            p_a_cut = self.remove_first_consonant_sounds(a)
            p_a_string = self.join_strings(p_a_cut)
            p_b_string = self.join_strings(p_b)

            for i in p_a_string:
                for j in p_b_string:
                    if j.endswith(i):
                        return True
            return False

        if comp == 2:
            p_b_cut = self.remove_first_consonant_sounds(b)
            p_a_string = self.join_strings(p_a)
            p_b_string = self.join_strings(p_b_cut)

            for i in p_b_string:
                for j in p_a_string:
                    if j.endswith(i):
                        return True
            return False

    def is_limerick(self, text):
        """
        Takes text where lines are separated by newline characters.  Returns
        True if the text is a limerick, False otherwise.

        A limerick is defined as a poem with the form AABBA, where the A lines
        rhyme with each other, the B lines rhyme with each other (and not the A
        lines).

        (English professors may disagree with this definition, but that's what
        we're using here.)
        """
        a = list(filter(bool, text.splitlines()))
        if len(a)>5:
            a = a[:5]
        c = []
        for i in a:
            i = self.remove_punctuation(i)
            c.append(nltk.word_tokenize(i)[-1])
            # c.append(self.apostrophe_tokenize(i))
        if len(c) < 5:
            return False
        if self.rhymes(c[0], c[1]) and self.rhymes(c[1],c[4]) and self.rhymes(c[2], c[3]):
            return True
        else:
            return False

if __name__ == "__main__":
    buffer = ""
    inline = " "
    while inline != "":
        buffer += "%s\n" % inline
        inline = input()

    ld = LimerickDetector()

    print("%s\n-----------\n%s" % (buffer.strip(), ld.is_limerick(buffer)))
