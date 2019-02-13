import operator
import argparse
from collections import defaultdict
from math import log
from nltk import bigrams
from numpy import zeros
from NER_Datasets import ToyDataset, tags,vocabulary, dataset_to_sents_and_tags, \
    CoNLL2003_Train, CoNLL2003_Valid, CoNLL2003_Test

# Utility functions
def argmax(stats):
    assert len(stats) > 0, "Array needs to be non-empty %s" % str(stats)
    max_vals = [x for x in stats if stats[x] == max(stats.values())]
    # break ties lexicographically
    return min(max_vals)

class TaggingPerceptron:
    def __init__(self, vocabulary, tag_set, only_basic_features=True):
        self._tags = list(tag_set)
        print(self._tags)
        self._voc = list(vocabulary)
        self._n = 0
        self._final = False
        self._feature_ids = {}
        self._feature_ids['UNK'] = 0 # for features not seen in training
        self._num_feats = 1
        self.only_basic_features = only_basic_features

        if self.only_basic_features:
            for ii in self._tags:
                for jj in self._tags:
                    self._feature_ids[(ii, jj)] = self._num_feats
                    self._num_feats += 1

            for word in vocabulary:
                for tag in tag_set:
                    self._feature_ids[(tag, word)] = self._num_feats
                    self._num_feats += 1
        else:
            """
            Extra credit: Define custom features for the CoNLL 2003 task
            """
            pass

        self._w = zeros(self._num_feats)
        self._sum = zeros(self._num_feats)

    def __call__(self, feat):
        if feat in self._feature_ids:
            return self._w[self._feature_ids[feat]]
        else:
            return self._w[self._feature_ids['UNK']]

    def feature_vector(self, sentence, tags):
        f = zeros(self._num_feats)
        if self.only_basic_features:
            """
            Task 1: build a feature vector of the basic features
            defined in the constructor.
            """
            for ii in range(len(tags)-1):
                f[self._feature_ids[tags[ii],tags[ii+1]]] += 1
            for ii in range(len(sentence)):
                f[self._feature_ids[tags[ii], sentence[ii]]] += 1
        else:
            """
            Extra Credit: build a feature vector of the custom
            features for the CoNLL task.
            """
            pass
        return f

    def decode(self, sentence):
        """
        Task 2: write code to return the most likely tag sequence
        (a python list) of the givensentence.
        """

        tag_sequence = []
        """
        Task 2: write code to return the most likely tag sequence
        (a python list) of the givensentence.
        """

        mat1 = zeros((len(sentence),len(self._tags)))
        mat2 = zeros((len(sentence),len(self._tags))).astype(int)

        for k in range(len(self._tags)):
            mat1[0][k] = self._w[self._feature_ids[(self._tags[k], sentence[0])]]

        for i in range(len(sentence)-1):
            for j in range(len(self._tags)):
                delta = [mat1[i][t] + self._w[self._feature_ids[(self._tags[j], sentence[i+1])]] + \
                                     self._w[self._feature_ids[(self._tags[t], self._tags[j])]] for t in range(len(self._tags))]
                mat1[i+1][j] = max(delta)
                mat2[i+1][j] = delta.index(max(delta))

        values = mat1[len(sentence)-1]
        max_value = max(values)
        max_index = values.argmax()
        tag_sequence.append(self._tags[max_index])

        i = len(sentence)-1
        while(i!=0):
            tag_sequence.append(self._tags[mat2[i][max_index]])
            max_index = mat2[i][max_index]
            i-=1

        return list(reversed(tag_sequence))

    def finalize(self):
        """
        We're done with learning, set the weight parameter to the averaged
        perceptron weight
        """

        self._final = True
        #Student Task 3: update part 2
        self._w = self._sum / float(self._n)

    def update(self, sentence, predicted, gold):

        assert not self._final, "Cannot update once we've finalized weights"

        """
        Task 3: update the weights vector w based on the given
        sentence, predicted tags and gold tags.
        """
        predict = defaultdict(lambda:0)
        truth = defaultdict(lambda:0)
        for ii in range(len(predicted)-1):
            predict[self._feature_ids[predicted[ii],predicted[ii+1]]] += 1
        for ii in range(len(sentence)):
            predict[self._feature_ids[predicted[ii], sentence[ii]]] += 1

        for ii in range(len(gold)-1):
            truth[self._feature_ids[gold[ii],gold[ii+1]]] += 1
        for ii in range(len(sentence)):
            truth[self._feature_ids[gold[ii], sentence[ii]]] += 1
        shared = predict.keys() & truth.keys()

        for i in shared:
            a = abs(predict[i] - truth[i])
            predict[i] -= a
            truth[i] -= a

        for i in predict.keys() :
            self._w[i] += (-1) * predict[i]
        for i in truth.keys() :
            self._w[i] += (1) * truth[i]

        self._sum += self._w
        self._n += 1
        return self._w

    def accuracy(self, test_sents, test_tags):
        total = 0
        right = 0
        confusion = defaultdict(dict)
        word = defaultdict(dict)

        for ss, tt in zip(test_sents, test_tags):
            pred = self.decode(ss)

            for ww, pp, gg in zip(ss, pred, tt):
                total += 1
                if pp != gg:
                    confusion[pp][gg] = confusion[pp].get(gg, 0) + 1
                    word[(ww, pp)][gg] = word[(ww, pp)].get(gg, 0) + 1
                else:
                    right += 1
        return float(right) / float(total), confusion, word

    def pretty_weights(self, weight_vector=None):
        """
        Create a vector with real features as keys of dictionary
        """

        if weight_vector is None:
            weight_vector = self._w
        d = {}

        for ii in self._feature_ids:
            val = weight_vector[self._feature_ids[ii]]
            if val != 0.0:
                d[ii] = val
        return d

    def train(self, iters, train_sents, train_tags, test_sents, test_tags,
              report_every=500):

        for ii in range(iters):
            for ss, tt in zip(train_sents, train_tags):
                pred = self.decode(ss)
                assert len(pred) == len(ss), "Pred tag len mismatch"
                assert len(ss) == len(tt), "Gold tag len mismatch"

                self.update(ss, pred, tt)
                if self._n % report_every == 1:
                    test_accuracy, confusion, word_errors = \
                        self.accuracy(test_sents, test_tags)
                    print("\t".join(ss))
                    print("\t".join(tt))
                    print("\t".join(pred))
                    print("After %i sents, accuracy is %f, nonzero feats %i" %
                          (self._n, test_accuracy,
                           sum(1 for x in self._w if x != 0.0)))

        self.finalize()
        test_accuracy, confusion, word_errors = \
            self.accuracy(test_sents, test_tags)
        print("---------------")
        print("Final accuracy: %f" % test_accuracy)


if __name__ == "__main__":

    #CoNLL
    conll_train = CoNLL2003_Train()
    conll_valid = CoNLL2003_Valid()
    train_sents, train_tags = dataset_to_sents_and_tags(conll_train)

    valid_sents, valid_tags = dataset_to_sents_and_tags(conll_valid)

    tp = TaggingPerceptron(vocabulary(conll_train),
                       tags(conll_train))

    itrs = 5
    tp.train(itrs, train_sents, train_tags,valid_sents, valid_tags)
