import math
import re

class Bayes_Classifier:

    def __init__(self):

        self.positive_bow = {}
        self.negative_bow = {}

        self.positive_vocabulary_ct = 0
        self.negative_vocabulary_ct = 0

        self.positive_word_ct = 0
        self.negative_word_ct = 0

        self.positive_review_ct = 0
        self.negative_review_ct = 0

        self.total_review_ct = 0

        self.stop_words = []

    def train(self, lines):

        print('******************TRAINING******************')
        # tokenize the data
        movies = self.tokenize(lines)

        for movie in movies:
            is_positive = False
            # check if a movie review is positive or negative
            if movie[0] == '5':
                is_positive = True
                self.positive_review_ct = self.positive_review_ct + 1
            elif movie[0] == '1':
                self.negative_review_ct = self.negative_review_ct + 1
            else:
                print('Invalid rating: ' + movie[0]+' .Continue..')
                continue

            # add words in the review to the vocabulary
            for word in movie[2:]:
                if is_positive:
                    if word in self.positive_bow.keys():
                        self.positive_bow[word] = self.positive_bow[word] + 1
                    else:
                        self.positive_bow[word] = 1
                        # increase positive vocabulary count as a new word is seen
                        self.positive_vocabulary_ct = self.positive_vocabulary_ct + 1

                    # increase positive word count
                    self.positive_word_ct = self.positive_word_ct + 1

                else:
                    if word in self.negative_bow.keys():
                        self.negative_bow[word] = self.negative_bow[word] + 1
                    else:
                        self.negative_bow[word] = 1
                        # increase negative vocabulary count as a new word is seen
                        self.negative_vocabulary_ct = self.negative_vocabulary_ct + 1

                    # increase negative word count
                    self.negative_word_ct = self.negative_word_ct + 1

            # increase total review count
            self.total_review_ct = self.total_review_ct + 1

    def classify(self, lines):
        print('******************TESTING******************')
        # tokenize the data
        movies = self.tokenize(lines)

        labels = []
        p_positive = math.log(self.positive_review_ct / self.total_review_ct)
        p_negative = math.log(self.negative_review_ct / self.total_review_ct)

        for movie in movies:
            p_review_positive = 0
            p_review_negative = 0

            for word in movie[2:]:
                if word in self.positive_bow.keys():
                    p_review_positive = p_review_positive + math.log((self.positive_bow[word] + 1) /
                                            (self.positive_word_ct + self.positive_vocabulary_ct + 1))

                else:
                    p_review_positive = p_review_positive + math.log(1 /
                                            (self.positive_word_ct + self.positive_vocabulary_ct + 1))
                if word in self.negative_bow.keys():
                    p_review_negative = p_review_negative + math.log((self.negative_bow[word] + 1) /
                                            (self.negative_word_ct + self.negative_vocabulary_ct + 1))

                else:
                    p_review_negative = p_review_negative + math.log(1 /
                                            (self.negative_word_ct + self.negative_vocabulary_ct + 1))

            p_review_positive = p_review_positive + p_positive
            p_review_negative = p_review_negative + p_negative

            # assign a class label to the review
            if p_review_positive >= p_review_negative:
                labels.append('5')
            else:
                labels.append('1')

        return labels

    def tokenize(self, lines):
        movies = []

        for i in range(len(lines)):
            line = lines[i].split('|')
            review = self.preprocess(line[2])
            line[2:] = review.split()
            movies.append(line)

        return movies

    def preprocess(self, review):

        # convert words to one case
        review = review.lower()

        # remove any special characters present in the data
        review = re.sub(r'[^a-zA-Z0-9]', ' ', review)

        return review
