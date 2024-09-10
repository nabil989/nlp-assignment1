# models.py

# from .sentiment_data import List
from sentiment_data import *
from utils import *
import numpy as np
import random
from collections import Counter

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.stop_words = ["the", "is", "in", "and", "of", "a", "to", "it"]

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        feature_counter = Counter()
        for word in sentence:
            word = word.lower()
            if word in self.stop_words:
                continue
            elif add_to_indexer:
                word_index = self.indexer.add_and_get_index(word)
            else:
                word_index = self.indexer.index_of(word)
                if word_index == -1:
                    continue
            feature_counter[word_index] += 1
        return feature_counter

class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.stop_words = ["the", "is", "in", "and", "of", "a", "to", "it"]
        
        
    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        feature_counter = Counter()
        n = len(sentence)
        for i in range(n-1):
            start_word = sentence[i].lower()
            adjacent_word = sentence[i+1].lower()
            if start_word in self.stop_words or adjacent_word in self.stop_words:
                continue
            bigram_word = start_word + " " + adjacent_word
            if add_to_indexer:
                word_index = self.indexer.add_and_get_index(bigram_word)
            else:
                word_index = self.indexer.index_of(bigram_word)
                if word_index == -1:
                    continue
            feature_counter[word_index] += 1
            
        return feature_counter


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.stop_words = ["the", "is", "in", "and", "of", "a", "to", "it"]
        self.word_counts = Counter()
        self.min_freq = 1
        self.max_freq = 1000
    
        
        
    def calculate_word_counts(self, train_exs: List[SentimentExample]):
        for ex in train_exs:
            words = [word.lower() for word in ex.words]
            n = len(words)
            for i in range(n):
                # unigrams
                self.word_counts[words[i]] += 1
                # bigrams
                # make sure we don't go out of bounds
                if i+1 < n:
                    bigram = words[i] + " " + words[i+1]
                    self.word_counts[bigram] += 1

                
    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        feature_counter = Counter()
        words = [word.lower() for word in sentence]
        sentence_counter = Counter(words)
        # extract unigrams
        for word, count in sentence_counter.items():
            # skip stop words
            if word in self.stop_words:
                continue
            # skip rare words
            if self.word_counts[word] < self.min_freq:
                continue
            # print(count)
            clipped_count = min(count, self.max_freq)
            if add_to_indexer:
                word_index = self.indexer.add_and_get_index(word)
            else:
                word_index = self.indexer.index_of(word)
                if word_index == -1:
                    continue
            feature_counter[word_index] = clipped_count
        # extract bigram
        n = len(words)-1
        for i in range(n):
            word, adj_word = words[i], words[i+1]
            # skip stop words
            if word in self.stop_words or adj_word in self.stop_words:
                continue
            bigram = word + " " + adj_word
            bigram_count = sum(1 for j in range(n) if words[j] == word and words[j + 1] == adj_word)
            # skip rare words
            if bigram_count < self.min_freq:
                continue
            clipped_bigram_count = min(bigram_count, self.max_freq)
            if add_to_indexer:
                bigram_index = self.indexer.add_and_get_index(bigram)
            else:
                bigram_index = self.indexer.index_of(bigram)
                if bigram_index == -1:
                    continue
            feature_counter[bigram_index] = clipped_bigram_count
        return feature_counter


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights, feat_extractor):
        self.weights = weights
        self.feature_extractor = feat_extractor
        # raise Exception("Must be implemented")
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        feature_vector = self.feature_extractor.extract_features(sentence, add_to_indexer=False)
        
        # score is the dot product of the weights and the feature vector
        score = sum(self.weights[index] * count for index, count in feature_vector.items())
        
        # Return 1 if the score is positive, otherwise return 0
        return 1 if score > 0 else 0


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights, feat_extractor):
        self.weights = weights
        self.feat_extractor = feat_extractor
        # raise Exception("Must be implemented")
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        # extract features from sentence
        feature_vector = self.feat_extractor.extract_features(sentence, add_to_indexer=False)
        # get dot product of weights andn feature vector
        score = sum(self.weights[index] * count for index, count in feature_vector.items())
        
        # apply sigmoid function
        # sigmoid_score = 1 / (1 + np.exp(-score))
        sigmoid_score = sigmoid(score)
        return 1 if sigmoid_score >= 0.5 else 0

def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    for ex in train_exs:
        feat_extractor.extract_features(ex.words, add_to_indexer=True)
    # for ex in train_exs[:10]:  # Limit to 10 for easier inspection
    #     features = feat_extractor.extract_features(ex.words, add_to_indexer=False)
    #     print(f"Example words: {ex.words}")
    #     print(f"Feature vector: {features}")

    
    indexer = feat_extractor.indexer
    vocab_size = len(indexer)
    weights = np.zeros(vocab_size)
    num_epochs = 10
    initial_learning_rate = 1.0
    for epoch in range(num_epochs):
        # randomize data
        random.shuffle(train_exs)

        learning_rate = initial_learning_rate / (epoch+1)
        for ex in train_exs:
            # extract features for cur example
            feature_vector = feat_extractor.extract_features(ex.words, add_to_indexer=False)
            # compute prediction (dot product of weights and feature vector)
            score = sum(weights[index] * count for index, count in feature_vector.items())
            prediction = 1 if score > 0 else 0
            # if the prediction is not the same as label then we update the weights
            if prediction != ex.label:
                for index, count in feature_vector.items():
                    weights[index] += (learning_rate * (ex.label - prediction) * count)
    
    return PerceptronClassifier(weights, feat_extractor)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    # add features to indexer
    for ex in train_exs:
        feat_extractor.extract_features(ex.words, add_to_indexer=True)
    learning_rate = .01
    indexer = feat_extractor.indexer
    vocab_size = len(indexer)
    weights = np.zeros(vocab_size)
    num_epochs = 10
    
    for epoch in range(num_epochs):
        random.shuffle(train_exs)
        for ex in train_exs:
            feature_vector = feat_extractor.extract_features(ex.words, add_to_indexer=False)
            score = sum(weights[index] * count for index, count in feature_vector.items())
            predicted_probability = sigmoid(score)
            # update weights based on prediction
            for index, count in feature_vector.items():
                update = learning_rate * (ex.label - predicted_probability) * count
                weights[index] += update
    return LogisticRegressionClassifier(weights, feat_extractor)

def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model