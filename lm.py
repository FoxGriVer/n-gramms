from corpus import *
from itertools import chain
from random import choices

class LanguageModel:    

    # Contain all words which occur in sequences 
    @property
    def vocabulary(self):
        return self.__vocabulary
    @vocabulary.setter
    def vocabulary(self, vocabulary):
        self.__vocabulary = vocabulary

    # Dict of dicts
    @property
    def counts(self):
        return self.__counts
    @counts.setter
    def counts(self, counts):
        self.__counts = counts
        
    @property
    def n(self):
        return self.__n
    @n.setter
    def n(self, n):
        self.__n = n

    # Smoothing option (False by default)
    @property
    def smoothing(self):
        return self.__smoothing  
    
    @property
    def smoothed_distribution(self):
        return self.__smoothed_distribution
    @smoothed_distribution.setter
    def smoothed_distribution(self, smoothed_distribution):
        self.__smoothed_distribution = smoothed_distribution

    def __init__(self, n):
        self.__n = n
        self.__vocabulary = set()
        self.__counts = dict()
        self.__smoothing = False
    
    def turn_on_smoothing(self):
        self.__smoothing = True
    
    def turn_off_smoothing(self):
        self.__smoothing = False
    
    # Return a list of n-grams present in a token sequence
    def get_ngrams(self, tokens):
        # Used for unseen words in training vocabularies
        UNK = None
        for i in range(0, self.n - 1):
            tokens.insert(0, UNK)
            tokens.append(UNK)

        # Will generate sequences of tokens starting from different
        # elements of the list of tokens.
        sequences = [tokens[i:] for i in range(self.n)]
        
        # The zip function takes the sequences as a list of inputs
        ngrams = zip(*sequences)

        # Concatentate the tokens into ngrams and return
        return list(ngrams)

    # Train the language model
    def train(self, token_sequences):
        print("Process of training is started")
        ngrams_lists = list()
        # Check the input data
        try:
            if(isinstance(token_sequences[0], list)):
                for token_sequence in token_sequences:
                    ngrams_lists.append(self.get_ngrams(token_sequence))
            else: 
                ngrams_lists.append(self.get_ngrams(token_sequences))
        except IndexError:
            print("Some problems with array size")   

        for ngrams in ngrams_lists:            
            # Add unique values in the vocabulary
            self.vocabulary = self.vocabulary.union(chain(*ngrams))

            # If smoothing option is Turned On, than also count smoothed_distribution
            if (self.smoothing):
                print("Smoothing is turned on")
                self.__smoothed_distribution = nltk.KneserNeyProbDist(
                                                nltk.FreqDist([ngram[-3:] for ngram in ngrams]),
                                                discount=0.05)
                
            # Count the number of times next_word follows by tuple_ngram prefix
            for ngram in ngrams:   
                # Take first n - 1 tokens
                tuple_ngram = (ngram[:-1])
                # Take the last token
                next_word = ngram[-1]

                if (self.counts.get(tuple_ngram) != None):
                    if (self.counts[tuple_ngram].get(next_word) != None):
                        self.counts[tuple_ngram][next_word] += 1
                    else:
                        self.counts[tuple_ngram][next_word] = 1
                else:
                    self.counts[tuple_ngram] = {next_word: 1}
        print("Process of training has ended\n")                    
        
    # Count probabilities for input words
    def normalize(self, word_counts):
        probabilities = dict(word_counts)
        counts_token_sequence = 0

        # Sort dict by ascending value
        probabilities = {k: v for k, v in sorted(probabilities.items(), key=lambda item: item[1])}

        for value in probabilities.values():
            counts_token_sequence += value

        for word in word_counts:
            probabilities[word] = float('{:.5f}'.format(word_counts[word] / counts_token_sequence))

        return probabilities

    # Return the estimated probability distribution for the next word 
    # that occurs after the input token sequence
    def p_next(self, tokens):
        estimated_probabilities = dict()        
        # Take the last (n - 1) tokens
        token_sequence = tuple(tokens[-(self.n - 1):])
        
        if(self.counts.get(token_sequence) != None):            
            estimated_probabilities = self.normalize(self.counts[token_sequence])
        else:
            # If the final (n − 1) tokens do not occur in self.counts:
            # Find the first key which end on the same word as 
            # key (n − 1) token
            for key in self.counts:
                if (key[-1] == tokens[-1]):
                    return self.normalize(self.counts[key])             
        
        return estimated_probabilities

    # Return a key from the input dict, chosen according
    # to its probability
    def sample(self, probability_distribution):
        try:
            probabilities = list()
            words = list()
            probability_number = 0
            for word in probability_distribution:
                words.append(word)
                probability_number += probability_distribution[word]
                probabilities.append(probability_distribution[word])

            # Return a random word, the higher the probability, 
            # the higher the chance to return this word
            return choices(words, probabilities)
        except IndexError:      
            print("Probabilities empty")      
    
    # Generate a random token sequence according to the underlying 
    # probability distribution
    def generate(self, text_for_prediction = None):
        try:
            print("Process of generating is started")
            tokens = text_for_prediction
            # Initialize list if input text_for_prediction is empty
            if (tokens == None):
                tokens = [None] * (self.n - 1)
            # If there is no (n − 1) tokens in input text_for_prediction
            if (len(tokens) < (self.n - 1)):
                for i in range((self.n - 1) - len(tokens)):
                    tokens.insert(0, None)    

            while True:
                probabilities = None
                if(self.smoothing):
                    probabilities = self.p_next_with_smoothing(tokens)
                else:
                    probabilities = self.p_next(tokens)
                predicted_word = self.sample(probabilities)[0]

                if (predicted_word != None):
                    tokens.append(predicted_word)
                else:
                    break            

            print("Process of generating has ended\n")
            
            return detokenize(tokens)
        except:
            return None
    
    # Use Kneser-Ney smoothing
    def p_next_with_smoothing(self, tokens):     
        estimated_probabilities = dict()        
        # Take the last (n - 1) tokens
        token_sequence = tokens[-(self.n - 1):]   

        prob_sum = 0

        for sample in self.__smoothed_distribution.samples():
            if sample[0] == token_sequence[0] and sample[1] == token_sequence[1]:
                prob_sum += self.__smoothed_distribution.prob(sample)
                # print("{0}:{1}".format(sample, self.__smoothed_distribution.prob(sample)))
                estimated_probabilities[sample[-1]] = float('{:.5f}'.format(self.__smoothed_distribution.prob(sample)))
        
        # print(prob_sum)
        if(estimated_probabilities.get(None) != None):
            estimated_probabilities[None] += (1 - prob_sum)
        else:
            estimated_probabilities[None] = (1 - prob_sum)

        return estimated_probabilities
