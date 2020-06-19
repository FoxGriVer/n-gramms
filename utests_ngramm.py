from unittest.mock import patch
from text_preprocessing import TextPrepocessing
from lm import LanguageModel

import unittest
import corpus
import main

class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
       print("\nCorpusTests starts")
       print("==========")

    @classmethod
    def tearDownClass(cls):
       print("==========")
       print("CorpusTests has ended")

    def setUp(self):
        self.input_text = "This text is written only for testing. There should not be any stop words at all"
        self.input_tokens = [None, "Simple", "array", "for", None, "testing", "detokenization"]

    def test_simple_tokenize(self):
        print("id: " + self.id())
        result_array_text = ["This", "text", "is", "written", "only", "for",
                        "testing.", "There", "should", "not", "be", "any", "stop", 
                        "words", "at", "all"]
        self.assertEqual(corpus.tokenize(self.input_text), result_array_text)
    
    def test_tokenize_with_stop_words(self):
        print("id: " + self.id())
        stop_words = ["the", "a", "on", "is", "all",
                        "for", "not", "no", "if", "in", "at"]
        result_array_text = ["This", "text", "written", "only", "testing.", "There", 
                        "should", "be", "any", "stop", "words"]
        self.assertEqual(corpus.tokenize(self.input_text, stop_words = stop_words), result_array_text)

    def test_tokenize_with_lemmatization(self):
        print("id: " + self.id())
        self.input_text += " adding words for testing lemmatization functions"
        result_array_text = ["This", "text", "be", "write", "only", "for",
                        "testing.", "There", "should", "not", "be", "any", "stop", 
                        "word", "at", "all", "add", "word", "for", "test",
                        "lemmatization", "function"]
        self.assertEqual(corpus.tokenize(self.input_text ,lemmatization=True), result_array_text)

    def test_detokenize(self):
        print("id: " + self.id())
        result_text = "Simple array for testing detokenization"
        self.assertEqual(corpus.detokenize(self.input_tokens), result_text)

    def test_tokenize_with_stop_words_drop_except(self):
        self.assertRaises(TypeError, 
                        corpus.tokenize(self.input_text, stop_words = True),
                        ["some", "array"])

class TextPreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
       print("\nTextPreprocessingTests starts")
       print("==========")    

    @classmethod
    def tearDownClass(cls):
       print("==========")
       print("TextPreprocessingTests has ended")
    
    def setUp(self):
       input_text = "    Small text for TESTING,    here you can find more than 10 words"
       self.text_preprocessing = TextPrepocessing(input_text)
    
    def test_preprocessing(self):
        print("id: " + self.id())
        result_tokens = ["small", "text", "test", "here",
                        "you", "can", "find", "more", "than", "word"]
        self.assertEqual(self.text_preprocessing.start_preprocessing(), result_tokens)

    def test_preprocessing_delete_whitespaces(self):
        print("id: " + self.id())
        result_tokens = ["Small", "text", "for", "TESTING,", "here",
                        "you", "can", "find", "more", "than", "10", "words"]
        self.assertEqual(self.text_preprocessing.start_preprocessing(extra_whitespace=True, 
                            lowercase = False, numbers = False,
                            special_chars = False, stop_words = False, 
                            lemmatization = False), result_tokens)
    
    def test_preprocessing_numbers(self):
        print("id: " + self.id())
        result_tokens = ["Small", "text", "for", "TESTING,", "here",
                        "you", "can", "find", "more", "than",  "words"]
        self.assertEqual(self.text_preprocessing.start_preprocessing(numbers = True, 
                            extra_whitespace = True, lowercase = False,
                            special_chars = False, stop_words = False, 
                            lemmatization = False), result_tokens)
    
    def test_preprocessing_lower_case(self):
        print("id: " + self.id())
        result_tokens = ["small", "text", "for", "testing,", "here",
                        "you", "can", "find", "more", "than", "10", "words"]
        self.assertEqual(self.text_preprocessing.start_preprocessing(lowercase = True, 
                            extra_whitespace = True, numbers = False,
                            special_chars = False, stop_words = False, 
                            lemmatization = False), result_tokens)
    
    def test_preprocessing_special_chars(self):
        print("id: " + self.id())
        result_tokens = ["Small", "text", "for", "TESTING", "here",
                        "you", "can", "find", "more", "than",  "words"]
        self.assertEqual(self.text_preprocessing.start_preprocessing(special_chars = True, 
                            extra_whitespace = True, lowercase = False,
                            numbers = False, stop_words = False, 
                            lemmatization = False), result_tokens)

class LanguageModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
       print("\LanguageModelTests starts")
       print("==========")    

    @classmethod
    def tearDownClass(cls):
       print("==========")
       print("LanguageModelTests has ended")

    def setUp(self):
        self.lm = LanguageModel(3)
        self.token_sequences = [[ 'the' , 'cat' , 'runs' ],[ 'the' , 'dog' , 'runs' ]]
        self.lm.train(self.token_sequences)
    
    def test_get_ngrams(self):
        print("id: " + self.id())
        self.lm.n = 4
        input_tokens = ['the', 'cat', 'in', 'the', 'hat']
        result_ngrams = [(None , None , None , 'the' ), (None , None , 'the', 'cat'), 
                        (None , 'the' , 'cat' , 'in' ), ( 'the' , 'cat' , 'in' , 'the'), 
                        ('cat' , 'in' , 'the' , 'hat' ), ( 'in' , 'the' , 'hat', None), 
                        ( 'the' , 'hat' , None , None), ( 'hat', None, None, None)]
        self.assertEqual(self.lm.get_ngrams(input_tokens), result_ngrams)
    
    def test_train_vocabulary_and_counts(self):
        print("id: " + self.id())        
        self.assertEqual(self.lm.vocabulary, {None , 'the' , 'cat' , 'runs' , 'dog'})

        result_counts = {(None , None): { 'the' : 2}, (None , 'the' ): { 'cat' : 1, 'dog' : 1}, 
                        ( 'the' , 'cat' ): { 'runs' : 1}, ( 'cat' , 'runs' ): {None: 1}, 
                        ( 'runs' , None): {None: 2}, ( 'the' , 'dog' ): {'runs' : 1}, 
                        ( 'dog' , 'runs' ): {None: 1}}
        self.assertEqual(self.lm.counts, result_counts)
    
    def test_normalize(self):
        print("id: " + self.id())
        input_words = { 'cat' : 1, 'dog' : 1}
        result_probabilities = {'cat' : 0.5, 'dog' : 0.5}
        self.assertEqual(self.lm.normalize(input_words), result_probabilities)
    
    def test_normalize_sum_probabilies(self):
        print("id: " + self.id())
        input_words = { 'cat' : 1, 'dog' : 1}
        probabilities = self.lm.normalize(input_words)

        prob_sum = 0
        for key in probabilities:
            prob_sum += probabilities[key]
        self.assertEqual(prob_sum, 1)
    
    def test_predict_next(self):
        print("id: " + self.id())
        input_tokens = [None, "zero", None, 'the', 'dog']
        result_probabilities = {'runs' : 1}
        self.assertEqual(self.lm.p_next(input_tokens), result_probabilities)

    def test_sample(self):
        print("id: " + self.id())
        input_probability_distribution = {'heads': 0.5, 'tails': 0.5}
        predicted_word = self.lm.sample(input_probability_distribution)[0]
        self.assertIn(predicted_word, input_probability_distribution)

@unittest.skip("Skip MainTests")
class MainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
       print("\nMainTests starts")
       print("==========")

    @classmethod
    def tearDownClass(cls):
       print("==========")
       print("MainTests has ended")

    @patch('main.get_input', return_value = '10')
    def test_enter_int_number(self, input):
        print("id: " + self.id())
        self.assertEqual(main.Main.enter_int_number(input), 10)

    @unittest.skip("Skiped because of loop")
    @patch('main.get_input', return_value = 'Text')
    def test_enter_int_number_value_error(self, input):
        print("id: " + self.id())
        self.assertRaises(ValueError, main.Main.enter_int_number(input))

if __name__ == '__main__':
    tests = unittest.TestSuite()
    tests.addTest(unittest.makeSuite(MainTests))
    tests.addTest(unittest.makeSuite(CorpusTests))
    tests.addTest(unittest.makeSuite(TextPreprocessingTests))
    tests.addTest(unittest.makeSuite(LanguageModelTests))

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(tests)