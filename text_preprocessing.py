import re
from corpus import tokenize

class TextPrepocessing():

    @property
    def input_text(self):
        return self.__input_text
    @input_text.setter
    def input_text(self, input_text):
        self.__input_text = input_text

    @property
    def processed_text(self):
        return self.__processed_text
    @processed_text.setter
    def processed_text(self, processed_text):
        self.__processed_text = processed_text
    
    @property
    def stop_words(self):
        return self.__stop_words
    @stop_words.setter
    def stop_words(self, stop_words):
        self.__stop_words = stop_words

    def __init__(self, input_text):
        self.__input_text = input_text
        self.__processed_text = ""
        self.__stop_words = None

    def start_preprocessing(self, extra_whitespace = True, 
                            lowercase = True, numbers = True,
                            special_chars = True, stop_words = True, 
                            lemmatization = True):
        self.processed_text = self.input_text
        if lowercase == True:
            self.processed_text = self.processed_text.lower()
        if numbers == True:
            self.processed_text = self.replace_numbers(self.processed_text)
        if special_chars == True:
            self.processed_text = self.remove_special_chars(self.processed_text)
        if extra_whitespace == True:
            self.processed_text = self.remove_whitespace(self.processed_text)   
        if stop_words == True:
            self.init_stop_words()            

        tokens = tokenize(self.processed_text, self.stop_words, lemmatization)

        return tokens
    
    # Remove extra whitespaces from the text
    def remove_whitespace(self, text):
        text = text.lstrip()
        text = text.rstrip("\r\t")
        text = re.sub("\\n", " None ", text)
        # replace series of spaces with single space
        text = re.sub(' +', ' ', text) 
      
        return text
    
    # Replace all digits
    def replace_numbers(self, text):
        text = re.sub(r"\d+", "", text)
        return text
    
    def remove_special_chars(self, text):
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text

    # Exclude words from words list for tokenising
    def init_stop_words(self):
        self.stop_words = ["the", "a", "on", "is", "all", "for", "not", "no", "if", "in"]
        
        