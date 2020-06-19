import re
import nltk
from nltk.stem import WordNetLemmatizer

# Return a list of tokens
def tokenize(text, stop_words = None, lemmatization = False):
    tokens = list()

    print("Process of tokenization is started")
    # Break sentence in the token, remove empty tokens
    try:
        for token in text.split(" "):
            if (token != "None"):
                tokens.append(token)
            else:
                tokens.append(None)
    except TypeError:
        print("Not valid format for input text.")

    # Exclude stop-words
    try:
        if stop_words != None:
            tokens = [i for i in tokens if not i in stop_words]
    except TypeError:
        print("Input stop words have not valid format. Must be an array.")

    # Lemmatize our tokens. Longer preprocessing, but increase accuracy
    try:
        if lemmatization:
            lemmatized_tokens = list()
            lemmatizer = WordNetLemmatizer()
            i = 0
            while i < len(tokens):
                if(tokens[i] != None):
                    tokens[i] = lemmatizer.lemmatize(tokens[i], pos="v")
                i += 1
    except Exception as e:
        raise e

    print("Process of tokenization has ended\n")

    return tokens

# Return a single string consisting of all of inputed tokens together
def detokenize(tokens):
    tokens_without_none_values = [token for token in tokens if token is not None]
    return ' '.join(tokens_without_none_values)

