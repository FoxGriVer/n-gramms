from lm import LanguageModel
from text_preprocessing import TextPrepocessing

import re
import os

# Necessary for unit-tests
def get_input(text):
    return input(text)

# Necessary for unit-tests
def main():
    pass

class Main():
    program_is_over = False

    @property
    def language_model(self):
        return self.__language_model
    @language_model.setter
    def language_model(self, language_model):
        self.__language_model = language_model

    def __init__(self):
        print("Main instance created")

    # Check entered int value
    def enter_int_number(self):
        while True:
            user_input = get_input("Please, enter an intager number:\n")
            try:
                value = int(user_input)
                if (value > 0):
                    return value
                else:
                    print("Entered number is negative")
            except ValueError:
                print("Entered incorrect input")

    # Present information about functionality of the program
    def present_options(self):
        print("\nEnter the number of row for desired operation.")
        print("For generating text, first of all you should create language model. \n")
        print("1. Create a new language model")
        print("2. Generate a text from the language model, and print it to the screen")
        print("3. Generate a user-specified number of texts from the language model, and write them to a file")
        print("4. Create a new language model with smoothing")
        print("5. Exit the program \n")

    # "Switch" imitation
    def choose_option(self):
        entered_option_number = self.enter_int_number()
        switcher = {
            1: self.create_language_model,
            2: self.generate_text,
            3: self.generate_text_and_save,
            4: self.create_language_model_with_smoothing,
            5: self.exit_the_program
        }
        func = switcher.get(entered_option_number, self.option_does_not_exist)
        
        return func()
        
    def create_language_model(self, smoothing = False):
        print("\nCreating a language model")
        print("Enter the number for N-parameter")
        n_parameter = self.enter_int_number()

        if(n_parameter > 10 or n_parameter <= 1):
            print("n-parameter is invalid. Please, enter the value less than 10 and more than 1.")
            return

        self.language_model = LanguageModel(n_parameter)
        if smoothing:
            self.language_model.turn_on_smoothing() 

        valid_file_path = self.find_file()

        with open(valid_file_path, "r") as openedFile:
            full_text = openedFile.read()
            text_preprocessing = TextPrepocessing(full_text)
            tokens = text_preprocessing.start_preprocessing()              
            self.language_model.train(tokens)

    def find_file(self):
        try:
            print("\nEnter the file-path with text for training the language model.")
            print("Or enter \"NONE\" or \"none\" (without \"\") or just press \"Enter\" key for reading train_shakespeare.txt from the same directory with main.py")
            entered_file_path = input()
            if (entered_file_path == "NONE" or entered_file_path == "none" or entered_file_path == ""):      
                # Read the default file from the same directory, if NONE entered  
                __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
                entered_file_path = __location__.replace('\\', '/') + "/train_shakespeare.txt"            
            else:
                # Read file from the entered file-path
                entered_file_path = entered_file_path.replace('\\', '/')   

            with open(entered_file_path, "r") as openedFile:
                print("\nFile succesfully found \n")

            return entered_file_path            
        except FileNotFoundError as ex:
            print("File not found. Please, try one more time")
            self.find_file()

    def generate_text(self, intered_text = None):
        try:
            if (self.language_model == None):
                raise AttributeError()
            print("Enter desired begining of the text")
            print("Or enter \"NONE\" or \"none\" (without \"\") or just press \"Enter\" key for generating random text")
            entered_begining = input()
            generated_text = ""
            if (entered_begining != "NONE" or entered_begining != "none" or entered_begining != ""):
                generated_text = self.language_model.generate(entered_begining.split())    
            else:
                generated_text = self.language_model.generate()

            if generated_text != None:
                print("\nGenerated text:")
                print(generated_text)
            else:
                print("\nUnfortunately with this beginning nothing was found")
            
        except AttributeError:
            print("\nIt is necessary first of all create a language model (option 1)")
            self.create_language_model()

    def generate_text_and_save(self):
        print("\nEnter desired number of texts")
        entered_number_of_texts = self.enter_int_number()
        if (entered_number_of_texts > 0 and entered_number_of_texts < 1000):
            print("Writing texts to a file has started")
            __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
            file_in_root_folder = __location__.replace('\\', '/') + "/new_shakespeare.txt" 
            with open(file_in_root_folder, "w") as created_file:
                i = 1
                while i in range(0, entered_number_of_texts + 1):
                    generated_text = self.language_model.generate()
                    created_file.write("{0}. {1}{2}".format(i, generated_text, "\n"))
                    i += 1
            print("Writing texts to a file has ended")
        else:
            print("Entered inappropriate number")
            self.generate_text_and_save()

    def create_language_model_with_smoothing(self):
        self.create_language_model(smoothing = True)

    def exit_the_program(self):
        print("\nProgram is over.")
        Main.program_is_over = True

    # If entered the not existed number of operation
    def option_does_not_exist(self):
        print("\nThere is no such option. Repeat entering option number again.\n")
        self.choose_option()        

    def start(self):
        while True:
            # Close program if this option picked
            if Main.program_is_over:
                break
            self.present_options()
            self.choose_option()

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    main = Main()
    main.start()