import nltk

from fact_check import FactChecking
from utilities import *
nltk.download('wordnet')
nltk.download('punkt')


def __main__():
    parameters = collect_parameters()
    fc = FactChecking(parameters)
    fc.process_scrapping()
    fc.process_info()


if __name__ == '__main__':
    __main__()
