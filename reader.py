import os
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from utilities import *

class ReadDataset:
    """
    Class is used to generate data reader object, which reads raw data and extract required information from them
    """
    def __init__(self, config_parameters: dict):
        """
        Method is utilized as an initializer of the class
        :param config_parameters: all required parameters for the project
        """

        self.configuration = self.set_configuration(config_parameters)
        self.dataset = self.get_dataset()

    @staticmethod
    def set_configuration(parameters: dict) -> dict:
        """
        Method is utilized to extract and generate task-specific parameters according to the provided parameters
        :param parameters: all required parameters for the project
        :return: configuration dictionary that contains required parameters for the specific task
        """
        check_dir(parameters['output_dir'])
        dataset_dir = os.path.join(parameters['input_dir'], f"{parameters['fact_set']}.tsv")
        raw_dataset = os.path.join(parameters['output_dir'], f"{parameters['fact_set']}_raw.pickle")
        return {
            'ds_type': parameters['fact_set'],
            'input_dir': dataset_dir,
            'raw_dir': raw_dataset
        }

    def get_dataset(self) -> dict:
        """
        Method is utilized as a main collector of the data
        :return: dictionary that contains all required information related to data
        """
        data = open(self.configuration['input_dir'], 'r')
        if not os.path.exists(self.configuration['raw_dir']):
            all_tokens = list()
            raw_dict = {'id': list(), 'data': list(), 'label': list(), 'unigrams': list(), 'bigrams': list(),
                        'trigrams': list()}
            sw = stopwords.words('english')
            sw.extend(["'s", "'ve", "'re"])

            for each_line in data:
                data = each_line.split('\t')
                raw_dict['id'].append(data[0])
                raw_dict['data'].append(data[1])
                raw_dict['label'].append(
                    data[2].replace('\n', '') if self.configuration['ds_type'] == 'training' else 'None'
                )
                tokens = word_tokenize(data[1])
                raw_dict['unigrams'].append(tokens)
                all_tokens.extend(tokens)
                raw_dict['bigrams'].append(self.collect_ngrams(tokens, gram_size=2))
                raw_dict['trigrams'].append(self.collect_ngrams(tokens, gram_size=3))

            with open(self.configuration['raw_dir'], 'wb') as data:
                pickle.dump(raw_dict, data)
        with open(self.configuration['raw_dir'], 'rb') as data:
            raw_dict = pickle.load(data)
        return raw_dict

    @staticmethod
    def collect_ngrams(sentence: str, gram_size: int) -> list:
        """
        Method is utilized for collecting n-grams according to the size of gram. Note: window shift is set to 1
        :param sentence: input data to be analyzed
        :param gram_size: size of grams
        :return: list of n-grams which were extracted from the sentence
        """
        bigrams = list()
        for idx in range(0, len(sentence)):
            window = sentence[idx: idx + gram_size]
            if len(window) == gram_size:
                bigrams.append(' '.join(window))
        return bigrams
