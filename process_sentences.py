from reader import ReadDataset
from categorize import CategorizeDataset
from nltk.tokenize import word_tokenize
from string import punctuation
from utilities import check_dir
import pickle
import os

class ProcessSentences:
    """
    Class is utilized to extract scrapping relevant data, since the project relies on Wikipedia information
    """
    def __init__(self, config_parameters: dict):
        """
        Method is utilized as an initializer of the class
        :param config_parameters: all required parameters for the project
        """
        self.configuration = self.set_configuration(config_parameters)
        self.dataset = self.read_data()

    def set_configuration(self, parameters: dict):
        """
        Method is utilized to extract and generate task-specific parameters according to the provided parameters
        :param parameters: all required parameters for the project
        :return: configuration dictionary that contains required parameters for the specific task
        """
        configuration = dict()
        check_dir(parameters['output_dir'])
        processed_dir = os.path.join(parameters['output_dir'], f"processed_{parameters['fact_set']}.pickle")
        category_extractor = CategorizeDataset(parameters)
        configuration['reader'] = ReadDataset(parameters)
        configuration['categories'] = category_extractor.categories
        configuration['sw'] = category_extractor.configuration['sw']
        configuration['processed_dir'] = processed_dir
        return configuration

    def read_data(self) -> dict:
        """
        Method is utilized to read and extract relevant information for scrapping
        :return: dictionary of the extracted information
        """
        if not os.path.exists(self.configuration['processed_dir']):
            dataset = {
                'id': list(),
                'label': list(),
                'data': list(),
                'category': list(),
                'main_info': list(),
                'secondary_info': list()
            }

            data = self.configuration['reader'].dataset
            for idx, each in enumerate(data['data']):
                label = data['label'][idx]
                data_id = data['id'][idx]
                for k in self.configuration['categories']:
                    if k in each:
                        main_info, secondary_info = self.process_structure(each, k)
                        dataset['main_info'].append(main_info)
                        dataset['secondary_info'].append(secondary_info)
                        dataset['category'].append(k)
                        dataset['id'].append(data_id)
                        dataset['label'].append(label)
                        dataset['data'].append(each)
            with open(self.configuration['processed_dir'], 'wb') as data:
                pickle.dump(dataset, data)
        with open(self.configuration['processed_dir'], 'rb') as data:
            dataset = pickle.load(data)

        return dataset

    def process_structure(self, sentence: str, category: str) -> tuple:
        """
        Method is utilized to process sentences according to the provided categories
        :param sentence: provided input data
        :param category: category that was extracted from the sentence
        :return: tuple that contains main and secondary information from the provided sentence
        """
        category_parts = category.split(' ')
        key = category_parts[1] if len(category_parts) == 2 else category

        tokens = word_tokenize(sentence)
        if tokens[-1] in punctuation:
            sentence = sentence[0: -1]

        tokens = word_tokenize(sentence)
        if tokens[-1] == key:
            main_info, secondary_info = self.process_last(sentence, category)
        else:
            main_info, secondary_info = self.process_mid(sentence, category)
        return main_info, secondary_info

    def process_last(self, sentence: str, category: str) -> tuple:
        """
        Process the sentence which is in form of: secondary info, main info and category
        :param sentence: input sentence
        :param category: category that was extracted from the given sentence
        :return: tuple that contains main and secondary info
        """
        sentence = sentence.replace(category, '')
        sentence = sentence.replace("'s", '')
        info = sentence.split(' is ')

        return info[1], info[0]

    def process_mid(self, sentence: str, category: str) -> tuple:
        """
        Method is utilized to extract information from sentences in form of main info, category, secondary info
        :param sentence: input sentnece
        :param category: category that was extracted from the given sentence
        :return: tuple that contains main and secondary info
        """
        sentence = sentence.replace("'s", '')
        if ' is ' in sentence:
            sentence = sentence.replace(f'{category}', '')
            info = sentence.split(' is ')

        else:
            info = sentence.split(f' {category} ')
        return info[0], info[1]

    def __iter__(self):
        """
        Method is utilized to transform the class to generator
        :return: yields all relevant information from the dataset
        """
        for idx, _ in enumerate(self.dataset['data']):
            yield {k: v[idx] for k, v in self.dataset.items()}

    def __len__(self) -> int:
        """
        Method is utilized to compute the length of the dataset
        :return: length of the dataset
        """
        return len(self.dataset['data'])

