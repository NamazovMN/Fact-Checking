from reader import ReadDataset
from collections import Counter
from nltk.corpus import stopwords
from string import punctuation


class CategorizeDataset:
    """
    Class is utilized to extract categories from the source dataset
    """
    def __init__(self, config_parameters: dict):
        """
        Method is utilized as an initializer of the class
        :param config_parameters: all required parameters for the project
        """
        self.configuration = self.set_configuration(config_parameters)
        self.categories = self.categorize()

    @staticmethod
    def set_configuration(parameters: dict) -> dict:
        """
        Method is utilized to extract and generate task-specific parameters according to the provided parameters
        :param parameters: all required parameters for the project
        :return: configuration dictionary that contains required parameters for the specific task
        """
        reader_obj = ReadDataset(parameters)
        sw = stopwords.words('english')
        sw.extend(["'re", "'s", "'ve"])
        return {
            'reader': reader_obj,
            'sw': sw,
            'dataset_type': parameters['fact_set']
        }

    def init_scenario(self, cross_check_first: list, cross_check_second: list) -> list:
        """
        Method is utilized to collect initial categories which are easy to extract. Then calls other sessions in
        cascade fashion
        :param cross_check_first: lower level checked data: unigrams vs. bigrams
        :param cross_check_second: higher level checked data: bigrams vs. trigrams
        :return:
        """
        unique_grams = list()
        for each in cross_check_second:
            unique_grams.extend(each)
        for key, value in Counter(unique_grams).items():
            if value > 140 and key not in punctuation and key not in self.configuration['sw']:
                cross_check_first.append(key)
        init_categories = self.clean_tokens(cross_check_first)
        categories = self.eliminate_uppercase(init_categories, unique_grams)
        all_categories = self.eliminate_repetition(categories)

        return all_categories

    def clean_tokens(self, cross_check_first: list) -> list:
        """
        Method is utilized to clean tokens (i.e., discard punctuation, stopwords)
        :param cross_check_first: lower level checked data: unigrams vs. bigrams
        :return:
        """
        keys = list()
        for each in cross_check_first:
            split_info = each.split(' ')
            if len(split_info) == 2:
                if split_info[1] in self.configuration['sw'] or split_info[1] in punctuation:
                    keys.append(split_info[0])
                else:
                    keys.append(each)
            else:
                keys.append(split_info[0])
        return keys

    def eliminate_uppercase(self, categories: list, unique_grams: list) -> list:
        """
        Method is utilized to clean uppercase words from the recent categorization results
        :param categories: list of previous categorization results
        :param unique_grams: list of higher level checked data where each data occurs more than once (makes things easy)
        :return: resulting list of categories after eliminating uppercase words which have higher frequencies
        """
        result = list()
        for each_word in set(categories):
            if each_word in Counter(unique_grams).keys():
                if Counter(unique_grams)[each_word] >= 12 and not each_word[0].isupper():
                    result.append(each_word)
            else:
                result.append(each_word)
        result_categories = self.eliminate_repetition(result)
        return result_categories

    @staticmethod
    def eliminate_repetition(result: list) -> list:
        """
        Method is utilized to eliminate possible repetition from the list (word can hide in the other word)
        :param result: list of recent categorization results
        :return: list of categorization results -> final step of categorization
        """
        removables = dict()
        for word in result:
            for each in result:
                if word != each and word in each:
                    removables[word] = True
        result_categories = [each for each in result if each not in removables.keys()]

        return result_categories

    @staticmethod
    def cross_check(lower: list, higher: list) -> list:
        """
        Method is utilized to compare lower and higher levels grams
        :param lower: lower level grams (e.g., unigrams, when higher level is bigrams)
        :param higher: higher level grams (e.g., bigrams, when lower level is unigrams)
        :return: list of cross-checking results
        """
        results = list()
        for each in lower:
            check_key = f"'s {each}"
            if check_key in higher:
                results.append(f'{each}')
        return results

    def categorize(self) -> list:
        """
        Method is utilized to extract categories from the provided sentences
        :return: list of categories
        """
        dataset = self.configuration['reader'].dataset

        unigrams_cross = list()
        bigrams_cross = list()
        for idx in range(len(dataset['id'])):
            unigrams_cross.extend(self.cross_check(dataset['unigrams'][idx], dataset['bigrams'][idx]))
            bigrams_cross.extend(self.cross_check(dataset['bigrams'][idx], dataset['trigrams'][idx]))

        categories = self.init_scenario(unigrams_cross, bigrams_cross)

        return categories
