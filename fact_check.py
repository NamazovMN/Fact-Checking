import pickle
import unicodedata
from typing import Any

from utilities import *
import bs4
from tqdm import tqdm
from process_sentences import ProcessSentences
import wikipedia as wiki
from bs4 import BeautifulSoup
import requests


class FactChecking:
    """
    Class is utilized as a final step of the project: Fact Checking
    """

    def __init__(self, config_parameters: dict):
        """
        Method is utilized as an initializer of the class
        :param config_parameters: all required parameters for the project
        """
        self.configuration = self.set_configuration(config_parameters)
        self.synonyms = self.set_synonyms()

    @staticmethod
    def set_configuration(parameters: dict) -> dict:
        """
        Method is utilized to extract and generate task-specific parameters according to the provided parameters
        :param parameters: all required parameters for the project
        :return: configuration dictionary that contains required parameters for the specific task
        """
        check_dir(parameters['output_dir'])
        return {
            'ds_type': parameters['fact_set'],
            'sentence_process': ProcessSentences(parameters),
            'wiki_file': os.path.join(parameters['output_dir'], 'wiki_match_data.pickle'),
            'predictions_dir': os.path.join(parameters['output_dir'], 'predictions.pickle')
        }

    def process_scrapping(self) -> list:
        """
        Method is utilized to check and retrieve wikipedia link which is relevant to the provided information
        :return: list of data in required forma
        """
        if not os.path.exists(self.configuration['wiki_file']):
            initial_scrapping_results = list()
            ti = tqdm(iterable=self.configuration['sp'], total=self.configuration['sentence_process'].__len__(),
                      desc=f'Wiki matches are collected for {self.configuration["ds_type"]}', leave=True)
            s_main = 0
            u_main = 0
            s_sec = 0
            u_sec = 0
            for each_data in ti:
                results = wiki.search(each_data['main_info'])
                num_results = len(results)
                each_data['wiki_match_main'] = results[0] if num_results else 'no_match'
                results_second = wiki.search(each_data['secondary_info'])
                num_results_second = len(results_second)
                each_data['wiki_match_secondary'] = results_second[0] if num_results_second else 'no_match'

                initial_scrapping_results.append(each_data)
                if num_results:
                    s_main += 1
                else:
                    u_main += 1

                if num_results_second:
                    s_sec += 1
                else:
                    u_sec += 1
                ti.set_description(f'WIki matches are collected for {self.configuration["ds_type"]}: '
                                   f'Main S/U => {s_main}/{u_main}, Secondary S/U => {s_sec}/{u_sec}')
            with open(self.configuration['wiki_file'], 'wb') as wiki_data:
                pickle.dump(initial_scrapping_results, wiki_data)
        with open(self.configuration['wiki_file'], 'rb') as wiki_data:
            initial_scrapping_results = pickle.load(wiki_data)
        return initial_scrapping_results

    @staticmethod
    def set_synonyms() -> dict:
        """
        Method is utilized to generate synonyms vocabulary. It was generated manually and must be modified in different
        similar tasks
        :return: dictionary of synonyms
        """
        return {
            'nascence place': ['nascence place', 'birth', 'born'],
            'death place': ['death place', 'dead', 'die', 'last place'],
            'stars': ['star', 'starring', 'actor'],
            'team': ["team", "squad"],
            'squad': ["squad", "team"],
            'author': ["author", 'writer', "novelist", 'playwright', 'creator'],
            'foundation place': ['founded', 'foundation', 'base', 'headquarter', 'innovation place'],
            'award': ['award', 'prize', 'honour', 'decoration', 'reward', 'medal'],
            'last place': ['last place', 'death place', 'dead', 'die'],
            'innovation place': ['innovation place', 'innovation', 'revolution', 'transformation', 'revolution',
                                 'headquarter', 'founded', 'foundation place'],
            'better half': ['better half', 'partner', 'wife', 'husband', 'spouse'],
            'honour': ['honour', 'distinction', 'tribute', 'award', 'prize', 'decoration', 'reward'],
            'subsidiary': ['subsidiary', 'successor', 'parent', 'predecessor', 'developer', 'acquire'],
            'generator': ['generator', 'author'],
            'subordinate': ['subordinate', 'fate', 'acquire', 'acquisition'],
            'spouse': ['spouse', 'partner', 'wife', 'husband', 'better half'],
            'birth place': ['birth place', 'nascence place', 'birth', 'born']
        }

    def process_info(self) -> dict:
        """
        Method is utilized as a main function of the fact-checker object, in which all processes are called
        either implicitly or explicitly
        :return: dataset dictionary that contains prediction and target values along with data and its label
        """
        if not os.path.exists(self.configuration['predictions_dir']):
            current_dataset = self.process_scrapping()
            data_dict = {'id': list(), 'data': list(), 'label': list(), 'prediction': list()}

            ti = tqdm(enumerate(current_dataset), total=len(current_dataset), desc='Fact checking:')
            successful = 0
            for idx, each_data in ti:
                label = True if each_data['label'] == '1.0' else False
                match = self.check_information(each_data)
                for key in data_dict.keys():
                    if key != 'prediction':
                        data_dict[key].append(each_data[key])
                    else:
                        data_dict[key].append('1.0' if match else '0.0')

                if match == label:
                    successful += 1

                ti.set_description(
                    f'Fact checking: accuracy => {successful}/{idx + 1} ({(successful / (idx + 1)):.4f})')
            with open(self.configuration['predictions_dir'], 'wb') as data:
                pickle.dump(data_dict, data)
        with open(self.configuration['predictions_dir'], 'rb') as data:
            data_dict = pickle.load(data)

        return data_dict

    def check_information(self, data: dict) -> bool:
        """
        Method is utilized to check single data
        :param data: dictionary that contains all information relevant to the task
        :return: checking results per given data
        """
        check_dict = {'main': dict(), 'secondary': dict()}
        for each in check_dict.keys():
            info_key = 'secondary' if each == 'main' else 'main'
            check_dict[each] = {
                'wiki_link': data[f'wiki_match_{each}'],
                'info': self.standardize(data[f'{info_key}_info']),
                'category': data['category']
            }
        match = self.check_table_match(check_dict)
        if not match:
            for info_type, info_dict in check_dict.items():
                match = self.check_page(info_dict)
                if match:
                    break

        return match

    def check_table_match(self, check_dict: dict) -> bool:
        """
        Method is utilized to check the fact by matching information and table data, which is retrieved by the given
        wiki-link (in the dict)
        :param check_dict: dictionary that contains all checking relevant information
        :return: boolean variable that specifies whether information was found on table or not
        """
        match = False
        table_match = False
        for info_type, info_dict in check_dict.items():
            table_data = self.check_table(info_dict['wiki_link'])
            if table_data == 'no_match':
                match = False
            elif table_data == 'no_table':
                match = self.check_page(info_dict)
                if match:
                    break
            else:
                cat_match, category = self.category_match(table_data, info_dict['category'])
                if cat_match and info_dict['info'] in table_data[category]:
                    table_match = True
                    break
        match = table_match or match

        return match

    def check_page(self, info_dict: dict) -> bool:
        """
        Method is utilized to check the fact by looking for it within the page
        :param info_dict: dictionary that contains all checking-relevant information
        :return: boolean variable specifies whether information was found on the webpage or not
        """
        match = False
        if info_dict['wiki_link'] != 'no_match':
            url = self.get_url(info_dict['wiki_link'])
            s = requests.Session()
            response = s.get(url, timeout=10).text
            soup = BeautifulSoup(response, 'html.parser')
            for paragraph in soup.find_all('p'):

                if info_dict['info'] in paragraph.text:
                    for syn in self.synonyms[info_dict['category']]:
                        if syn in paragraph.text:
                            match = True
                            break
                    if match:
                        break
        return match

    def category_match(self, table_dict: dict, category: str) -> tuple:
        """
        Method is utilized to check the category through the table
        :param table_dict: dictionary which contains table data: keys are labels, values are information
        :param category: category which was extracted from the input sentence
        :return: tuple of matching results in terms of category and the key that was matched (it can be synonym, too)
        """
        match = False
        search_key = None
        for key in table_dict.keys():
            if key in self.synonyms[category]:
                match = True
                search_key = key
                break
        return match, search_key

    @staticmethod
    def standardize(info: str) -> str:
        """
        Method is utilized to put information into standard shape, since there are some data that do not share standard
        format
        :param info: information which form is requested to be checked
        :return: information in the standard form
        """
        if info[-1] == ' ':
            return info[:-1]
        else:
            return info

    @staticmethod
    def get_table_data(table_info: bs4.element.Tag) -> dict:
        """
        Method is utilized to collect table information in the dictionary data structure
        :param table_info: table information which was extracted from the given web page's html
        :return: dictionary data that contains table information
        """
        table_dict = dict()
        for row in table_info.find_all('tr'):
            if len(row) > 1:
                cols = [each.text for each in row.find_all(['th', 'td'])]
                for idx in range(1, len(cols)):
                    table_dict[cols[0].replace('\xa0', ' ').lower()] = unicodedata.normalize(
                        "NFKC", cols[idx].replace('\u200b', '')
                    )
        return table_dict

    def check_table(self, url: str) -> Any:
        """
        Method is utilized to collect table information according to its existence
        :param url: wiki link information that wikipedia's standard page link is generated according to it
        :return: table info if it exists else information about its existence
        """
        if url != 'no_match':
            url = self.get_url(url)
            s = requests.Session()
            response = s.get(url, timeout=10).text
            soup = BeautifulSoup(response, 'html.parser')

            tables = soup.find_all('table', {'class': 'infobox'})
            table_info = tables[0] if tables else 'no_table'
        else:
            table_info = 'no_match'
        return self.get_table_data(table_info) if table_info != 'no_table' and table_info != 'no_match' else table_info

    @staticmethod
    def get_url(wiki_link: str) -> str:
        """
        Method is utilized to generate wikipedia's standard webpage link
        :param wiki_link: information that is utilized to generate wikipedia link
        :return: web link to the wikipedia page
        """
        wiki_info = wiki_link.split(' ')
        wiki_link_page = str()
        for idx, each in enumerate(wiki_info):
            wiki_link_page += f'{each}_' if idx != len(wiki_info) - 1 else f'{each}'
        url = f"https://en.wikipedia.org/wiki/{wiki_link_page}"
        return url
