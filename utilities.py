import argparse
import os


def collect_arguments() -> argparse.Namespace:
    """
    Function is utilized user-defined parameters
    :return: Namespace object that includes all arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='dataset', required=False, type=str)
    parser.add_argument('--output_dir', default='output', required=False, type=str)
    parser.add_argument('--fact_set', default='training', choices=['training', 'test'], required=False, type=str)
    return parser.parse_args()


def collect_parameters() -> dict:
    """
    Function is utilized to generate dictionary of parameters
    :return: all required parameters for the project
    """
    arguments = collect_arguments()
    parameters = dict()
    for argument in vars(arguments):
        parameters[argument] = getattr(arguments, argument)
    return parameters


def check_dir(directory: str) -> None:
    """
    Function is utilized to check the existence of the provided path
    :param directory: directory which existence is requested to be checked
    :return: None
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
