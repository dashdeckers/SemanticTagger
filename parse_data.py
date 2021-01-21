import xml.etree.ElementTree as ET
import pathlib
import random


def get_data(fn='train.conll.txt', perc_train=20):
    """Returns the data, splitted into test/train according to perc_train.

    This function can parse the data files as they are found in the repo:
    https://github.com/RikVN/DRS_parsing

    Args:
        fn (str): The file name of the data file.
        perc_train (int): The percentage of the dataset to use for training.

    Returns:
        Tuple[List[List[Tuple[str, str]]]]: The (train, test) splitted dataset
            as lists of sentences, where each sentence is a list tuples with
            the words and their corresponding labels.
    """
    sentences = list()
    curr_sentence = list()

    with open(fn, 'r') as data_fileobject:
        for line in data_fileobject:

            # If the line is relevant
            if not (line == '\n' or line.startswith('#')):
                elements = line.split()
                curr_sentence.append((elements[0], elements[2]))

            # If the next sentence starts
            if line.startswith('#') and len(curr_sentence) > 0:
                sentences.append(curr_sentence)
                curr_sentence = list()

    # Shuffle, then split the data into test/train
    random.shuffle(sentences)
    split_idx = int(len(sentences) / 100 * perc_train)

    # (train, test)
    return sentences[:split_idx], sentences[split_idx:]


def get_data_xml(dn='pmb-3.0.0', perc_train=20, datasets=['gold', 'silver']):
    """Returns the data, splitted into test/train according to perc_train.

    This function can parse the original data files as they are found in:
    https://pmb.let.rug.nl/data.php

    The sentences across datasets are combined and shuffled when splitting
    into train/test.

    Args:
        dn (str): The name of the directory containing the data files.
        perc_train (int): The percentage of the dataset to use for training.
        datasets (List[str]): The datasets to include. Can be one or more of
            ['gold', 'silver', 'bronze']

    Returns:
        Tuple[List[List[Tuple[str, str]]]]: The (train, test) splitted dataset
            as lists of sentences, where each sentence is a list tuples with
            the words and their corresponding labels.
    """
    data_directory = pathlib.Path(dn) / 'data' / 'en'

    # Gather all sentences of all datasets together
    sentences = list()
    for dataset in datasets:

        # One sentence per .xml file
        curr_sentence = list()
        for filename in [f for f in (data_directory / dataset).rglob('*.xml')]:

            # Get only the text and the semantic tag of each token
            token_type_pair = list()
            for node in ET.parse(filename).getroot().iter('*'):

                # We assume that 'tok' always comes before 'sem'
                if node.get('type') in ['tok', 'sem']:
                    token_type_pair.append(node.text)

                # When we have a ('tok', 'sem') pair, save it and get another
                if len(token_type_pair) == 2:
                    curr_sentence.append(tuple(token_type_pair))
                    token_type_pair = list()

            sentences.append(curr_sentence)
            curr_sentence = list()

    # Shuffle, then split the data into test/train
    random.shuffle(sentences)
    split_idx = int(len(sentences) / 100 * perc_train)

    # (train, test)
    return sentences[:split_idx], sentences[split_idx:]
