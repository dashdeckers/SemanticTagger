import random


def get_data(fn='train.conll.txt', perc_train=20):
    """Returns the data, splitted into test/train according to perc_train.

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
