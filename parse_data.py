def get_training_data_generator(fn='train.conll.txt'):
    """Returns a generator that iterates over sentences and labels.

    Args:
        fn (str): Description of parameter `fn`.

    Returns:
        generator: A generator that iterates over sentences and labels.

    Examples
        >>> generator = get_training_data()
        >>> print(next(generator))
    """
    batch = list()
    with open(fn, 'r') as data_fileobject:
        for line in data_fileobject:

            # if the line is relevant
            if not (line == '\n' or line.startswith('#')):
                elements = line.split()
                batch.append((elements[0], elements[2]))

            # if the next sentence starts
            if line.startswith('#') and len(batch) > 0:
                yield batch
                batch = list()
