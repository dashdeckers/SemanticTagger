import torch
import numpy as np


def read_glove_file(fn='glove.42B.300d.txt', n_lines=500_000, n_dims=300):
    """Read the GloVe file and return a Torch layer and the token-index mapping.

    I am using the Common Crawl, 1.9M vocab, uncased GloVe file which has 300
    dimensions per token and a total of 1_917_494 lines, of which 500_000
    should be enough for now (this takes about 30s on my machine).

    This takes the n_lines most frequent tokens right now.
    We could also first scan our dataset and only take the tokens we actually
    need, but this seems better because it's not dataset dependent.

    Args:
        fn (str): The name of the GloVe file.
        n_lines (int): The number of lines to parse.
        n_dims (int): The number of dimensions per token in the file.

    Returns:
        (torch.FloatTensor, dict): The GloVe embedding layer and the
        token-index mapping.
    """
    matrix = np.zeros((n_lines, n_dims))
    token_to_idx = dict()

    with open(fn, 'r') as glove_fileobject:
        for index, line in enumerate(glove_fileobject):

            # stop early
            if index >= n_lines:
                # convert matrix to torch embedding layer
                tensor = torch.FloatTensor(matrix)
                glove = torch.nn.Embedding.from_pretrained(tensor)
                # the embedding layer is fixed, so don't let it train
                glove.weight.requires_grad = False
                return glove, token_to_idx

            # parse the line, fill the matrix
            token, *vector = line.split()
            matrix[index] = np.asarray(vector, dtype='float')
            token_to_idx[token] = index
