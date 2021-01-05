import numpy as np
import torch

# PARSING pre-trained glove:
# generator-parse the glove file, cause way too big for memory
# extract a word-to-index dict (indices are rows of the file)
# build a (n_words, n_dims) matrix where indices are kept (300D)
# TODO: think about how to deal with memory while building the matrix

# create some fake glove data and index mapping
word_to_idx = {'fuck': 0, 'off': 1}
glove_data = torch.FloatTensor(np.array([
    [0.5, 1.5, 2.5],
    [0.0, 1.0, 2.0]
]))

# create the glove layer (it is fixed, so don't let it train)
glove = torch.nn.Embedding.from_pretrained(glove_data)
glove.weight.requires_grad = False

# essentially our preprocessing step ('fuck off' --> [0, 1])
example_sentence = 'fuck off'.split()
input_to_glove = torch.LongTensor(
    [word_to_idx[word] for word in example_sentence]
)

# let the glove layer convert the input into embedded vectors
converted = glove(input_to_glove)

print(example_sentence)
print(converted)
