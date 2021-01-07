from parse_glove import read_glove_file
from parse_data import get_training_data_generator
from preprocess import preprocess


# create the embedding layer
glove_layer, token_to_idx = read_glove_file()

# get some training data
generator = get_training_data_generator()
labeled_sentence = next(generator)
sentence, labels = list(zip(*labeled_sentence))

# preprocess the training data
model_input = preprocess(sentence, token_to_idx)

# feed to embedding layer and output result
model_output = glove_layer(model_input)

print(sentence, labels)
print(model_input)
print(model_output)
