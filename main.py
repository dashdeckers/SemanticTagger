import random
import pprint

import numpy as np
import torch
import stanza
import matplotlib.pyplot as plt

from tag_dict import tag_to_idx
from parse_glove import read_glove_file
from parse_data import get_data
from preprocess import preprocess
from model import SemTag


# Set the random seed for reproducibility
random.seed(111)

# If running this script for the first time on your computer, uncomment this
# stanza.download('en')

# Create the embedding layer
glove_layer, token_to_idx = read_glove_file()

# Initialize the model and the tokenizer
model = SemTag(glove_layer, len(tag_to_idx))
tokenizer = stanza.Pipeline('en', processors='tokenize')

# Define the optimizer and the loss function
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss()

# Get the data and define loop vars
train, test = get_data(perc_train=80)
max_epochs = 10
losses = []

# Start training loop
for epoch in range(max_epochs):
    # Reset the running loss and shuffle the training data
    running_loss = 0.0
    random.shuffle(train)

    # Start data loop. We don't do batches, training on 1 sentence at a time
    for labeled_sentence in train:

        # Zero the gradients
        optimizer.zero_grad()

        # Preprocess and make prediction
        idx_input, idx_labels, tokens, labels = preprocess(
            labeled_sentence, token_to_idx, tag_to_idx, tokenizer
        )
        output = model(idx_input)

        # Compute loss and backprop
        loss = criterion(output.squeeze(dim=0), idx_labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'\nEpoch {epoch}: Loss {running_loss / len(train)}\n')
    losses.append(running_loss / len(train))


# Create the confusion matrix
cm = np.zeros((len(tag_to_idx), len(tag_to_idx)), dtype='int')

# Evaluate the model on the test data
model.train(False)
for labeled_sentence in test:

    # Preprocess and make prediction
    idx_input, idx_labels, _, _ = preprocess(
        labeled_sentence, token_to_idx, tag_to_idx, tokenizer
    )
    output = model(idx_input)

    # Get prediction indices, then add 1 to entries in the confusion matrix
    # where rows represent the true labels and columns the predicted labels
    output_labels = output.squeeze(dim=0).argmax(dim=1).numpy()
    cm[idx_labels, output_labels] += 1


# Determine the per-tag precision and recall values
prdict = dict()
for tag, idx in tag_to_idx.items():
    row_sum = cm[idx, :].sum()
    col_sum = cm[:, idx].sum()

    # Tag was never encountered, no need for metrics
    if row_sum == 0:
        continue
    recall = round(cm[idx, idx] / row_sum, 2)

    # Tag was never predicted, precision is ill defined and set to 0.0
    if col_sum == 0:
        precision = 0.0
    else:
        precision = round(cm[idx, idx] / col_sum, 2)

    # Save results as (precision, recall, num_occurances)
    prdict[tag] = (precision, recall, int(row_sum))


# Show all metrics
print(f'Global Accuracy: {cm.trace() / cm.sum()}')
pprint.pprint(prdict)

plt.matshow(cm, cmap='binary')
plt.title(f'Confusion matrix for {len(tag_to_idx)} tags')
plt.show()

plt.plot(range(max_epochs), losses)
plt.title('Normalized Cross Entropy loss as a function of training epoch')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
