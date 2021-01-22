# SemanticTagger
An approach to semantic tagging using a GloVe word embedding as a basis for a BiLSTM.

## Prepare the data files
- Download a pre-trained GloVe embedding from <https://nlp.stanford.edu/projects/glove/>. We used the Common Crawl embedding with 42B tokens and 300 dimensions per word (glove.42B.300d.zip).
- Extract this file into the working directory.
- Download the Parallel Meaning Bank dataset from <https://pmb.let.rug.nl/data.php>. We used version 3.0.0, released 12-02-2020.
- Extract this folder into the working directory.

## Install dependencies
In a new virtual environment running python>=3.6.0, do:
- `pip install -r requirements.txt`

## Train and evaluate the model
Then run the script:
- `python main.py`

## View results
You will find the per-tag precision and recall metrics, as well as the total number of occurances, in a file named `results.txt`. The first line of that file will also tell you the global accuracy. There will also be a saved image of the confusion matrix and the normalized cross-entropy loss across epochs in the files `confusion_matrix.png` and `loss_per_epoch.png`, respectively.

Training the model on the gold dataset with a 90/10 train/test split (and close to no model fitting) yields a global accuracy of 78.9% across 81 semantic tags (results and plots for this run are uploaded). The confusion matrix is shown below:

![alt text](https://github.com/dashdeckers/SemanticTagger/blob/main/confusion_matrix.png?raw=true)


## Play around with the code
The code is simple and well documented. Feel free to use, modify, or copy any and all portions of the code, just mention where you found it :)
