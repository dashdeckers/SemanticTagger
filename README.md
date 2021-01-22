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
