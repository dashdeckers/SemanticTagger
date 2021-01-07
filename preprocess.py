import torch
import stanza


def preprocess(sentence, token_to_idx, download=False):
    """Preprocess a list of tokens into suitable input for the model.

    This means possibly further tokenizing using the Stanford Tokenizer, then
    lowercasing them and converting them into a tensor of token indices.
    We use the Stanford Tokenizer because that is also what they used when
    making the GloVe embedding so we can maximize tokens found in embedding.

    Args:
        sentence (List[str]): The sentence to preprocess.
        token_to_idx (dict): The token-index mapping.
        download (bool): Whether to first download the necessary files.

    Returns:
        torch.LongTensor: The preprocessed sentence as an index tensor.

    Raises:
        KeyError: If the word is not in the vocabulary.
    """
    if download:
        stanza.download('en')

    tokenizer = stanza.Pipeline('en', processors='tokenize')
    stanford_analysis = tokenizer(' '.join(sentence))
    tokens = [token.text.lower() for token in stanford_analysis.iter_tokens()]

    try:
        token_indices = [token_to_idx[token] for token in tokens]
    except KeyError:
        problematic_tokens = [t for t in tokens if t not in token_to_idx]
        print('Token not in vocabulary:', problematic_tokens)
        raise KeyError

    return torch.LongTensor(token_indices)
