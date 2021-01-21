import torch


def preprocess(
            labeled_sentence,
            token_to_idx,
            tag_to_idx,
            tokenizer,
            DEF_IDX=0,         # 0 is chosen arbitrarily
            DEF_TAG=43,        # 43 is the NIL tag
        ):
    """Preprocess a list of tokens into suitable input for the model.

    This means splitting words joined by a tilde, removing trailing hyphens,
    then lowercasing and possibly further tokenizing them using the Stanford
    Tokenizer, and finally converting them into a tensor of token indices.
    We use the Stanford Tokenizer because that is also what they used when
    making the GloVe embedding, so we can maximize tokens found in embedding.

    Args:
        labeled_sentence (List[Tuple[str, str]]): The sentence to preprocess.
        token_to_idx (dict): The token-index mapping.
        tag_to_idx (dict): The tag-index mapping.
        DEF_IDX (int): The default token index to use if word is out-of-vocab.
        DEF_TAG (int): The default tag index to use if tag is unknown.

    Returns:
        Tuple[torch.LongTensor, torch.LongTensor, List[str], List[str]]:
            The preprocessed:
                - sentence as an index tensor,
                - labels as an index tensor,
                - sentence as a list of tokens as strings
                - labels as a list of tags as strings
    """
    sentence, labels = list(zip(*labeled_sentence))

    out_indices = list()  # Token indices for embedding layer
    out_tokens = list()   # Final tokenization of sentence
    out_labels = list()   # Final labels of tokenized sentence

    # Iterate over each word in sentence to preprocess and maybe tokenize
    for word, label in zip(sentence, labels):

        # Stanford Tokenizer ignores the tilde '~', so we strip it manually
        ppword = ' '.join(word.split('~'))

        # This might be a single item (not further tokenized, but lowercased)
        for token in tokenizer(ppword.lower()).iter_tokens():

            # Remove trailing hyphens (a common cause for unknown words)
            if token.text.endswith('-') and len(token.text) > 1:
                token.text = token.text[:-1]

            out_indices.append(token_to_idx.get(token.text, DEF_IDX))
            out_tokens.append(token.text)
            out_labels.append(label)

            # Print any words not in vocabulary
            if token.text not in token_to_idx:
                print(f'ERROR: {token.text} not in vocabulary!')

    # Label indices of final labels of tokenized sentence
    label_indices = list()
    for tag in out_labels:

        # Print any tags not in tag_to_idx
        if tag not in tag_to_idx:
            print(f'ERROR (TAG): {tag} not in tag_to_idx!')

        label_indices.append(tag_to_idx.get(tag, DEF_TAG))

    return (
        torch.as_tensor(out_indices, dtype=torch.int64),
        torch.as_tensor(label_indices, dtype=torch.int64),
        out_tokens,
        out_labels
    )
