import torch

# 10 epochs, old data:
# single-directional: 0.7383
# bi-directional: 0.7449
# bi-directional + 2 layers: 0.7834

# 10 epochs, new data:
# bi-directional + 2 layers: 0.8540


class SemTag(torch.nn.Module):
    def __init__(self, embedding_layer, n_tags):
        super(SemTag, self).__init__()

        self.embedding = embedding_layer
        self.lstm = torch.nn.LSTM(
            input_size=300,         # 300 dimensional GloVe embedding
            hidden_size=n_tags,     # output one-hot encoded vectors
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, input):
        vectors = self.embedding(input).unsqueeze(dim=0)
        output, _ = self.lstm(vectors)
        return self.softmax(output)
