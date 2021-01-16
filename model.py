import torch


class SemTag(torch.nn.Module):
    def __init__(self, embedding_layer, n_tags):
        super(SemTag, self).__init__()

        self.batch_size = 1
        self.input_dim = 300
        self.hidden_dim = n_tags  # output one-hot vectors
        self.n_layers = 1

        self.embedding = embedding_layer
        self.lstm = torch.nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.n_layers,
            batch_first=True,
            # bidirectional=True
        )
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, input):
        vectors = self.embedding(input).unsqueeze(dim=0)
        output, _ = self.lstm(vectors)
        return self.softmax(output)
