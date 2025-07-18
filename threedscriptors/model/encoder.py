import torch.nn as nn
from torch.nn import MultiheadAttention

from threedscriptors.configuration.architecture_config import EncoderConfig
from threedscriptors.model.global_aggregator import GlobalAggregator
from threedscriptors.data_handling.sample import PreprocessedSample


class EncoderBlock(nn.Module):
    def __init__(
        self, embedding_dim, num_heads, dim_feedforward, dropout=0.0
    ):  # TODO: Recommend to write type hints for all arguments
        """
        Inputs:
            embedding_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        # Attention layer
        self.q = nn.Linear(embedding_dim, embedding_dim)
        self.k = nn.Linear(embedding_dim, embedding_dim)
        self.v = nn.Linear(embedding_dim, embedding_dim)

        self.self_attn = MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(embedding_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=False),
            nn.Linear(dim_feedforward, embedding_dim),
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        # Attention part
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        attn_out, _ = self.self_attn(q, k, v, key_padding_mask=padding_mask)
        x = x + self.dropout(attn_out)
        y = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(y)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self, encoder_config: EncoderConfig, global_aggregator: GlobalAggregator
    ):
        super().__init__()

        self.encoder_config = encoder_config

        self.layers = nn.ModuleList(
            [
                EncoderBlock(**encoder_config.attention_layer_config.model_dump())
                for _ in range(encoder_config.N_layers)
            ]
        )

        self.aggregator = global_aggregator

    def forward(self, preprocessed_sample: PreprocessedSample):
        S = preprocessed_sample.preprocessed_atomic_embeddings

        for layer in self.layers:
            S = layer(S, padding_mask=preprocessed_sample.padding_mask)

        molecular_descriptor = self.aggregator(S, preprocessed_sample.padding_mask)

        return molecular_descriptor
