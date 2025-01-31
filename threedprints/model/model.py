import torch
import torch.nn as nn
from torch.nn import MultiheadAttention


class EncoderBlock(nn.Module):
    def __init__(
        self, input_dim, embedding_dim, num_heads, dim_feedforward, dropout=0.0
    ):  # TODO: Recommend to write type hints for all arguments
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        # Attention layer
        self.q = nn.Linear(input_dim, embedding_dim)
        self.k = nn.Linear(input_dim, embedding_dim)
        self.v = nn.Linear(input_dim, embedding_dim)

        self.self_attn = MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=False),
            nn.Linear(dim_feedforward, input_dim),
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
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
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderBlock(**block_args) for _ in range(num_layers)]
        )

    def forward(self, x, padding_mask=None):
        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask)

        # Calculate the global descriptor by averaging the sequence
        des = torch.mean(x, dim=1)

        return des

    def get_attention_maps(self, x, padding_mask=None):
        attention_maps = []
        for layer in self.layers:
            _, attn_map = layer.self_attn(
                x, padding_mask=padding_mask, return_attention=True
            )
            attention_maps.append(attn_map)
            x = layer(x)
        return attention_maps


class DecoderBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        embedding_dim,
        num_heads,
        dim_feedforward,
        descriptor_dim=256,
        dropout=0.0,
    ):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer

        self.q = nn.Linear(input_dim, embedding_dim)
        self.k = nn.Linear(input_dim, embedding_dim)
        self.v = nn.Linear(input_dim, embedding_dim)

        self.self_attn = MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.Linear(descriptor_dim, input_dim)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=False),
            nn.Linear(dim_feedforward, input_dim),
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.norm3 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, global_descriptor, padding_mask=None):
        # Attention part
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        attn_out, _ = self.self_attn(q, k, v, key_padding_mask=padding_mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        cross_attn_out = self.cross_attn(global_descriptor)
        cross_attn_out = cross_attn_out.unsqueeze(1)
        x = x + self.dropout(cross_attn_out)

        x = self.norm2(x)
        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm3(x)

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderBlock(**block_args) for _ in range(num_layers)]
        )
        self.reconstruction_embedding = nn.Parameter(
            torch.randn(size=(1, block_args["input_dim"]))
        )

    def forward(
        self, x, global_descriptor, reconstruction_mask=None, padding_mask=None
    ):
        x = torch.where(
            reconstruction_mask.unsqueeze(-1).bool(), self.reconstruction_embedding, x
        )

        for layer in self.layers:
            x = layer(x, global_descriptor=global_descriptor, padding_mask=padding_mask)

        return x

    def get_attention_maps(self, x, padding_mask=None):
        attention_maps = []
        for layer in self.layers:
            _, attn_map = layer.self_attn(x, mask=padding_mask, return_attention=True)
            attention_maps.append(attn_map)
            x = layer(x)
        return attention_maps


class Transformer(nn.Module):
    def __init__(self, encoder: TransformerEncoder, decoder: TransformerDecoder):
        super().__init__()
        self.encoder: TransformerEncoder = encoder
        self.decoder: TransformerDecoder = decoder

    def forward(self, x, padding_mask=None, reconstruction_mask=None):
        global_descriptor = self.encoder(x, padding_mask=padding_mask)
        x = self.decoder(
            x,
            global_descriptor,
            reconstruction_mask=reconstruction_mask,
            padding_mask=padding_mask,
        )

        return x
