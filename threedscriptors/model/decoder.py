import torch.nn as nn

from threedscriptors.configuration.architecture_config import DecoderConfig
from threedscriptors.data_handling.sample import PreprocessedSample
from threedscriptors.model.multihead_self_attention import (
    MultiHeadCrossAttention,
    MultiHeadSelfAttention,
)
from threedscriptors.model.pair_biased_attention import (
    PairBiasedSelfAttention,
    PairOuterProdUpdate,
)
from threedscriptors.model.pair_block import FeedForward, PairFFN


class DecoderPairBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dim_feedforward,
        d_pair: int,
        d_descriptor,
        d_geo,
        dropout=0.1,
    ):
        super().__init__()

        # 1 attention sub-layer
        self.layer_norm_0 = nn.LayerNorm(embedding_dim)
        self.attn = PairBiasedSelfAttention(embedding_dim, num_heads, d_pair, dropout)

        # Cross Attention layer to the molecular descriptor
        self.cross_attention = MultiHeadCrossAttention(
            embedding_dim, d_descriptor, num_heads, dropout
        )
        self.layer_norm_1 = nn.LayerNorm(embedding_dim)

        # Two-layer MLP
        self.ffn = FeedForward(embedding_dim, d_hidden=dim_feedforward, dropout=dropout)
        self.layer_norm_2 = nn.LayerNorm(embedding_dim)

        # Pair Representation update
        self.pair_update = PairOuterProdUpdate(embedding_dim, d_pair, d_geo)
        self.pair_ffn = PairFFN(d_pair, dropout)

        # Layers to apply in between the main layers
        self.dropout = nn.Dropout(dropout)

    def forward(self, S, P, M, p_geo, padding_mask, mask_pair):

        # Self Attention based update
        S = S + self.attn(self.layer_norm_0(S), P, padding_mask)

        S = S + self.cross_attention(self.layer_norm_1(S), M, padding_mask)

        S = S + self.ffn(self.layer_norm_2(S))

        P = P + self.pair_update(S, P, p_geo, mask_pair)  # Outer product update
        P = P + self.pair_ffn(P)

        return S, P


class TransformerPairDecoder(nn.Module):
    def __init__(self, decoder_config: DecoderConfig):

        super().__init__()

        self.config = decoder_config
        self.layers = nn.ModuleList(
            [
                DecoderPairBlock(**decoder_config.attention_layer_config.model_dump(), d_descriptor=self.config.d_descriptor,d_pair=decoder_config.d_pair, d_geo=decoder_config.d_geo)
                for _ in range(decoder_config.N_layers)
            ]
        )

    def forward(self, preprocessed_sample: PreprocessedSample, molecular_descriptor):


        S = preprocessed_sample.preprocessed_atomic_embeddings
        P = preprocessed_sample.initial_pair_representation

        for layer in self.layers:
            S, P = layer(
                S,
                P,
                molecular_descriptor,
                preprocessed_sample.geometrical_encoding,
                preprocessed_sample.padding_mask,
                preprocessed_sample.pair_mask,
            )

        return S


class DecoderBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        d_descriptor,
        num_heads: int,
        dim_feedforward,
        dropout=0.3,
    ):
        super().__init__()

        # 1 attention sub-layer
        self.layer_norm_0 = nn.LayerNorm(embedding_dim)
        self.attn = MultiHeadSelfAttention(embedding_dim, num_heads, dropout)

        # Cross Attention layer to the molecular descriptor
        self.cross_attention = MultiHeadCrossAttention(
            embedding_dim, d_descriptor, num_heads, dropout
        )
        self.layer_norm_1 = nn.LayerNorm(embedding_dim)

        # Two-layer MLP
        self.ffn = FeedForward(embedding_dim, d_hidden=dim_feedforward, dropout=dropout)
        self.layer_norm_2 = nn.LayerNorm(embedding_dim)

        # Layers to apply in between the main layers
        self.dropout = nn.Dropout(dropout)

    def forward(self, S, M, padding_mask):

        # Self Attention based update
        S = S + self.dropout(self.attn(self.layer_norm_0(S), padding_mask))

        S = S + self.dropout(
            self.cross_attention(self.layer_norm_1(S), M, padding_mask)
        )


        S = S + self.dropout(self.ffn(self.layer_norm_2(S)))


        return S


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_config: DecoderConfig):

        super().__init__()

        self.config = decoder_config
        self.layers = nn.ModuleList(
            [
                DecoderBlock(**decoder_config.attention_layer_config.model_dump(), d_descriptor=self.config.d_descriptor)
                for _ in range(decoder_config.N_layers)
            ]
        )

    def forward(self, prepocessed_sample : PreprocessedSample, molecular_descriptor):

        S = prepocessed_sample.preprocessed_atomic_embeddings


        for layer in self.layers:
            S = layer(S, molecular_descriptor, prepocessed_sample.padding_mask)
        return S



