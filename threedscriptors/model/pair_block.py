from threedscriptors.configuration.architecture_config import EncoderConfig
from torch import nn, Tensor
import torch
import torch.nn.functional as F
from threedscriptors.model.pair_biased_attention import PairBiasedSelfAttention, PairOuterProdUpdate

class FeedForward(nn.Module):
    """GeGLU-style FFN (a bit better than ReLU/GELU + Linear)."""
    def __init__(self, d_model: int, d_hidden: int | None = None, p_drop=0.1):
        super().__init__()
        d_hidden = d_hidden or 4 * d_model          # usual width factor
        self.proj_in  = nn.Linear(d_model, d_hidden * 2)  # 2× for GEGLU
        self.proj_out = nn.Linear(d_hidden, d_model)
        self.dropout  = nn.Dropout(p_drop)

    def forward(self, x: Tensor) -> Tensor:
        x_in = self.proj_in(x)                      # (…, 2 d_hidden)
        x_g, x_h = x_in.chunk(2, dim=-1)
        x = F.gelu(x_g) * x_h                       # GEGLU
        return self.proj_out(self.dropout(x))


class PairFFN(nn.Module):
    def __init__(self, d_pair: int, dropout=0.1):
        super().__init__()
        self.lin = nn.Linear(d_pair, d_pair)
        self.gate = nn.Linear(d_pair, d_pair, bias=False)
        nn.init.zeros_(self.gate.weight)          # zero-init → no effect at start
        self.dropout = nn.Dropout(dropout)

    def forward(self, P: Tensor) -> Tensor:
        delta = self.lin(F.silu(P))
        return P + self.dropout(torch.sigmoid(self.gate(P)) * delta)



class PairBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, d_pair: int,
                 dim_feedforward, dropout=0.1, pair_ffn: bool = True):
        super().__init__()

        # ① attention sub-layer
        self.ln_s1 = nn.LayerNorm(embedding_dim)
        self.attn  = PairBiasedSelfAttention(embedding_dim, num_heads, d_pair, dropout)

        # ② FFN sub-layer for atoms
        self.ln_s2 = nn.LayerNorm(embedding_dim)
        self.ffn_s = FeedForward(embedding_dim, dim_feedforward, dropout)

        # pair update as before
        self.pair_up = PairOuterProdUpdate(embedding_dim, d_pair)

        # optional mini-FFN over the pair tensor
        self.pair_ffn = PairFFN(d_pair) if pair_ffn else nn.Identity()

    
    
    def forward(self, S, mask, P, mask_pair):
        # --- atom stream ------------------------------------------------------
        S = S + self.attn(self.ln_s1(S), P, mask)   # residual 1
        S = S + self.ffn_s(self.ln_s2(S))           # residual 2 (FFN)

        # --- pair stream ------------------------------------------------------
        P = self.pair_up(S, P, mask_pair)
        P = self.pair_ffn(P)                       

        return S, P
    


class TransformerPairEncoder(nn.Module):
    def __init__(self, encoder_config: EncoderConfig):
        super().__init__()

        self.encoder_config = encoder_config

        self.layers = nn.ModuleList(
            [
                PairBlock(**encoder_config.attention_layer_config.model_dump(), d_pair = self.encoder_config.d_pair)
                for _ in range(encoder_config.N_layers)
            ]
        )

    def forward(self, x, padding_mask, P, pair_masks):
        for layer in self.layers:
            x, P = layer(x, padding_mask, P, pair_masks)

        return x, P