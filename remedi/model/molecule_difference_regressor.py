from torch import nn

from remedi.model.molecular_descriptor import MolecularDescriptor


class MolecularDifferenceRegressor(nn.Module):
    def __init__(self, descriptor_input_dim, aux_input_dim, aux_embedding_dim):
        super().__init__()
        self.descriptor_input_dim = descriptor_input_dim

        self.diff_layer = nn.Sequential(
            nn.Linear(descriptor_input_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
        )

        self.experimental_cond_gate = nn.Sequential(
            nn.Linear(aux_input_dim, aux_embedding_dim),
            nn.LayerNorm(aux_embedding_dim),
            nn.SiLU(),
            nn.Linear(aux_embedding_dim, aux_embedding_dim),
            nn.Sigmoid(),
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(aux_embedding_dim, aux_embedding_dim),
            nn.LayerNorm(aux_embedding_dim),
            nn.SiLU(),
            nn.Linear(aux_embedding_dim, 1),
        )

    def forward(self, descriptors: MolecularDescriptor, auxillary_data):
        flat = descriptors.flat
        B = flat.shape[0]
        if B % 2 != 0:
            raise ValueError(f"Batch size must be even — got {B}")
        N = B // 2

        descriptors_e1 = flat[:N, :]
        descriptors_e2 = flat[N:, :]

        diff_descriptor = descriptors_e1 - descriptors_e2

        embedded_difference = self.diff_layer(diff_descriptor)

        auxillary_data_per_pair = auxillary_data[:N, :]
        experimental_cond = self.experimental_cond_gate(auxillary_data_per_pair)

        gated_embedding = embedded_difference * experimental_cond

        out = self.output_mlp(gated_embedding)

        return out
