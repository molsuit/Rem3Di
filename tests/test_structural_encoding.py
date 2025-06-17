import numpy as np
from threedscriptors.model.structural_encoding import PairDistanceMatrixEncodingBlock
from ase import Atoms
import torch


def test_distance_loading(molecule: Atoms, positional_encoding_config):

    pos = torch.from_numpy(molecule.get_positions()).unsqueeze(0).float()

    atom_mask = torch.ones(size=(1, pos.shape[1])).bool()

    pe = PairDistanceMatrixEncodingBlock(positional_encoding_config)
    P, distances, _ = pe(pos, atom_mask)




def test_distance_encoding_with_padding(molecule: Atoms, positional_encoding_config):

    pos = molecule.get_positions()

    N_atoms = pos.shape[0]

    padding = 1
    N_atoms_padded = N_atoms + padding

    pos = np.pad(
        pos,
        ((0, padding), (0, 0)),
        mode="constant",
    )

    pos = torch.from_numpy(pos).float().unsqueeze(0)

    atom_mask = torch.ones(size=(1, N_atoms_padded))
    atom_mask[0,N_atoms:] = 0

    atom_mask = atom_mask.bool()

    assert atom_mask.sum() == N_atoms

    pe = PairDistanceMatrixEncodingBlock(positional_encoding_config)

    P, distances, mask_pair = pe(pos, atom_mask)
    N_pairs = N_atoms**2

    assert mask_pair.sum().cpu().numpy().item() == N_pairs
    assert (P.transpose(1,2) == P).all()

    print(P.shape)
