import torch
from e3nn.o3 import Irreps
from mace.calculators import MACECalculator


def get_mace_calculator_irrep_signature(mace_calculator: MACECalculator) -> Irreps:
    signature = None

    for products in mace_calculator.models[0].products:
        if signature is None:
            signature = Irreps(str(products.linear.__dict__["irreps_out"]))
        else:
            signature = signature + Irreps(str(products.linear.__dict__["irreps_out"]))

    return signature


def get_mace_calculator_embedding_dimension(mace_calculator: MACECalculator) -> int:
    irrep_signature = get_mace_calculator_irrep_signature(mace_calculator)
    embdeding_dimension = Irreps(irrep_signature).dim
    return embdeding_dimension


def remove_equivariants(atomic_embeddings, invariant_indices):
    ind = torch.tensor(list(invariant_indices)).to(atomic_embeddings.device)
    return atomic_embeddings[:, :, ind]


def get_invariant_indices(irreps: Irreps):
    """
    Gets the slices for all irreps indices, and only returns those with degree 0
    """
    total_dim = irreps.dim
    slices = irreps.slices()
    invariant_slices = []
    out_irrep = []
    for idx, irrep_slice in enumerate(irreps):
        if irrep_slice.ir[0] == 0:
            out_irrep.append(irrep_slice)
            invariant_slices.append(slices[idx])

    index_list = slices_to_index_list(
        slices=invariant_slices, sequence_length=total_dim
    )
    return index_list, Irreps(out_irrep)


def slices_to_index_list(slices, sequence_length):
    # Using a list comprehension to flatten the list of indices
    return [i for s in slices for i in range(*s.indices(sequence_length))]


def get_pseudoscalars_indices(irreps: Irreps):
    """
    Gets the slices for all irreps indices, and only returns those with degree 0 and odd parity
    """
    total_dim = irreps.dim
    slices = irreps.slices()
    invariant_slices = []
    out_irrep = []

    for idx, irrep_slice in enumerate(irreps):
        # print(irrep_slice.ir)
        if irrep_slice.ir[0] == 0 and irrep_slice.ir[-1] == -1:
            out_irrep.append(irrep_slice)
            invariant_slices.append(slices[idx])

    index_list = slices_to_index_list(
        slices=invariant_slices, sequence_length=total_dim
    )
    return index_list, Irreps(out_irrep)
