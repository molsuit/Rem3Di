import torch
from e3nn.o3 import Irreps


def get_mace_model_irrep_signature(mace_model: torch.nn.Module) -> Irreps:
    signature = None

    for products in mace_model.products:  # type: ignore
        if signature is None:
            signature = Irreps(str(products.linear.__dict__["irreps_out"]))
        else:
            signature = signature + Irreps(str(products.linear.__dict__["irreps_out"]))

    return signature


def remove_equivariants(atomic_embeddings, invariant_indices):
    ind = torch.tensor(list(invariant_indices)).to(atomic_embeddings.device)
    return atomic_embeddings[:, :, ind]


def split_invariants_equivariants(
    emb: torch.Tensor, invariant_indices
) -> tuple[torch.Tensor, torch.Tensor]:
    # emb: [B, N, C]
    C = emb.shape[2]  # ensure tensor of long indices on the correct device
    if not torch.is_tensor(invariant_indices):
        invariant_indices = torch.tensor(
            list(invariant_indices), dtype=torch.long, device=emb.device
        )
    # build mask
    mask = torch.zeros(C, dtype=torch.bool, device=emb.device)
    mask[invariant_indices] = True

    invariants = emb[:, :, mask]  # picks out the True positions
    equivariants = emb[:, :, ~mask]  # picks out the False positions
    return invariants, equivariants


def get_invariant_indices(irreps: Irreps) -> Irreps:
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


def get_equivariant_irreps(irreps: Irreps):
    irreps = Irreps(irreps)  # normalise input
    filtered = [
        (mul, ir)
        for mul, ir in irreps  # keep l>0
        if ir.l > 0
    ]
    return Irreps(filtered)


def get_pseudoscalar_indices(irreps: Irreps):
    all_slices = irreps.slices()

    # Zip together blocks and their slices so we can filter in one pass
    pseudoscalar_slices = [
        sl
        for (mul, ir), sl in zip(irreps, all_slices, strict=False)
        if ir.l == 0 and ir.p == -1  # l == 0  ➜ scalar,  p == -1 ➜ odd
    ]

    return pseudoscalar_slices


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
