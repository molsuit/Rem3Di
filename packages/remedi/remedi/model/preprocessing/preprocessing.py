import torch
from ase.data import atomic_masses
from e3nn.o3 import Irreps
from mace.calculators.mace_torchsim import MaceTorchSimModel
from torch import nn
from torch_sim.state import SimState
from torch_sim.typing import SystemExtras

from remedi.model.preprocessing.atomic_descriptor_preprocessor import (
    AtomicDescriptorPreprocessor,
)
from remedi.model.preprocessing.geometric_preprocessor import (
    PairDistanceMatrixGeometricPreprocessor,
)
from remedi.training.data.sample import PreprocessedSample, Sample
from remedi.training.data.samplers import quantize_pad_length

_ATOMIC_MASS_TABLE = torch.as_tensor(atomic_masses, dtype=torch.float32)


def ir_mul_to_mul_ir(features: torch.Tensor, irreps: Irreps) -> torch.Tensor:
    """Reorder flat irreps features from cuequivariance's ``ir_mul`` layout to
    e3nn's ``mul_ir`` layout.

    Within each ``(multiplicity, irrep)`` block, ``ir_mul`` stores the
    ``irrep.dim`` components outermost; ``mul_ir`` (what e3nn and the RMS norm
    downstream expect) stores the channels outermost. Scalar blocks are
    unchanged.
    """
    if features.shape[-1] != irreps.dim:
        raise ValueError(
            f"features have {features.shape[-1]} channels, irreps {irreps} "
            f"describe {irreps.dim}."
        )
    leading_shape = features.shape[:-1]
    blocks = []
    for (multiplicity, irrep), block_slice in zip(irreps, irreps.slices(), strict=True):
        block = features[..., block_slice]
        if irrep.dim > 1:
            block = (
                block.reshape(*leading_shape, irrep.dim, multiplicity)
                .transpose(-1, -2)
                .reshape(*leading_shape, multiplicity * irrep.dim)
            )
        blocks.append(block)
    return torch.cat(blocks, dim=-1)


def _uses_ir_mul_layout(model: MaceTorchSimModel) -> bool:
    """Whether the wrapper converted MACE to cuequivariance.

    ``MaceTorchSimModel`` converts with ``layout="ir_mul"`` (also on the hybrid
    cueq + oeq fallback), so the node features of such a model come out in
    ``ir_mul`` order.
    """
    return bool(getattr(model, "_enable_cueq", False))


def _capture_node_feats(
    model: MaceTorchSimModel,
    state: SimState,
    node_feature_irreps: Irreps | None = None,
) -> torch.Tensor:
    """Run the MaceTorchSimModel and return the underlying model's node_feats
    in e3nn ``mul_ir`` layout.

    The wrapper only forwards node_feats in its result dict for PolarMACE
    checkpoints; for stock MACE we register a forward hook on
    ``model.model`` to grab them out of the raw output dict. The hook is
    one-shot — it is removed before this function returns.

    ``node_feature_irreps`` (the concatenated per-layer product irreps) is
    required for a cuequivariance-converted model, whose features arrive in
    ``ir_mul`` layout and are transposed back here.
    """
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        if isinstance(output, dict) and "node_feats" in output:
            captured["node_feats"] = output["node_feats"]

    handle = model.model.register_forward_hook(hook)
    try:
        result = model(state)
    finally:
        handle.remove()

    if isinstance(result, dict) and "node_feats" in result:
        node_feats = result["node_feats"]
    elif "node_feats" in captured:
        node_feats = captured["node_feats"]
    else:
        raise RuntimeError("MACE model did not return 'node_feats'.")

    if not _uses_ir_mul_layout(model):
        return node_feats
    if node_feature_irreps is None:
        raise ValueError(
            "The MACE model runs on cuequivariance (ir_mul layout); pass "
            "node_feature_irreps to transpose its node features to mul_ir."
        )
    return ir_mul_to_mul_ir(node_feats, node_feature_irreps)


class Preprocessor(nn.Module):
    def __init__(
        self,
        atomic_preprocessor,
        geometric_preprocessor: PairDistanceMatrixGeometricPreprocessor,
    ):
        super().__init__()

        self.atomic_preprocessor: AtomicDescriptorPreprocessor = atomic_preprocessor
        self.geometric_preprocessor = geometric_preprocessor

    def forward(self, sample: Sample) -> PreprocessedSample:
        preprocessed_sample: PreprocessedSample = self.atomic_preprocessor(
            sample.embeddings, sample.padding_mask
        )

        preprocessed_sample.padding_mask = sample.padding_mask

        (
            preprocessed_sample.initial_pair_representation,
            preprocessed_sample.geometrical_encoding,
            preprocessed_sample.pair_mask,
        ) = self.geometric_preprocessor(sample)

        return preprocessed_sample


def centre_per_system(
    positions: torch.Tensor, system_index: torch.Tensor, atom_counts: torch.Tensor
) -> torch.Tensor:
    """Subtract each system's mean position from its atoms' positions."""
    sums = positions.new_zeros((atom_counts.shape[0], positions.shape[-1]))
    sums.index_add_(0, system_index, positions)
    means = sums / atom_counts.to(positions.dtype).unsqueeze(1).clamp_min(1.0)
    return positions - means.index_select(0, system_index)


def _sample_to_simstate(sample: Sample, r_max: float) -> tuple[SimState, torch.Tensor]:
    """Build a torch-sim SimState directly from flat Sample tensors.

    Mirrors the cell/position prep that the md_clustering pipeline does in ASE
    (`_prepare_atoms_for_md`): per-system centering, extent sized for torch-sim's
    fixed neighbour list (padding = max(20 Å, 5·r_max)), and shift into
    [0, extent). Vectorised across the batch so we never hit a Python for-loop
    over systems.

    Returns the SimState together with the per-system atom counts tensor so the
    caller can redistribute outputs into padded per-system tensors.
    """
    if sample.atomic_positions is None or sample.atomic_numbers is None:
        raise ValueError(
            "Sample must provide atomic positions and numbers for torchsim state initialization."
        )
    if sample.system_index is None:
        raise ValueError("Sample must provide system_index for batched inference.")

    positions = sample.atomic_positions
    atomic_numbers = sample.atomic_numbers.to(device=positions.device, dtype=torch.long)
    system_idx = sample.system_index.to(device=positions.device, dtype=torch.long)

    device = positions.device
    dtype = positions.dtype

    if system_idx.numel() == 0:
        raise ValueError("Cannot build SimState from an empty batch.")
    if system_idx.min() < 0:
        raise ValueError("system_index must be non-negative.")
    if (system_idx[1:] < system_idx[:-1]).any():
        raise ValueError(
            "system_index must be sorted: each system's atoms must be contiguous "
            "(as yield_molecules_collate_fn emits them)."
        )
    n_systems = int(system_idx.max().item()) + 1

    lengths = torch.bincount(system_idx, minlength=n_systems)

    centered = centre_per_system(positions, system_idx, lengths)

    per_atom_norm = centered.norm(dim=-1)
    radius = torch.zeros(n_systems, device=device, dtype=dtype).scatter_reduce(
        0, system_idx, per_atom_norm, reduce="amax", include_self=False
    )

    padding = max(20.0, 5.0 * float(r_max))
    extent = 2.0 * (radius + padding)  # (n_systems,)

    shifted_positions = centered + (extent / 2.0).index_select(0, system_idx).unsqueeze(
        -1
    )

    eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
    cells = eye * extent.view(n_systems, 1, 1)

    mass_lookup = _ATOMIC_MASS_TABLE.to(device=device, dtype=dtype)
    masses = mass_lookup.index_select(0, atomic_numbers)

    # mace.calculators.mace_torchsim.MaceTorchSimModel reads
    # total_charge/total_spin off the SimState (extras or attributes) and
    # builds the PolarMACE-specific fermi_level / external_field /
    # rcell / volume / density_coefficients entries itself. The MACE key is
    # named "total_spin" but actually consumes spin multiplicity (2S+1) — our
    # internal field is named `multiplicity` to reflect that; we still write
    # it under SystemExtras.TOTAL_SPIN because that is the key MACE expects.
    system_extras: dict[str, torch.Tensor] = {}
    if sample.total_charge is not None:
        system_extras[SystemExtras.TOTAL_CHARGE] = sample.total_charge.to(
            device=device, dtype=dtype
        ).reshape(n_systems)
    if sample.multiplicity is not None:
        system_extras[SystemExtras.TOTAL_SPIN] = sample.multiplicity.to(
            device=device, dtype=dtype
        ).reshape(n_systems)

    state = SimState(
        positions=shifted_positions,
        masses=masses,
        cell=cells,
        pbc=False,
        atomic_numbers=atomic_numbers.to(torch.int),
        system_idx=system_idx,
        _system_extras=system_extras,
    )
    return state, lengths


class PreprocessorWithAtomicEmbedding(Preprocessor):
    def __init__(
        self,
        mace_model: MaceTorchSimModel,
        atomic_preprocessor,
        geometric_preprocessor: PairDistanceMatrixGeometricPreprocessor,
        node_feature_irreps: Irreps | None = None,
        pad_multiple: int = 1,
    ):
        super().__init__(atomic_preprocessor, geometric_preprocessor)
        self.torch_sim_mace_model = mace_model
        # Irreps of MACE's node features (all product layers concatenated);
        # needed to undo the cuequivariance ir_mul layout.
        self.node_feature_irreps = node_feature_irreps
        # Padded atom count is rounded up to a multiple of `pad_multiple` so
        # torch.compile sees only a small set of distinct shapes. 1 = off.
        if pad_multiple <= 0:
            raise ValueError("pad_multiple must be a positive integer.")
        self.pad_multiple = int(pad_multiple)

    def _run_mace(self, state: SimState) -> torch.Tensor:
        """Run the wrapped MACE model on a SimState and return node_feats.

        The wrapper handles neighbour-list construction, total_charge / spin
        propagation, and (for PolarMACE) auto-filling fermi_level /
        external_field / rcell / volume / density_coefficients. Its public
        forward only forwards node_feats for PolarMACE, so we capture it
        through a one-shot forward hook on the underlying model — that path
        works uniformly for both PolarMACE and stock MACE.
        """
        return _capture_node_feats(
            self.torch_sim_mace_model, state, self.node_feature_irreps
        )

    def _atomic_embeddings(
        self, sample: Sample
    ) -> tuple[torch.Tensor, SimState, torch.Tensor]:
        """MACE node features of the flat batch ``(atoms, C)``, with the
        SimState they were computed on and the per-system atom counts."""
        state, lengths = _sample_to_simstate(
            sample, r_max=float(self.torch_sim_mace_model.r_max)
        )
        with torch.inference_mode():
            atomic_embeddings = self._run_mace(state).detach()
        return atomic_embeddings, state, lengths

    def raw_invariants(self, sample: Sample) -> torch.Tensor:
        """The unnormalised invariants ``(atoms, D)`` of a flat batch: what the
        invariant normalisation statistics are estimated from. Leaves
        ``sample`` untouched."""
        atomic_embeddings, _, _ = self._atomic_embeddings(sample)
        # A plain copy: inference tensors cannot enter later autograd-tracked math.
        return self.atomic_preprocessor.select_invariants(
            atomic_embeddings.clone()[None]
        )[0]

    def forward(self, sample: Sample) -> PreprocessedSample:
        atomic_embeddings, state, lengths = self._atomic_embeddings(sample)

        n_systems = lengths.shape[0]
        max_atoms = int(lengths.max().item()) if lengths.numel() > 0 else 0
        if max_atoms > 0 and self.pad_multiple > 1:
            max_atoms = quantize_pad_length(max_atoms, self.pad_multiple)

        device = state.positions.device
        system_index = state.system_idx

        # The pair-distance preprocessor gets each system's centred positions
        # (the frame MACE saw, before the shift into the simulation cell).
        centred_positions = centre_per_system(
            sample.atomic_positions.to(device), system_index, lengths
        )

        padded_positions = pad_per_system(
            centred_positions, system_index, lengths, max_atoms
        )
        padded_embeddings = pad_per_system(
            atomic_embeddings, system_index, lengths, max_atoms
        )

        padding_mask = torch.arange(max_atoms, device=device).unsqueeze(0).expand(
            n_systems, max_atoms
        ) >= lengths.unsqueeze(1)

        sample.embeddings = padded_embeddings
        sample.atomic_positions = padded_positions
        sample.padding_mask = padding_mask

        return super().forward(sample)


def pad_per_system(
    values: torch.Tensor,
    system_index: torch.Tensor,
    atom_counts: torch.Tensor,
    max_atoms: int,
) -> torch.Tensor:
    """Scatter flat per-atom ``values`` (N, F) into a zero-padded (B, max_atoms, F).

    ``system_index`` must be sorted (``_sample_to_simstate`` checks it), so an
    atom's slot within its system is its flat index minus the system's offset.
    """
    offsets = torch.cumsum(atom_counts, dim=0) - atom_counts
    local_index = (
        torch.arange(values.shape[0], device=values.device) - offsets[system_index]
    )
    padded = values.new_zeros((atom_counts.shape[0], max_atoms, values.shape[-1]))
    padded[system_index, local_index] = values
    return padded
