"""MACE node features of padded molecule batches from the local MACE layers alone.

A MACE model's node features are the outputs of its product layers. In PolarMACE the
charges, the self-consistent field and the electrostatics only read them, so the
features need neither those nor the readouts or the forces. The featurizer runs the
embedding, the edge harmonics and radial basis, and the interaction and product
layers, sharing the model's parameters, without gradients.

The graph of a padded batch ``(molecules, atoms)`` is its neighbour list: the ordered
atom pairs of a molecule within the model's cutoff, built on the CPU when the batch
is collated (``MoleculeCollator``), so the pair count is an input shape and the
computation has no data-dependent sizes and no host reads. ``torch.compile`` captures
it whole, as one graph for every ``(molecules, atoms, pairs)`` shape. The features
equal those of the model's own radius graph: pairs beyond the cutoff, and the inert
pairs the featurizer may add (placed beyond the cutoff), carry exactly nothing for
the supported models (``check_distant_pairs_are_inert``).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import torch
from e3nn.o3 import Irreps
from torch import nn

from remedi.model.featurizer.padding import PaddedMolecules
from remedi.utils.model_utils import concatenate_irreps

NodeFeatureLayout = Literal["mul_ir", "ir_mul"]

# The element lookup covers atomic numbers 0 (padding) to 118.
_LARGEST_ATOMIC_NUMBER = 118
# torch.compile specialises (and recompiles for) dimensions of size 0 and 1: a batch
# with fewer pairs (single atoms, or atoms beyond the cutoff of each other) gets inert
# pairs up to this count.
MINIMUM_PAIRS = 2
# The cutoff of a batch's neighbour list and the model's agree up to the rounding of
# the model's r_max to float32.
_CUTOFF_RELATIVE_TOLERANCE = 1e-6


def ir_mul_to_mul_ir_index(irreps: Irreps) -> torch.Tensor:
    """The gather index that reorders flat features from cuequivariance's ``ir_mul``
    layout to e3nn's ``mul_ir`` layout: ``mul_ir = ir_mul[..., index]``.

    Within each ``(multiplicity, irrep)`` block, ``ir_mul`` stores the ``irrep.dim``
    components outermost and ``mul_ir`` the channels; scalar blocks are unchanged.
    """
    index: list[int] = []
    for (multiplicity, irrep), block in zip(irreps, irreps.slices(), strict=True):
        for channel in range(multiplicity):
            for component in range(irrep.dim):
                index.append(block.start + component * multiplicity + channel)
    return torch.tensor(index, dtype=torch.long)


def _submodule(model: nn.Module, name: str) -> nn.Module:
    submodule = getattr(model, name, None)
    if not isinstance(submodule, nn.Module):
        raise ValueError(f"the MACE model has no module {name!r}")
    return submodule


def _layer_list(model: nn.Module, name: str) -> nn.ModuleList:
    layers = getattr(model, name, None)
    if not isinstance(layers, nn.ModuleList):
        raise ValueError(f"the MACE model has no layer list {name!r}")
    return layers


def _tensor(model: nn.Module, name: str) -> torch.Tensor:
    value = getattr(model, name, None)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"the MACE model has no tensor {name!r}")
    return value.detach()


class MaceFeaturizer(nn.Module):
    """The node features of a MACE model on padded molecule batches.

    ``model`` is a MACE model prepared for ``torch.compile`` (built without e3nn code
    generation; see ``build_mace_featurizer``); the featurizer keeps references to its
    local layers. ``layout`` is the layout the model emits its features in
    (``ir_mul`` after a cuequivariance conversion); the featurizer always returns
    ``mul_ir``. ``permute_edge_vectors``: whether the model feeds its edge vectors to
    the spherical harmonics as ``(y, z, x)`` (PolarMACE) instead of ``(x, y, z)``.
    It runs eagerly until :meth:`compile_features` is called.
    """

    element_index: torch.Tensor
    model_atomic_numbers: torch.Tensor
    mul_ir_index: torch.Tensor

    def __init__(
        self,
        model: nn.Module,
        *,
        per_layer_irreps: list[Irreps],
        layout: NodeFeatureLayout,
        permute_edge_vectors: bool,
    ):
        super().__init__()
        self.node_embedding = _submodule(model, "node_embedding")
        self.spherical_harmonics = _submodule(model, "spherical_harmonics")
        self.radial_embedding = _submodule(model, "radial_embedding")
        self.interactions = _layer_list(model, "interactions")
        self.products = _layer_list(model, "products")
        if len(per_layer_irreps) != len(self.products):
            raise ValueError(
                f"{len(per_layer_irreps)} layer irreps for {len(self.products)} layers"
            )
        self.per_layer_irreps = list(per_layer_irreps)
        self.permute_edge_vectors = permute_edge_vectors
        self.r_max = float(_tensor(model, "r_max"))
        # Inert pairs (an atom with itself, or a pair with a padding atom) are placed
        # this far apart: beyond the cutoff, so they carry nothing, at a finite length
        # (a zero vector would give NaN harmonics).
        self.inert_pair_distance = 2.0 * self.r_max

        atomic_numbers = _tensor(model, "atomic_numbers")
        model_atomic_numbers = atomic_numbers.to(torch.long).cpu()
        if int(model_atomic_numbers.max()) > _LARGEST_ATOMIC_NUMBER:
            raise ValueError("the model lists atomic numbers beyond 118")
        element_index = torch.full((_LARGEST_ATOMIC_NUMBER + 1,), -1, dtype=torch.long)
        element_index[model_atomic_numbers] = torch.arange(len(model_atomic_numbers))
        device = atomic_numbers.device
        self.number_of_elements = len(model_atomic_numbers)
        self.register_buffer(
            "element_index", element_index.to(device), persistent=False
        )
        # The radial embedding takes the model's atomic numbers (distance transforms).
        self.register_buffer(
            "model_atomic_numbers", atomic_numbers.clone(), persistent=False
        )

        self.layout = layout
        self.register_buffer(
            "mul_ir_index",
            ir_mul_to_mul_ir_index(concatenate_irreps(per_layer_irreps)).to(device),
            persistent=False,
        )
        # A plain attribute (a function, not a module): the parameters stay registered
        # once, under the featurizer.
        self._compiled_features: Callable[..., torch.Tensor] | None = None

    def compile_features(self) -> None:
        """Evaluate the features through ``torch.compile`` from now on (whole graph,
        dynamic shapes): one graph for every ``(molecules, atoms, pairs)`` shape,
        built on the first call."""
        if self._compiled_features is None:
            self._compiled_features = torch.compile(
                self.features, fullgraph=True, dynamic=True
            )

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the features: the parameters' (follows ``.to``)."""
        return next(self.parameters()).dtype

    def forward(self, molecules: PaddedMolecules) -> torch.Tensor:
        """Node features ``(molecules, atoms, channels)`` in ``mul_ir`` layout, zero on
        padding atoms, of a padded batch (``pad_molecules``) whose neighbour list was
        built with the model's cutoff."""
        atomic_numbers, positions = molecules.atomic_numbers, molecules.positions
        edge_index = molecules.edge_index
        if atomic_numbers.dim() != 2 or positions.shape != (*atomic_numbers.shape, 3):
            raise ValueError(
                f"expected atomic numbers (molecules, atoms) and positions "
                f"(molecules, atoms, 3), got {tuple(atomic_numbers.shape)} and "
                f"{tuple(positions.shape)}"
            )
        if edge_index.dim() != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                f"expected the neighbour list as [2, pairs], got "
                f"{tuple(edge_index.shape)}"
            )
        if not math.isclose(
            molecules.neighbor_cutoff,
            self.r_max,
            rel_tol=_CUTOFF_RELATIVE_TOLERANCE,
        ):
            raise ValueError(
                f"the batch's neighbour list was built with cutoff "
                f"{molecules.neighbor_cutoff}, the MACE model's is {self.r_max}: "
                f"collate with MoleculeCollator(cutoff=<the model's r_max>)"
            )
        missing_pairs = MINIMUM_PAIRS - edge_index.shape[1]
        if missing_pairs > 0:
            # Pairs of the first atom with itself: inert.
            edge_index = torch.cat(
                [edge_index, edge_index.new_zeros((2, missing_pairs))], dim=1
            )
        positions = positions.to(self.dtype)
        with torch.no_grad():
            if self._compiled_features is None:
                return self.features(atomic_numbers, positions, edge_index)
            for tensor in (atomic_numbers, positions):
                torch._dynamo.maybe_mark_dynamic(tensor, 0)
                torch._dynamo.maybe_mark_dynamic(tensor, 1)
            torch._dynamo.maybe_mark_dynamic(edge_index, 1)
            return self._compiled_features(atomic_numbers, positions, edge_index)

    def features(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """The computation of ``forward`` (the compiled region): ``edge_index``
        indexes the flattened padded atoms (``molecule * atoms + slot``)."""
        molecules, atoms = atomic_numbers.shape
        flat_atomic_numbers = atomic_numbers.reshape(-1)
        real_atom = flat_atomic_numbers > 0
        element = self.element_index[
            flat_atomic_numbers.clamp(0, _LARGEST_ATOMIC_NUMBER)
        ]
        torch._assert_async(
            ((element >= 0) | ~real_atom).all()
            & (flat_atomic_numbers <= _LARGEST_ATOMIC_NUMBER).all(),
            "an atom's element is not one the MACE model was trained on",
        )
        node_attrs = torch.nn.functional.one_hot(
            element.clamp_min(0), self.number_of_elements
        ).to(self.dtype) * real_atom[:, None].to(self.dtype)

        sender, receiver = edge_index[0], edge_index[1]
        torch._assert_async(
            (sender // atoms == receiver // atoms).all(),
            "a pair of the neighbour list joins atoms of different molecules",
        )
        real_pair = real_atom[sender] & real_atom[receiver] & (sender != receiver)
        flat_positions = positions.reshape(-1, 3)
        vectors = flat_positions[receiver] - flat_positions[sender]
        inert = vectors.new_tensor([self.inert_pair_distance, 0.0, 0.0])
        vectors = torch.where(real_pair[:, None], vectors, inert)
        lengths = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)

        if self.permute_edge_vectors:
            edge_attrs = self.spherical_harmonics(vectors[..., [1, 2, 0]])
        else:
            edge_attrs = self.spherical_harmonics(vectors)
        edge_feats, cutoff = self.radial_embedding(
            lengths, node_attrs, edge_index, self.model_atomic_numbers
        )

        node_feats = self.node_embedding(node_attrs)
        layer_features = []
        for layer, (interaction, product) in enumerate(
            zip(self.interactions, self.products, strict=True)
        ):
            node_feats, skip = interaction(
                node_attrs=node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
                cutoff=cutoff,
                first_layer=(layer == 0),
            )
            node_feats = product(node_feats=node_feats, sc=skip, node_attrs=node_attrs)
            layer_features.append(node_feats)

        features = torch.cat(layer_features, dim=-1)
        if self.layout == "ir_mul":
            features = features.index_select(-1, self.mul_ir_index)
        features = features * real_atom[:, None].to(features.dtype)
        return features.reshape(molecules, atoms, -1)
