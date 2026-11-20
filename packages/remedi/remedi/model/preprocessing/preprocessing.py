import torch
from torch import nn

from remedi.model.featurizer import MaceFeaturizer, PaddedMolecules, pad_molecules
from remedi.model.preprocessing.atomic_descriptor_preprocessor import (
    AtomicDescriptorPreprocessor,
)
from remedi.model.preprocessing.geometric_preprocessor import (
    PairDistanceMatrixGeometricPreprocessor,
)
from remedi.training.data.sample import MoleculeCollator, PreprocessedSample, Sample


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


class PreprocessorWithAtomicEmbedding(Preprocessor):
    """Computes the atoms' MACE features on the fly from a flat batch of molecules
    (``molecule_collator``) and preprocesses them.

    The batch is padded to ``(molecules, atoms)`` on the device without host reads;
    the padded atom count is the longest molecule rounded up to a multiple of
    ``pad_multiple``, so torch.compile sees only a small set of distinct shapes
    (1 = no rounding).
    """

    def __init__(
        self,
        mace_featurizer: MaceFeaturizer,
        atomic_preprocessor,
        geometric_preprocessor: PairDistanceMatrixGeometricPreprocessor,
        pad_multiple: int = 1,
    ):
        super().__init__(atomic_preprocessor, geometric_preprocessor)
        self.mace_featurizer = mace_featurizer
        if pad_multiple <= 0:
            raise ValueError("pad_multiple must be a positive integer.")
        self.pad_multiple = int(pad_multiple)

    def molecule_collator(
        self, with_regression_targets: bool = False
    ) -> MoleculeCollator:
        """The collate function of the batches this preprocessor takes: flat
        molecules with their neighbour list at the MACE model's cutoff."""
        return MoleculeCollator(
            cutoff=self.mace_featurizer.r_max,
            with_regression_targets=with_regression_targets,
        )

    def _atomic_embeddings(
        self, sample: Sample
    ) -> tuple[torch.Tensor, PaddedMolecules]:
        """MACE node features ``(molecules, atoms, C)`` of a flat batch, zero on
        padding, with the padded molecules they were computed on."""
        molecules = pad_molecules(sample, self.pad_multiple)
        return self.mace_featurizer(molecules), molecules

    def raw_invariants(self, sample: Sample) -> torch.Tensor:
        """The unnormalised invariants ``(atoms, D)`` of a flat batch's real atoms:
        what the invariant normalisation statistics are estimated from. Leaves
        ``sample`` untouched."""
        features, molecules = self._atomic_embeddings(sample)
        invariants = self.atomic_preprocessor.select_invariants(features)
        return invariants[~molecules.padding_mask]

    def mean_raw_invariants(self, sample: Sample) -> torch.Tensor:
        """The unnormalised selected invariants averaged over each molecule's
        atoms, ``(systems, D)`` in the batch's system order: the mean-pooled
        MACE descriptor, the baseline for the learned one. Leaves ``sample``
        untouched."""
        features, molecules = self._atomic_embeddings(sample)
        invariants = self.atomic_preprocessor.select_invariants(features).to(
            torch.float64
        )
        totals = invariants.sum(dim=1)
        counts = molecules.atom_counts.to(totals.dtype)[:, None]
        return (totals / counts).to(torch.float32)

    def forward(self, sample: Sample) -> PreprocessedSample:
        features, molecules = self._atomic_embeddings(sample)

        # The pair-distance preprocessor gets each molecule's centred positions,
        # the frame MACE saw.
        sample.embeddings = features
        sample.atomic_positions = molecules.positions
        sample.padding_mask = molecules.padding_mask

        preprocessed = super().forward(sample)
        preprocessed.atomic_numbers = molecules.atomic_numbers
        return preprocessed
