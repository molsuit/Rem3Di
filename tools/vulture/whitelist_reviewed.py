# Reviewed vulture whitelist: names vulture reports as unused that are kept on purpose.
# Every entry says why. Vulture only needs the names to appear here; this file is
# never imported or executed (excluded from ruff and ty). Pydantic fields and enum
# members are whitelisted automatically by generate_whitelist.py.
# ruff: noqa
# fmt: off

# --- Attribute assignments on objects vulture cannot see into ---------------------------------
# RDKit EmbedParameters / ETKDG properties, set on C++ objects and read by RDKit.
_.numThreads
_.maxIterations
_.useRandomCoords
_.useSmallRingTorsions
_.useExpTorsionAnglePrefs
_.enforceChirality
_.randomSeed
# torch._dynamo.config option, read by torch.
_.cache_size_limit
# ase.Atoms.pbc, read by ase (test_preprocessing_mace_inference.py).
_.pbc
# wandb run handle kept on the telemetry object.
_._run

# --- Language and library hooks called by name ------------------------------------------------
_._missing_  # enum lookup fallback (configuration/data_config.py)
__getattr__  # module-level lazy imports (data_handling/dataset_creation/__init__.py)
__dir__

# --- Record fields written for the reports, read by humans and serialization --------------------
# Dataclass fields of the ChiralCat preparer's audit records and Sample.
_.stereo_audit
_.old_smiles
_.old_min_distance
_.new_min_distance
_.atom_a
_.atom_b
_.pair_distance_matrix
_.chiral_embeddings
# Pydantic Annotated type alias used in field annotations (configuration/config_utils.py).
NumpyArrayType

# --- Model and training code kept for future work (user 2026-11-01) -----------------------------
# No caller on any branch today; alternatives and earlier research code that may come back.
MolecularDifferenceRegressor  # model/molecule_difference_regressor.py
MultiHeadSelfAttention  # model/multihead_self_attention.py
PairCrossAttentionBlock  # model/pair_block.py
FusedMultiHeadRegression  # model/regression_models.py
OddMLP  # alternative output MLP of ChiralEmbeddingModel (commented-out use)
_.get_remedi_descriptor  # REM3DIModel
_.d_cutoff  # GeometricPreprocessor
_.layer_norm  # PairBiasedAttention: part of the state dict; removing it breaks strict checkpoint loads
chiral_difference_loss  # training/regression_training.py, enantiomer-difference losses (CMRT work)
direct_difference_loss
BaseMultitaskLoss
DynamicallyWeighedMultitaskLoss
ChiralDifferenceLoss
get_random_mask  # training/pretraining.py
atomic_embedding_loss
cosine_matrix  # training/training_utils.py
DataNormalizationModule  # training/data/data_normalization.py
_.inverse_transform
_.get_atomic_embedding_normalization_constants
_._compute_regression_stats
_.get_pairwise_differences
_._bemis_murcko_scaffold_splitting  # training/data/dataset_splitting.py
_.visualise_splitting
_.general_split
_.materialise_dataset
benchmark_loader  # training/data/util.py
_.log_epoch  # training/telemetry.py
_.get_track_grad_norm_fn
_.plot_grad_norm
get_mace_calculator_embedding_dimension  # utils/model_utils.py
remove_equivariants
get_pseudoscalars_indices
_.insert_task_configs  # ArchitectureConfig
to_yaml  # configuration/config_utils.py
pretraining_padded_collate_fn  # data_handling/sample.py
paired_sample_collate_fn

# --- Latent-space analysis helpers kept for future analyses -------------------------------------
_cluster_label_str  # latent_evaluation/analysis_tasks.py
get_descriptor_channel_distribution  # latent_evaluation/capacity_diagnostic.py
write_report  # latent_evaluation/mace_invariant_stats.py
_.serialize  # latent_evaluation/runner.py

# --- Public API without an in-repository caller -----------------------------------------------
_.embed_atoms  # RemediCalculator: embed molecules that are not in a zarr
registered_kinds  # remedi_eval.framework.plotting: lists the plotter registry

# --- Old pretraining path, deleted with it in BENCHMARK_DATA_FORMAT.md §1.4 ---------------------
_.concatenate_datasets  # remedi_data/dataset/dataset_concatenation.py
_.concatenate_datasets_chunked
LabeldDatasetConcatenation
_.string_to_id  # remedi_data/dataset/smiles_storage.py
_.verify_integrity
TmqmGenerator  # data_handling/dataset_creation/generators/tmqm_generator.py
_._structure_atom_span  # MoleculeDataset
get_ase_atoms  # remedi_data/chemistry/data_utils.py
