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
# torch.library fake (meta) kernel, registered by decorator and called by torch.compile.
_pair_channel_matmul_fake  # model/pair_linear.py
# nn.Module submodule of a test model, reached through named_parameters (test_optimization.py).
_.pair_projection

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
_.d_cutoff  # GeometricPreprocessor
chiral_difference_loss  # training/regression_training.py, enantiomer-difference losses (CMRT work)
direct_difference_loss
BaseMultitaskLoss
DynamicallyWeighedMultitaskLoss
ChiralDifferenceLoss
get_random_mask  # training/pretraining.py
atomic_embedding_loss
cosine_matrix  # training/training_utils.py
benchmark_loader  # training/data/util.py
_.log_epoch  # training/telemetry.py
_.get_track_grad_norm_fn
_.plot_grad_norm
remove_equivariants  # utils/model_utils.py
get_pseudoscalars_indices
to_yaml  # configuration/config_utils.py
pretraining_padded_collate_fn  # training/data/sample.py
paired_sample_collate_fn

# --- Latent-space analysis helpers kept for future analyses -------------------------------------
_cluster_label_str  # remedi_eval/framework/tasks/descriptor_analysis/analysis_tasks.py
get_descriptor_channel_distribution  # .../descriptor_analysis/capacity_diagnostic.py
write_report  # remedi/latent_evaluation/mace_invariant_stats.py

# --- Public API without an in-repository caller -----------------------------------------------
_.embed_atoms  # RemediCalculator: embed molecules that are not in a dataset
registered_kinds  # remedi_eval.framework.plotting: lists the plotter registry
