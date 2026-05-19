from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import yaml

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from threedscriptors.evaluation.eval001.datasets import (
    canonical_smiles,
    canonical_smiles_polar,
    load_moleculenet_raw,
    select_moleculenet_tasks,
    select_tdc_tasks,
)
from threedscriptors.evaluation.eval001.descriptors import compute_ecfp, read_npz_cache
from threedscriptors.evaluation.eval001.heads import (
    fit_predict_binary,
    fit_predict_multilabel,
    fit_predict_regression,
    validate_mlp_device,
)
from threedscriptors.evaluation.eval001.metrics import (
    binary_metric,
    multilabel_macro_auroc,
    regression_metric,
)
from threedscriptors.evaluation.eval001.mapping import (
    is_polar_standardisation,
    load_moleculenet_source_to_zarr,
    load_tdc_index_mapping,
    map_source_indices_to_zarr,
    map_tdc_frames_to_source_indices,
    map_tdc_smiles,
    reconstruct_moleculenet_source_to_zarr,
    reconstruct_tdc_source_to_zarr,
    tdc_mapping_for_task,
    tdc_source_to_zarr,
)
from threedscriptors.evaluation.eval001.splits import (
    deepchem_scaffold_split,
    random_train_val_test_split,
    scaffold_train_val_test_split,
)


DEFAULT_CONFIG = Path("configs/eval/2026-04-27_eval_001_mumo_core.yaml")

RESULT_COLUMNS = (
    "suite", "dataset", "task", "split_variant", "metric_name", "metric_mode",
    "row", "head", "seed", "value", "value_std", "n_train", "n_val", "n_test",
    "n_labels", "n_labels_scored", "coverage", "coverage_limited",
    "descriptor_cache", "source_table", "notes",
)


def _result_row(
    *,
    suite: str,
    dataset: str,
    task: str,
    split_variant: str,
    metric_name: str,
    metric_mode: str,
    row: str,
    head: str,
    seed,
    value,
    n_test,
    n_train="",
    n_val="",
    value_std="",
    n_labels: int = 1,
    n_labels_scored: int = 1,
    coverage: float = 1.0,
    coverage_limited: bool = False,
    descriptor_cache: str = "",
    source_table: str = "",
    notes: str = "",
) -> dict:
    """Single definition of the EVAL-001 shard-CSV schema (21 columns).

    Defaults are the common case (full-coverage per-seed row); aggregate /
    skip rows pass the few fields that differ explicitly.
    """
    return {
        "suite": suite,
        "dataset": dataset,
        "task": task,
        "split_variant": split_variant,
        "metric_name": metric_name,
        "metric_mode": metric_mode,
        "row": row,
        "head": head,
        "seed": seed,
        "value": value,
        "value_std": value_std,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "n_labels": n_labels,
        "n_labels_scored": n_labels_scored,
        "coverage": coverage,
        "coverage_limited": coverage_limited,
        "descriptor_cache": descriptor_cache,
        "source_table": source_table,
        "notes": notes,
    }


class SourceToZarrResolution(NamedTuple):
    """Result of resolving a source-row → zarr-row mapping for one task.

    `forces_coverage_limited` carries the mapping-quality decision explicitly
    instead of re-parsing `note` for control flow.
    """

    source_to_zarr: dict | None
    note: str
    forces_coverage_limited: bool = False


def _molnet_scaffold(args, smiles, seed):
    """Pick the MolNet split based on --molnet-splitter flag.
    'deepchem' (default) = canonical Hu et al. 2020 / Uni-Mol 2023 scaffold protocol.
    'legacy' = our custom Bemis-Murcko seeded splitter (pre-publication-grade).
    'random' = random 80/10/10 split (MuMo headline / MoLFormer regression panel
               convention; weaker than scaffold but produces alternative supervisor-
               facing numbers).
    """
    splitter = getattr(args, "molnet_splitter", "deepchem")
    if splitter == "random":
        return random_train_val_test_split(len(smiles), int(seed))
    if splitter == "legacy":
        return scaffold_train_val_test_split(smiles, int(seed))
    return deepchem_scaffold_split(smiles, int(seed), include_chirality=True)


def _metric_for_task(task_type: str, default_metric: str, y_true, y_pred):
    if task_type == "regression":
        return regression_metric(y_true, y_pred, default_metric)
    if task_type == "binary":
        return binary_metric(y_true, y_pred, default_metric)
    if task_type == "multilabel":
        return multilabel_macro_auroc(y_true, y_pred)
    raise ValueError(f"Unknown task type: {task_type}")


def _tdc_stats(official_result, task_name: str):
    if not isinstance(official_result, dict):
        raise RuntimeError(str(official_result))
    return official_result.get(task_name.lower(), official_result.get(task_name))


def _append_tdc_official_skip(
    rows: list[dict],
    *,
    spec,
    row_name: str,
    head: str,
    n_test: int,
    coverage: float,
    coverage_limited: bool,
    descriptor_cache: str = "",
    notes: str,
) -> None:
    rows.append(
        _result_row(
            suite="tdc",
            dataset=spec.name,
            task=spec.short_name,
            split_variant="tdc_default",
            metric_name=spec.mumo_metric,
            metric_mode="tdc_official",
            row=row_name,
            head=head,
            seed="aggregate",
            value=float("nan"),
            n_test=n_test,
            n_labels_scored=0,
            coverage=coverage,
            coverage_limited=coverage_limited,
            descriptor_cache=descriptor_cache,
            notes=notes,
        )
    )


def _fit_predict(task_type: str, head: str, X_train, y_train, X_val, y_val, X_test, seed, mlp_cfg):
    kwargs = {
        "max_epochs": int(mlp_cfg.get("max_epochs", 100)),
        "patience": int(mlp_cfg.get("early_stop_patience", 10)),
        "fast": bool(mlp_cfg.get("fast", False)),
        "mlp_device": str(mlp_cfg.get("device", "cuda")),
        "allow_cpu_mlp": bool(mlp_cfg.get("allow_cpu_mlp", False)),
    }
    if task_type == "regression":
        return fit_predict_regression(head, X_train, y_train, X_val, y_val, X_test, seed, **kwargs)
    if task_type == "binary":
        return fit_predict_binary(head, X_train, y_train, X_val, y_val, X_test, seed, **kwargs)
    if task_type == "multilabel":
        return fit_predict_multilabel(head, X_train, y_train, X_val, y_val, X_test, seed, **kwargs)
    raise ValueError(f"Unknown task type: {task_type}")


def _resolve_tdc_source_to_zarr(
    config: dict,
    task_name: str,
    x_full_len: int,
    index_mapping: dict | None,
    all_source_df: pd.DataFrame,
    log_prefix: str,
) -> SourceToZarrResolution:
    """Resolve the TDC source→zarr map, with the documented fallback ladder.

    Order: zarr reconstruction → generator mapping → SMILES fallback. The last
    one is `forces_coverage_limited` because it cannot guarantee canonical
    coverage of the official split.
    """
    if index_mapping is None:
        return SourceToZarrResolution(None, "mapping=smiles_fallback")
    try:
        source_to_zarr = reconstruct_tdc_source_to_zarr(config, task_name, all_source_df)
        return SourceToZarrResolution(source_to_zarr, "mapping=zarr_reconstructed")
    except Exception as exc:
        print(
            f"{log_prefix}: zarr/source reconstruction failed ({exc!r}); "
            "falling back to generator mapping"
        )
        fallback = tdc_source_to_zarr(index_mapping)
        if fallback and max(fallback.values()) >= x_full_len:
            print(
                f"{log_prefix}: generator mapping is stale "
                f"(max zarr index {max(fallback.values())}, descriptor rows {x_full_len}); "
                "using SMILES fallback as coverage_limited diagnostic only."
            )
            return SourceToZarrResolution(
                None,
                f"mapping=smiles_fallback_after_stale_generator:{type(exc).__name__}",
                forces_coverage_limited=True,
            )
        return SourceToZarrResolution(fallback, "mapping=generator_mapping")


def _resolve_moleculenet_source_to_zarr(
    out_root: Path,
    dataset: str,
    raw_smiles: list[str],
    x_full_len: int,
    is_polar: bool,
    log_prefix: str,
) -> SourceToZarrResolution:
    """Resolve the MoleculeNet source→zarr map.

    POLAR zarrs store post-standardisation SMILES, so the persisted
    source_index_mapping.json does not apply; rebuild from the zarr's stored
    SMILES. Non-POLAR mappings are also rebuilt if they are stale relative to
    the descriptor cache.
    """
    source_to_zarr = load_moleculenet_source_to_zarr(out_root, dataset)
    if is_polar:
        old_size = len(source_to_zarr)
        old_max = max(source_to_zarr.values()) if source_to_zarr else -1
        source_to_zarr = reconstruct_moleculenet_source_to_zarr(out_root, dataset, raw_smiles)
        print(
            f"{log_prefix}: reconstructed POLAR MoleculeNet mapping "
            f"from zarr/isomeric_smiles.txt (source map size {old_size}, "
            f"max zarr index {old_max}, descriptor rows {x_full_len}, "
            f"reconstructed {len(source_to_zarr)} rows)"
        )
        return SourceToZarrResolution(
            source_to_zarr, "mapping=zarr_smiles_reconstructed_for_polar_standardisation"
        )
    # Genuine out-of-bounds guard only. A benign count mismatch (canonical
    # dedupe in the raw release vs the filtered zarr) is NOT staleness; treating
    # it as such flips the mapping method per-descriptor and would score the
    # POLAR and OFF24 rows of one benchmark on different molecule sets.
    stale_mapping = bool(source_to_zarr) and max(source_to_zarr.values()) >= x_full_len
    if stale_mapping:
        old_size = len(source_to_zarr)
        old_max = max(source_to_zarr.values())
        source_to_zarr = reconstruct_moleculenet_source_to_zarr(out_root, dataset, raw_smiles)
        print(
            f"{log_prefix}: source_index_mapping.json is stale "
            f"(size {old_size}, max zarr index {old_max}, descriptor rows {x_full_len}); "
            f"reconstructed {len(source_to_zarr)} rows from zarr/isomeric_smiles.txt"
        )
        return SourceToZarrResolution(
            source_to_zarr, "mapping=zarr_smiles_reconstructed_after_stale_source_mapping"
        )
    return SourceToZarrResolution(source_to_zarr, "mapping=raw_release_to_native_zarr")


def run_tdc(config: dict, args) -> list[dict]:
    from tdc.benchmark_group import admet_group

    group = admet_group(path=str(config["paths"]["tdc_cache"]))
    specs = select_tdc_tasks(args.datasets, args.limit)
    seeds = args.seeds or config["tdc"]["seeds"]
    rows: list[dict] = []
    mlp_cfg = config.get("heads", {}).get("mlp", {})

    for spec in specs:
        benchmark = group.get(spec.name)
        test_df = benchmark["test"].reset_index(drop=True)
        y_test = test_df["Y"].to_numpy(dtype=float)
        split_cache = {}
        unique_smiles = set(test_df["Drug"].astype(str).tolist())
        for seed in seeds:
            train_df, val_df = group.get_train_valid_split(
                benchmark=spec.name, split_type="default", seed=int(seed)
            )
            split_cache[int(seed)] = (train_df, val_df)
            unique_smiles.update(train_df["Drug"].astype(str).tolist())
            unique_smiles.update(val_df["Drug"].astype(str).tolist())
        unique_smiles_list = sorted(unique_smiles)
        X_unique = compute_ecfp(unique_smiles_list)
        ecfp_cache = {smi: X_unique[i] for i, smi in enumerate(unique_smiles_list)}

        def ecfp_for(series: pd.Series) -> np.ndarray:
            return np.stack([ecfp_cache[s] for s in series.astype(str).tolist()])

        X_test = ecfp_for(test_df["Drug"])
        for head in args.heads:
            pred_by_seed = []
            for seed in seeds:
                train_df, val_df = split_cache[int(seed)]
                X_train = ecfp_for(train_df["Drug"])
                y_train = train_df["Y"].to_numpy(dtype=float)
                X_val = ecfp_for(val_df["Drug"])
                y_val = val_df["Y"].to_numpy(dtype=float)
                fit = _fit_predict(
                    spec.task_type, head, X_train, y_train, X_val, y_val, X_test, int(seed), mlp_cfg
                )
                pred_by_seed.append(np.asarray(fit.predictions))
                mumo_metric = _metric_for_task(spec.task_type, spec.mumo_metric, y_test, fit.predictions)
                rows.append(
                    _result_row(
                        suite="tdc",
                        dataset=spec.name,
                        task=spec.short_name,
                        split_variant="tdc_default",
                        metric_name=mumo_metric.metric_name,
                        metric_mode="mumo_table",
                        row="ECFP",
                        head=head,
                        seed=seed,
                        value=mumo_metric.value,
                        n_train=len(train_df),
                        n_val=len(val_df),
                        n_test=len(test_df),
                        n_labels=mumo_metric.n_labels,
                        n_labels_scored=mumo_metric.n_labels_scored,
                        notes=fit.notes or mumo_metric.notes,
                    )
                )

            if args.metric_mode in {"all", "tdc_official"}:
                if len(pred_by_seed) < 5:
                    _append_tdc_official_skip(
                        rows,
                        spec=spec,
                        row_name="ECFP",
                        head=head,
                        n_test=len(test_df),
                        coverage=1.0,
                        coverage_limited=False,
                        notes=(
                            "group.evaluate_many_skipped:requires_at_least_5_prediction_runs;"
                            f"got_{len(pred_by_seed)};local_diagnostic_only"
                        ),
                    )
                    continue
                try:
                    official = group.evaluate_many([{spec.name: p} for p in pred_by_seed])
                    stats = _tdc_stats(official, spec.name)
                    if stats is not None:
                        rows.append(
                            _result_row(
                                suite="tdc",
                                dataset=spec.name,
                                task=spec.short_name,
                                split_variant="tdc_default",
                                metric_name=spec.mumo_metric,
                                metric_mode="tdc_official",
                                row="ECFP",
                                head=head,
                                seed="aggregate",
                                value=float(stats[0]),
                                value_std=float(stats[1]),
                                n_test=len(test_df),
                                notes="group.evaluate_many",
                            )
                        )
                except Exception as exc:
                    _append_tdc_official_skip(
                        rows,
                        spec=spec,
                        row_name="ECFP",
                        head=head,
                        n_test=len(test_df),
                        coverage=1.0,
                        coverage_limited=False,
                        notes=f"group.evaluate_many_failed:{exc!r}",
                    )
    return rows


def _tdc_cache_path(config: dict, dataset: str, row: str) -> Path:
    return Path(config["paths"]["descriptor_cache_root"]) / dataset / f"{row}.npz"


def run_tdc_cached(config: dict, args) -> list[dict]:
    from tdc.benchmark_group import admet_group

    group = admet_group(path=str(config["paths"]["tdc_cache"]))
    specs = select_tdc_tasks(args.datasets, args.limit)
    seeds = args.seeds or config["tdc"]["seeds"]
    rows: list[dict] = []
    mlp_cfg = config.get("heads", {}).get("mlp", {})
    row_name = args.descriptor
    threshold_reg = float(config["coverage_thresholds"]["regression"])
    threshold_class = float(config["coverage_thresholds"]["classification"])

    for spec in specs:
        cache_path = _tdc_cache_path(config, spec.name, row_name)
        if not cache_path.exists():
            print(f"{spec.name}/{row_name}: cache missing at {cache_path}; skipping")
            continue
        X_full, _ = read_npz_cache(cache_path)
        index_mapping = load_tdc_index_mapping(config, spec.name)
        exact, canon = tdc_mapping_for_task(config, spec.name)
        benchmark = group.get(spec.name)
        train_val_source_df = benchmark["train_val"].reset_index(drop=True)
        test_df = benchmark["test"].reset_index(drop=True)
        all_source_df = pd.concat([train_val_source_df, test_df], ignore_index=True)
        res = _resolve_tdc_source_to_zarr(
            config, spec.name, len(X_full), index_mapping, all_source_df,
            log_prefix=f"{spec.name}/{row_name}",
        )
        source_to_zarr = res.source_to_zarr
        mapping_note = res.note

        if source_to_zarr is not None:
            test_source_idx = np.arange(
                len(train_val_source_df),
                len(train_val_source_df) + len(test_df),
                dtype=int,
            )
            test_idx, test_mask = map_source_indices_to_zarr(test_source_idx, source_to_zarr)
        else:
            test_idx, test_mask = map_tdc_smiles(test_df["Drug"], exact, canon)
        y_test_full = test_df["Y"].to_numpy(dtype=float)
        y_test = y_test_full[test_mask]
        coverage = float(test_mask.mean())
        coverage_threshold = threshold_reg if spec.task_type == "regression" else threshold_class
        coverage_limited = (coverage < coverage_threshold) or res.forces_coverage_limited
        X_test = X_full[test_idx]

        # Train/val split → index mapping depends only on seed (not head); compute
        # once per seed and reuse across heads.
        seed_data = {}
        for seed in seeds:
            train_df, val_df = group.get_train_valid_split(
                benchmark=spec.name, split_type="default", seed=int(seed)
            )
            if source_to_zarr is not None:
                train_source_idx, val_source_idx = map_tdc_frames_to_source_indices(
                    [train_df, val_df], train_val_source_df, offset=0
                )
                train_idx, train_mask = map_source_indices_to_zarr(train_source_idx, source_to_zarr)
                val_idx, val_mask = map_source_indices_to_zarr(val_source_idx, source_to_zarr)
            else:
                train_idx, train_mask = map_tdc_smiles(train_df["Drug"], exact, canon)
                val_idx, val_mask = map_tdc_smiles(val_df["Drug"], exact, canon)
            y_train = train_df["Y"].to_numpy(dtype=float)[train_mask]
            y_val = val_df["Y"].to_numpy(dtype=float)[val_mask]
            assert len(train_idx) == int(train_mask.sum())
            assert len(val_idx) == int(val_mask.sum())
            seed_data[int(seed)] = (train_idx, val_idx, y_train, y_val, train_mask, val_mask)

        for head in args.heads:
            pred_by_seed = []
            for seed in seeds:
                train_idx, val_idx, y_train, y_val, train_mask, val_mask = seed_data[int(seed)]
                fit = _fit_predict(
                    spec.task_type,
                    head,
                    X_full[train_idx],
                    y_train,
                    X_full[val_idx],
                    y_val,
                    X_test,
                    int(seed),
                    mlp_cfg,
                )
                pred_by_seed.append(np.asarray(fit.predictions))
                mumo_metric = _metric_for_task(spec.task_type, spec.mumo_metric, y_test, fit.predictions)
                rows.append(
                    _result_row(
                        suite="tdc",
                        dataset=spec.name,
                        task=spec.short_name,
                        split_variant="tdc_default",
                        metric_name=mumo_metric.metric_name,
                        metric_mode="mumo_table",
                        row=row_name,
                        head=head,
                        seed=seed,
                        value=mumo_metric.value,
                        n_train=int(train_mask.sum()),
                        n_val=int(val_mask.sum()),
                        n_test=len(test_df),
                        n_labels=mumo_metric.n_labels,
                        n_labels_scored=mumo_metric.n_labels_scored,
                        coverage=coverage,
                        coverage_limited=coverage_limited,
                        descriptor_cache=str(cache_path),
                        notes=";".join(
                            n for n in [mapping_note, fit.notes or mumo_metric.notes] if n
                        ),
                    )
                )

            if args.metric_mode in {"all", "tdc_official"}:
                if len(pred_by_seed) < 5:
                    _append_tdc_official_skip(
                        rows,
                        spec=spec,
                        row_name=row_name,
                        head=head,
                        n_test=len(test_df),
                        coverage=coverage,
                        coverage_limited=coverage_limited,
                        descriptor_cache=str(cache_path),
                        notes=(
                            "group.evaluate_many_skipped:requires_at_least_5_prediction_runs;"
                            f"got_{len(pred_by_seed)};local_diagnostic_only"
                        ),
                    )
                    continue
                if coverage < 1.0 and not args.allow_official_coverage_imputation:
                    _append_tdc_official_skip(
                        rows,
                        spec=spec,
                        row_name=row_name,
                        head=head,
                        n_test=len(test_df),
                        coverage=coverage,
                        coverage_limited=True,
                        descriptor_cache=str(cache_path),
                        notes="group.evaluate_many_skipped:coverage_less_than_1.0;no_molecule_drop_allowed",
                    )
                    continue
                full_preds = []
                for pred in pred_by_seed:
                    full = np.full(len(test_df), float(np.nanmean(y_test_full)))
                    full[test_mask] = pred
                    full_preds.append(full)
                try:
                    official = group.evaluate_many([{spec.name: p} for p in full_preds])
                    stats = _tdc_stats(official, spec.name)
                except Exception as exc:
                    stats = None
                    note = f"group.evaluate_many_failed:{exc!r}"
                else:
                    note = "group.evaluate_many"
                if stats is not None:
                    rows.append(
                        _result_row(
                            suite="tdc",
                            dataset=spec.name,
                            task=spec.short_name,
                            split_variant="tdc_default",
                            metric_name=spec.mumo_metric,
                            metric_mode="tdc_official",
                            row=row_name,
                            head=head,
                            seed="aggregate",
                            value=float(stats[0]),
                            value_std=float(stats[1]),
                            n_test=len(test_df),
                            coverage=coverage,
                            coverage_limited=coverage_limited,
                            descriptor_cache=str(cache_path),
                            notes=note,
                        )
                    )
    return rows


def run_moleculenet(config: dict, args) -> list[dict]:
    raw_root = Path(config["paths"]["moleculenet_raw_root"])
    specs = select_moleculenet_tasks(raw_root, args.datasets, args.limit)
    seeds = args.seeds or config["moleculenet"]["seeds"]
    rows: list[dict] = []
    mlp_cfg = config.get("heads", {}).get("mlp", {})

    for spec in specs:
        smiles, y, counts = load_moleculenet_raw(spec)
        X = compute_ecfp(smiles)
        for seed in seeds:
            if spec.split_variant == "scaffold":
                train_idx, val_idx, test_idx = _molnet_scaffold(args, smiles, seed)
            else:
                strat_y = y if np.asarray(y).ndim == 1 else None
                train_idx, val_idx, test_idx = random_train_val_test_split(len(smiles), int(seed), strat_y)

            for head in args.heads:
                fit = _fit_predict(
                    spec.task_type,
                    head,
                    X[train_idx],
                    y[train_idx],
                    X[val_idx],
                    y[val_idx],
                    X[test_idx],
                    int(seed),
                    mlp_cfg,
                )
                metric = _metric_for_task(spec.task_type, spec.metric, y[test_idx], fit.predictions)
                rows.append(
                    _result_row(
                        suite="moleculenet",
                        dataset=spec.dataset,
                        task=spec.task,
                        split_variant=spec.split_variant,
                        metric_name=metric.metric_name,
                        metric_mode="mumo_table",
                        row="ECFP",
                        head=head,
                        seed=seed,
                        value=metric.value,
                        n_train=len(train_idx),
                        n_val=len(val_idx),
                        n_test=len(test_idx),
                        n_labels=metric.n_labels,
                        n_labels_scored=metric.n_labels_scored,
                        coverage=counts["n_valid_smiles"] / max(counts["n_source_rows"], 1),
                        notes=fit.notes or metric.notes,
                    )
                )
    return rows


def run_moleculenet_cached(config: dict, args) -> list[dict]:
    """Cached MoleculeNet path: read MACE/REM3DI features from .npz cache.

    Splits are computed over the raw canonical release, then mapped into the
    native zarr. This keeps OFF24 coverage limitations visible instead of
    silently splitting only the filtered zarr subset.
    """
    raw_root = Path(config["paths"]["moleculenet_raw_root"])
    out_root = Path(config["paths"]["moleculenet_output_root"])
    specs = select_moleculenet_tasks(raw_root, args.datasets, args.limit)
    seeds = args.seeds or config["moleculenet"]["seeds"]
    rows: list[dict] = []
    mlp_cfg = config.get("heads", {}).get("mlp", {})
    row_name = args.descriptor
    threshold_reg = float(config["coverage_thresholds"]["regression"])
    threshold_class = float(config["coverage_thresholds"]["classification"])

    for spec in specs:
        cache_path = Path(config["paths"]["descriptor_cache_root"]) / spec.dataset / f"{row_name}.npz"
        if not cache_path.exists():
            print(f"{spec.dataset}/{row_name}: cache missing at {cache_path}; skipping")
            continue
        X_full, _ = read_npz_cache(cache_path)

        # POLAR zarrs store post-standardisation SMILES; the canonicaliser must
        # match the zarr's standardisation or the raw→zarr lookup misses ~55%
        # of BACE-S, ~60% of ClinTox, etc. (the May-2026 30% coverage bug).
        # Standardisation is resolved explicitly (--standardisation), falling
        # back to path inference only in 'auto' mode.
        is_polar = is_polar_standardisation(
            config, auto_path=str(out_root), auto_token="moleculenet_polar"
        )
        canonicaliser = canonical_smiles_polar if is_polar else canonical_smiles
        raw_smiles, y_raw, counts = load_moleculenet_raw(spec, canonicaliser)
        res = _resolve_moleculenet_source_to_zarr(
            out_root, spec.dataset, raw_smiles, len(X_full), is_polar,
            log_prefix=f"{spec.dataset}/{row_name}",
        )
        source_to_zarr = res.source_to_zarr
        mapping_note = res.note
        source_coverage = len(source_to_zarr) / max(counts["n_source_rows"], 1)
        raw_to_zarr = {
            i: source_to_zarr[s] for i, s in enumerate(raw_smiles) if s in source_to_zarr
        }

        for seed in seeds:
            if spec.split_variant == "scaffold":
                train_raw_idx, val_raw_idx, test_raw_idx = _molnet_scaffold(args, raw_smiles, seed)
            else:
                strat_y = y_raw if np.asarray(y_raw).ndim == 1 else None
                train_raw_idx, val_raw_idx, test_raw_idx = random_train_val_test_split(
                    len(raw_smiles), int(seed), strat_y
                )

            train_idx, train_keep = map_source_indices_to_zarr(np.asarray(train_raw_idx), raw_to_zarr)
            val_idx, val_keep = map_source_indices_to_zarr(np.asarray(val_raw_idx), raw_to_zarr)
            test_idx, test_keep = map_source_indices_to_zarr(np.asarray(test_raw_idx), raw_to_zarr)
            y_train = y_raw[train_raw_idx][train_keep]
            y_val = y_raw[val_raw_idx][val_keep]
            y_test = y_raw[test_raw_idx][test_keep]
            assert len(train_idx) == int(train_keep.sum())
            assert len(val_idx) == int(val_keep.sum())
            assert len(test_idx) == int(test_keep.sum())
            coverage = float(test_keep.mean()) if len(test_keep) else 0.0
            coverage_threshold = threshold_reg if spec.task_type == "regression" else threshold_class
            coverage_limited = coverage < coverage_threshold

            for head in args.heads:
                fit = _fit_predict(
                    spec.task_type,
                    head,
                    X_full[train_idx],
                    y_train,
                    X_full[val_idx],
                    y_val,
                    X_full[test_idx],
                    int(seed),
                    mlp_cfg,
                )
                metric = _metric_for_task(spec.task_type, spec.metric, y_test, fit.predictions)
                rows.append(
                    _result_row(
                        suite="moleculenet",
                        dataset=spec.dataset,
                        task=spec.task,
                        split_variant=spec.split_variant,
                        metric_name=metric.metric_name,
                        metric_mode="mumo_table",
                        row=row_name,
                        head=head,
                        seed=seed,
                        value=metric.value,
                        n_train=int(train_keep.sum()),
                        n_val=int(val_keep.sum()),
                        n_test=len(test_raw_idx),
                        n_labels=metric.n_labels,
                        n_labels_scored=metric.n_labels_scored,
                        coverage=coverage,
                        coverage_limited=coverage_limited,
                        descriptor_cache=str(cache_path),
                        notes=";".join(
                            n
                            for n in [
                                f"{mapping_note};source_coverage={source_coverage:.4f};"
                                f"train_coverage={train_keep.mean() if len(train_keep) else 0.0:.4f};"
                                f"val_coverage={val_keep.mean() if len(val_keep) else 0.0:.4f};"
                                f"test_coverage={coverage:.4f}",
                                fit.notes or metric.notes,
                            ]
                            if n
                        ),
                    )
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EVAL-001 raw/zarr evaluation shards.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--suite", choices=["tdc", "moleculenet", "all"], default="all")
    parser.add_argument("--descriptor", default="ECFP")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--heads", nargs="+", default=["linear", "lightgbm"])
    parser.add_argument(
        "--mlp-device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device for head=mlp. Defaults to CUDA and refuses CPU fallback.",
    )
    parser.add_argument(
        "--allow-cpu-mlp",
        action="store_true",
        help=(
            "Explicitly allow head=mlp on CPU. Use only for tiny diagnostics; "
            "full panels should run MLP on GPU."
        ),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--molnet-splitter",
        choices=["deepchem", "legacy", "random"],
        default="deepchem",
        help="Split strategy for MoleculeNet tasks. "
             "'deepchem' (default) = Hu 2020 / Uni-Mol 2023 deterministic "
             "scaffold split with include_chirality=True (publication-grade). "
             "'legacy' = our custom Bemis-Murcko seeded splitter (reproduces "
             "historical shards pre-2026-05-12). "
             "'random' = stratified random 80/10/10, matches MuMo headline + "
             "MoLFormer regression panel conventions.",
    )
    parser.add_argument(
        "--standardisation",
        choices=["auto", "off24", "polar"],
        default="auto",
        help="Which SMILES standardisation the descriptor zarr/cache was built "
             "with. 'polar' = salt-strip + uncharge (MACE-POLAR family); "
             "'off24' = plain RDKit canonical (MACE-OFF24 family); 'auto' "
             "(default) infers from the dataset path (legacy behaviour). Set "
             "this explicitly when evaluating on non-standard cache paths so "
             "the raw→zarr SMILES lookup is not silently mismatched.",
    )
    parser.add_argument("--metric-mode", choices=["all", "tdc_official", "mumo_table"], default="all")
    parser.add_argument(
        "--allow-official-coverage-imputation",
        action="store_true",
        help=(
            "Allow legacy cached TDC official scoring to fill missing canonical test rows. "
            "Leave off for fair-comparison runs."
        ),
    )
    parser.add_argument(
        "--fast-smoke",
        action="store_true",
        help="Use reduced grids/epochs for smoke gates; do not use for final tables.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--descriptor-cache-root",
        type=Path,
        default=None,
        help="Override config['paths']['descriptor_cache_root']. Useful when "
             "the descriptor cache for this run lives in a non-default location "
             "(e.g. descriptors_canonical_v1/ vs descriptors/).",
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    config["standardisation"] = args.standardisation
    if args.descriptor_cache_root is not None:
        config.setdefault("paths", {})["descriptor_cache_root"] = str(args.descriptor_cache_root)
    if args.fast_smoke:
        config.setdefault("heads", {}).setdefault("mlp", {})
        config["heads"]["mlp"]["fast"] = True
        config["heads"]["mlp"]["max_epochs"] = min(
            int(config["heads"]["mlp"].get("max_epochs", 100)), 20
        )
    config.setdefault("heads", {}).setdefault("mlp", {})
    config["heads"]["mlp"]["device"] = args.mlp_device
    config["heads"]["mlp"]["allow_cpu_mlp"] = bool(args.allow_cpu_mlp)
    if "mlp" in args.heads:
        validate_mlp_device(args.mlp_device, args.allow_cpu_mlp)
    rows: list[dict] = []
    if args.descriptor == "ECFP":
        if args.suite in {"tdc", "all"}:
            rows.extend(run_tdc(config, args))
        if args.suite in {"moleculenet", "all"}:
            rows.extend(run_moleculenet(config, args))
    else:
        if args.suite in {"tdc", "all"}:
            rows.extend(run_tdc_cached(config, args))
        if args.suite in {"moleculenet", "all"}:
            rows.extend(run_moleculenet_cached(config, args))

    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = Path(config["paths"]["shard_root"]) / stamp / "shards" / "ecfp_raw.csv"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=list(RESULT_COLUMNS))
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} metric rows to {args.output}")
    if len(df):
        print(df.groupby(["suite", "dataset", "head"]).size().head(40).to_string())


if __name__ == "__main__":
    main()
