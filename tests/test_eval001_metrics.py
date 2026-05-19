import numpy as np

from threedscriptors.evaluation.eval001.datasets import TDC_TASK_BY_NAME
from threedscriptors.evaluation.eval001.metrics import binary_metric, regression_metric


def test_binary_metric_supports_auprc_aliases():
    y_true = np.array([0, 1, 1, 0])
    y_score = np.array([0.1, 0.9, 0.8, 0.2])

    result = binary_metric(y_true, y_score, "PR-AUC")

    assert result.metric_name == "AUPRC"
    assert result.value == 1.0
    assert result.n_labels_scored == 1


def test_regression_metric_supports_spearman():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([10.0, 20.0, 30.0, 40.0])

    result = regression_metric(y_true, y_pred, "Spearman")

    assert result.metric_name == "Spearman"
    assert result.value == 1.0
    assert result.n_labels_scored == 1


def test_tdc_official_metric_registry_matches_imbalanced_tasks():
    assert TDC_TASK_BY_NAME["CYP2C9_Veith"].mumo_metric == "AUPRC"
    assert TDC_TASK_BY_NAME["CYP2D6_Veith"].mumo_metric == "AUPRC"
    assert TDC_TASK_BY_NAME["CYP3A4_Veith"].mumo_metric == "AUPRC"
    assert TDC_TASK_BY_NAME["VDss_Lombardo"].mumo_metric == "Spearman"
    assert len(TDC_TASK_BY_NAME) == 22
