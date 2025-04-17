from collections.abc import Callable

import numpy as np


def tanimoto_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the Tanimoto distance between two vectors a and b.

    Tanimoto similarity is defined as:
        sim(a, b) = dot(a, b) / (||a||^2 + ||b||^2 - dot(a, b))

    We define the distance as:
        distance = 1 - sim(a, b)

    If the denominator is 0 (which may happen for zero vectors),
    the function returns 0.0.
    """
    dot_product = np.dot(a, b)
    denominator = np.sum(a * a) + np.sum(b * b) - dot_product
    if denominator == 0:
        return (
            0.0  # handle zero vectors, or cases where both vectors have no information
        )
    return dot_product / denominator


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # TODO: Should we add a shift here, so that the cosine similarity is in the [0,1] interval? Linear shift/ Sigmoid?
    return (cos_sim + 1.0) / 2


def calculate_similiarities(
    reference_descriptor: np.ndarray, class_descriptors, similarity_fn: Callable
):
    similiarities = np.zeros(size=class_descriptors.shape[0])

    # vectorize the similiarities calculation
    for idx, desc in enumerate(class_descriptors):
        similiarities[idx] = similarity_fn(reference_descriptor, desc)

    return similiarities
