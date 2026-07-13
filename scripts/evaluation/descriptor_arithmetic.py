import math

import matplotlib.pyplot as plt
import torch
from mace.calculators import MACECalculator

from remedi.configuration.architecture_config import (
    EncoderOnlyArchitectureConfig,
)
from remedi.data_handling.data_utils import get_ase_atoms, relax_atoms
from remedi.data_handling.pipelines import reload_dataset_pipeline
from remedi.evaluation.clustering import (
    UMAPCalculator,
)
from remedi.evaluation.evaluation_utils import (
    evaluate_molecular_descriptor_on_dataset,
)

# Load a model

model_directory = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/training_runs/209-2025_08_12_15_25_09-FixedPCQM"


mace_calc = MACECalculator(
    model_paths="/share/snw30/projects/mace_model/MACE-OFF24_medium.model"
)
model = EncoderOnlyArchitectureConfig.from_directory(model_directory).build(
    mace_calculator=mace_calc
)
model.encoder.load_state_dict(torch.load(f"{model_directory}/encoder.pth"))
model.preprocessor.atomic_preprocessor.load_state_dict(
    torch.load(f"{model_directory}/atomic_preprocessor.pth")
)
model.preprocessor.geometric_preprocessor.load_state_dict(
    torch.load(f"{model_directory}/geometric_preprocessor.pth")
)

model.eval()

# Very good example!
# smi_1 = "C1=C(O)CCCC1"
# smi_2 = "C1C(=O)CCCC1"
# smi_3 = "CC=C(O)C"
# smi_4 = "CCC(=O)C"


smi_1 = "C1CCCCCCC1"
smi_2 = "C1=CC=CC=CC=C1"
smi_3 = "C1CCCCC1"
smi_4 = "c1ccccc1"


def smi_to_desc(smi):
    a = get_ase_atoms(smi)
    relax_atoms(a, mace_calc, 0.003, max_steps=1000)
    # Flatten the (1, L, D) seed sequence so downstream arithmetic / cat /
    # cosine works on a 2D (1, L*D) tensor that matches what
    # `evaluate_molecular_descriptor_on_dataset` returns.
    return model.get_remedi_descriptor(a).flat, a


desc_0, a0 = smi_to_desc(smi_1)
desc_1, a1 = smi_to_desc(smi_2)
desc_2, a2 = smi_to_desc(smi_3)
desc_3, a3 = smi_to_desc(smi_4)


p_diff = desc_0 - desc_1
h_diff = desc_2 - desc_3

print(torch.linalg.norm(p_diff))
print(torch.linalg.norm(h_diff))
print(torch.linalg.norm(p_diff - h_diff))


cos = torch.nn.functional.cosine_similarity(p_diff, h_diff, dim=1)
print(cos)
breakpoint()

dataset_directory = (
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/qm9full"
)
dataset = reload_dataset_pipeline(dataset_directory).build()


clustering_calculator = UMAPCalculator()

descriptors = evaluate_molecular_descriptor_on_dataset(model, dataset)


descriptors = descriptors[:20000]

descriptors = torch.cat([descriptors, desc_0, desc_1, desc_2, desc_3], dim=0)


@torch.no_grad()
def mean_euclidean_distance_mc(
    X: torch.Tensor,
    num_pairs: int = 50_000,
    *,
    generator: torch.Generator | None = None,
    chunk_size: int = 200_000,
    return_stderr: bool = True,
):
    """
    Monte Carlo estimate of E[||X_i - X_j||_2] for i != j.
    X: (N, D) tensor
    num_pairs: number of random pairs to sample
    generator: optional torch.Generator for reproducibility
    chunk_size: compute distances in chunks to limit memory
    return_stderr: if True, also return standard error of the mean
    """
    N, D = X.shape
    if N < 2:
        raise ValueError("Need at least two points")

    device = X.device
    # Sample i ~ U{0..N-1}, j ~ U{0..N-1}\{i} using the 'skip i' trick
    i = torch.randint(N, (num_pairs,), device=device, generator=generator)
    j = torch.randint(N - 1, (num_pairs,), device=device, generator=generator)
    j = j + (j >= i)  # shift up values >= i so j != i uniformly

    # Compute distances in chunks
    samples = []
    for start in range(0, num_pairs, chunk_size):
        end = min(start + chunk_size, num_pairs)
        diff = X[i[start:end]] - X[j[start:end]]
        d = diff.square().sum(dim=1).sqrt()
        samples.append(d)

    dists = torch.cat(samples)
    mean = dists.mean()
    if return_stderr:
        stderr = dists.std(unbiased=True) / math.sqrt(num_pairs)
        return mean, stderr
    return mean


mean = mean_euclidean_distance_mc(
    descriptors, num_pairs=50_000_000, chunk_size=2_000_000
)


print(mean)


projection = clustering_calculator.get_dimensionality_reduction(descriptors)
plt.figure(figsize=(10, 10))
plt.scatter(projection[:-4, 0], projection[:-4, 1], s=1, alpha=0.5)
plt.scatter(
    projection[-4:, 0],
    projection[-4:, 1],
    s=50,
    c="red",
    label="Pentol, Pentanon, Hexanol, Hexanon",
)
plt.plot(projection[-4:-2, 0], projection[-4:-2, 1], marker="x", c="blue")
plt.plot(projection[-2:, 0], projection[-2:, 1], marker="x", c="green")
plt.legend()
plt.title("UMAP Projection of Molecular Descriptors")
plt.savefig("umap_projection.png", dpi=300)

breakpoint()


desc_0 = smi_to_desc("C=CC=C")
desc_1 = smi_to_desc("CCCC")
desc_2 = smi_to_desc("C(=O)CCC")
desc_3 = smi_to_desc("C(=O)=CC=C")
