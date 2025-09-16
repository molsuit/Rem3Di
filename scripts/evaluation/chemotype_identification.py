import json
import os
from collections import defaultdict

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# clustering + viz
import umap

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

# stats
from scipy.stats import fisher_exact, ks_2samp
from statsmodels.stats.multitest import multipletests
from threedscriptors.data_handling.pipelines import reload_dataset_pipeline

from threedscriptors.evaluation.clustering import (
    UMAPCalculator,
)
from threedscriptors.evaluation.evaluation_utils import (
    evaluate_molecular_descriptor_on_dataset,
)
from threedscriptors.model.model_builder import ModelBuilder

model_directory = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/training_runs/209-2025_08_12_15_25_09-FixedPCQM"


model = ModelBuilder.from_directory(model_directory).build_remedi_model()

dataset_directory = (
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/qm9full"
)


dataset = reload_dataset_pipeline(dataset_directory).build()

clustering_calculator = UMAPCalculator()

descriptors = evaluate_molecular_descriptor_on_dataset(model, dataset)
smiles = dataset.smiles_list


# --- 0) Imports ---------------------------------------------------------------

# clustering + viz

# stats

# RDKit

# --- 1) I/O setup -------------------------------------------------------------
OUTDIR = "qm9_umap_audit"
os.makedirs(OUTDIR, exist_ok=True)

# Expect these exist from your snippet:
# descriptors: np.ndarray (n x d)
# smiles: list[str] length n
assert len(smiles) == descriptors.shape[0]

# --- 2) Molecules + quick filters --------------------------------------------
mols = [Chem.MolFromSmiles(s) for s in smiles]
valid = np.array([m is not None for m in mols])
if not valid.all():
    print(f"Dropping {np.sum(~valid)} invalid SMILES")
mols = [m for m,ok in zip(mols,valid, strict=False) if ok]
X = descriptors[valid]
SMI = [s for s,ok in zip(smiles,valid, strict=False) if ok]

# --- 3) UMAP for visualization only ------------------------------------------
reducer = umap.UMAP(
    n_neighbors=15, min_dist=0.25, metric="cosine",
    n_components=2, init="spectral"
)
XY = reducer.fit_transform(X)

# --- 4) HDBSCAN clustering (on embeddings, not 2D) ---------------------------
clust = hdbscan.HDBSCAN(
    min_cluster_size=200,    # tune: ~0.5–1% of dataset often good for QM9
    min_samples=20,
    metric="cosine"
).fit(X)
labels = clust.labels_
print("Clusters (excluding noise):", sorted(set(labels) - {-1}))
print("Noise points:", int(np.sum(labels==-1)))

# --- 5) Physchem descriptors (compact set) -----------------------------------
def physchem(m):
    return dict(
        MW=Descriptors.MolWt(m),
        logP=Descriptors.MolLogP(m),
        TPSA=rdMolDescriptors.CalcTPSA(m),
        HBA=rdMolDescriptors.CalcNumHBA(m),
        HBD=rdMolDescriptors.CalcNumHBD(m),
        RotB=rdMolDescriptors.CalcNumRotatableBonds(m),
        Ring=rdMolDescriptors.CalcNumRings(m),
        AromR=rdMolDescriptors.CalcNumAromaticRings(m),
        N=sum(1 for a in m.GetAtoms() if a.GetSymbol()=="N"),
        O=sum(1 for a in m.GetAtoms() if a.GetSymbol()=="O"),
        F=sum(1 for a in m.GetAtoms() if a.GetSymbol()=="F"),
        Charge=Chem.GetFormalCharge(m),
    )

PC = pd.DataFrame([physchem(m) for m in mols])

# --- 6) Scaffolds -------------------------------------------------------------
def murcko_smiles(m):
    core = MurckoScaffold.GetScaffoldForMol(m)
    return Chem.MolToSmiles(core) if core is not None else ""
scaff = [murcko_smiles(m) for m in mols]

# --- 7) SMARTS panel (QM9-flavored) ------------------------------------------
SMARTS = {
    "phenyl": "c1ccccc1",
    "pyridine": "n1ccccc1",
    "furan/pyrrole/oxazole-ish": "[o,n]1cccc1",
    "alcohol": "[CX4;!$(C=O)][OX2H]",
    "phenol": "c[OX2H]",
    "ether": "[OD2]([#6])[#6]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "carboxylic_acid": "C(=O)[OX2H1]",
    "ester": "C(=O)O[#6]",
    "amide": "C(=O)N",
    "nitrile": "C#N",
    "imine": "C=N",
    "alkene": "C=C",
    "alkyne": "C#C",
    "small_ring": "[R3,R4]",
    "fluorinated": "[F]"
}
SMARTS_PAT = {k: Chem.MolFromSmarts(v) for k,v in SMARTS.items()}
FG = {k: np.array([int(m.HasSubstructMatch(p)) for m in mols], dtype=int)
      for k,p in SMARTS_PAT.items()}

# --- 8) ECFP bit table (optional but useful) ---------------------------------
RADIUS, N_BITS = 2, 2048
bit_hits = defaultdict(list)   # bit -> indices where present
bit_per_mol = []
bitInfo_per_mol = []
for i,m in enumerate(mols):
    bitInfo = {}
    bv = AllChem.GetMorganFingerprintAsBitVect(m, RADIUS, nBits=N_BITS, bitInfo=bitInfo)
    onbits = list(bv.GetOnBits())
    bit_per_mol.append(onbits)
    bitInfo_per_mol.append(bitInfo)
    for b in onbits:
        bit_hits[b].append(i)

# --- 9) Assemble master frame -------------------------------------------------
DF = pd.DataFrame({
    "smiles": SMI,
    "cluster": labels,
    "x": XY[:,0], "y": XY[:,1],
    "scaffold": scaff,
    **{k: FG[k] for k in FG}
})
DF = pd.concat([DF, PC], axis=1)
DF.to_csv(f"{OUTDIR}/master_table.csv", index=False)

# --- 10) Per-cluster summaries ------------------------------------------------
def fisher_enrichment(mask_in, mask_fg):
    # mask_in: bool in-cluster; mask_fg: bool has-feature
    a = np.sum(mask_in & mask_fg)
    b = np.sum(mask_in & ~mask_fg)
    c = np.sum(~mask_in & mask_fg)
    d = np.sum(~mask_in & ~mask_fg)
    OR = (a*d + 1e-9)/((b*c) + 1e-9)
    p = fisher_exact([[a,b],[c,d]], alternative="greater")[1]
    return OR, p, int(a), int(b), int(c), int(d)

cluster_reports = {}
for c in sorted(set(labels) - {-1}):
    mask = DF.cluster.values == c
    sub = DF[mask]; rest = DF[~mask]
    # physchem shifts
    stats=[]
    for col in ["MW","logP","TPSA","HBA","HBD","RotB","Ring","AromR","N","O","F"]:
        d = (sub[col].mean() - rest[col].mean()) / (rest[col].std() + 1e-9)  # Cohen-ish
        p = ks_2samp(sub[col], rest[col]).pvalue
        stats.append((col, float(sub[col].median()), float(rest[col].median()), float(d), float(p)))
    physchem_tbl = pd.DataFrame(stats, columns=["feat","median_cluster","median_bg","cohen_d","p_ks"])
    physchem_tbl["q_ks"] = multipletests(physchem_tbl.p_ks, method="fdr_bh")[1]
    physchem_tbl.sort_values("q_ks", inplace=True)

    # scaffold top-N
    top_scaff = sub.scaffold.value_counts().head(12)

    # SMARTS enrichment
    rows=[]
    for name in SMARTS:
        OR,p,a,b,c_,d_ = fisher_enrichment(mask, DF[name].values.astype(bool))
        rows.append((name, OR, p, a, b, c_, d_))
    enrich = pd.DataFrame(rows, columns=["feature","odds_ratio","p","a","b","c","d"])
    enrich["q"] = multipletests(enrich.p, method="fdr_bh")[1]
    enrich.sort_values(["q","odds_ratio"], ascending=[True,False], inplace=True)

    # ECFP bit enrichment (top few)
    bit_rows=[]
    for bit, idxs in bit_hits.items():
        has = np.zeros(len(DF), dtype=bool); has[np.array(idxs)] = True
        OR,p,_,_,_,_ = fisher_enrichment(mask, has)
        bit_rows.append((bit, OR, p))
    bit_enrich = pd.DataFrame(bit_rows, columns=["bit","odds_ratio","p"])
    bit_enrich["q"] = multipletests(bit_enrich.p, method="fdr_bh")[1]
    bit_enrich = bit_enrich.sort_values(["q","odds_ratio"], ascending=[True,False]).head(20)

    # simple “medoid”: nearest to cluster centroid in embedding (cosine)
    Xc = X[mask]
    centroid = Xc.mean(0)
    # cosine distance ~ 1 - cosine similarity (normalize)
    def normalize(A):
        n = np.linalg.norm(A, axis=1, keepdims=True) + 1e-9
        return A / n
    Z = normalize(Xc); zc = centroid / (np.linalg.norm(centroid)+1e-9)
    cosdist = 1 - (Z @ zc)
    medoid_local_idx = int(np.argmin(cosdist))
    medoid_global_idx = np.where(mask)[0][medoid_local_idx]
    medoid_smiles = DF.loc[medoid_global_idx, "smiles"]

    # save CSVs
    physchem_tbl.to_csv(f"{OUTDIR}/cluster_{c:02d}_physchem.csv", index=False)
    top_scaff.to_csv(f"{OUTDIR}/cluster_{c:02d}_top_scaffolds.csv")
    enrich.to_csv(f"{OUTDIR}/cluster_{c:02d}_smarts_enrichment.csv", index=False)
    bit_enrich.to_csv(f"{OUTDIR}/cluster_{c:02d}_ecfp_enrichment.csv", index=False)

    cluster_reports[c] = dict(
        size=int(mask.sum()),
        medoid=medoid_smiles,
        top_scaffold=list(top_scaff.index[:3]),
        top_fg=enrich.query("q<0.05").head(5)[["feature","odds_ratio","q"]].to_dict("records")
    )

with open(f"{OUTDIR}/cluster_reports.json","w") as f:
    json.dump(cluster_reports, f, indent=2)
print("Wrote per-cluster reports to", OUTDIR)

# --- 11) Plots ---------------------------------------------------------------
# UMAP colored by cluster
plt.figure(figsize=(5,5), dpi=200)
plt.scatter(DF.x, DF.y, s=1, c=DF.cluster, cmap="tab20", linewidths=0)
plt.axis("off")
plt.title("UMAP (colored by HDBSCAN cluster)")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/umap_clusters.png", dpi=300)

# Property overlays (a few informative ones)
for col in ["logP","TPSA","AromR","F"]:
    plt.figure(figsize=(5,5), dpi=200)
    plt.scatter(DF.x, DF.y, s=1, c=DF[col], cmap="viridis", linewidths=0)
    plt.axis("off")
    plt.title(f"UMAP colored by {col}")
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/umap_{col}.png", dpi=300)

# Quick table: cluster sizes & medoids
sizes = DF.groupby("cluster").size().rename("size").sort_values(ascending=False)
tbl = pd.DataFrame({"size": sizes}).reset_index()
tbl = tbl[tbl.cluster!=-1]
tbl["medoid_smiles"] = tbl["cluster"].map(lambda c: cluster_reports[c]["medoid"])
tbl.to_csv(f"{OUTDIR}/cluster_sizes_medoids.csv", index=False)
tbl.head(10)
