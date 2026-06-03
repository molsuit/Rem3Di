"""Vector-retrieval evaluation: build a vector store from a trained model and a
molecular dataset, then probe it.

The shared object is :class:`~threedscriptors.evaluation.retrieval.vector_store.VectorStore`
(embeddings + aligned SMILES/ids + a swappable nearest-neighbor index). Two
independent, yaml-toggleable tasks consume it:

* ``tanimoto_similarity`` — does embedding-space proximity recover chemical
  (fingerprint) similarity? (within-kNN vs global baseline, enrichment, rank
  correlation, neighborhood overlap@k)
* ``nearest_molecule`` — given a query (existing entry, raw vector, or a new
  SMILES embedded through the model), retrieve the closest dataset molecules.
"""
