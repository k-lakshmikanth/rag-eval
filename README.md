# 18-Combination Hybrid RAG Benchmark

## Scope

This benchmark tests ONLY:

> Dense + Sparse + Late Interaction

### Models

Dense (3):
- jina-embeddings-v5-omni-small
- nomic-ai/nomic-embed-text-v2-moe
- jina-embeddings-v4

Sparse (3):
- BM25
- SPLADE++
- miniCOIL

Late interaction (2):
- ColBERTv2
- Answer.AI ColBERT Small

### Experiment matrix

3 Dense × 3 Sparse × 2 Late = **18 retrieval configurations**

If using N LLMs:
**18 × N end-to-end configurations**

With 3 LLMs:
**54 configurations**

## Architecture

Config-driven: `config.yaml` lists the dense, sparse, and late-interaction
model candidates. The Cartesian product of those three lists is the
experiment matrix — add or remove a model in the yaml and the matrix, the
collections, and the pipeline all follow.

Each combination gets its own Qdrant collection, named
`<dense_model>__<sparse_model>__<late_model>`, holding three named vectors
(`dense`, `sparse`, `late`) over the same corpus.

Each query:
1. embeds the question with the combo's dense/sparse/late models
2. retrieves from each named vector in the combo's collection
3. fuses the three result lists with RRF
4. sends the fused context to the LLM
5. stores answer + context + latency + usage metadata

DeepEval then evaluates the persisted responses.

## Scripts

`python scripts/create_collections.py` — creates all combo collections from
`config.yaml`. Re-run after editing the yaml to pick up new/removed models.

## Notebooks

00_qdrant_ngrok.ipynb
01_collection_creator.ipynb
02_ingestion.ipynb
03_retrieval_and_generation.ipynb
04_deepeval_report.ipynb

## Security

Qdrant is bound to localhost and exposed through ngrok only when needed.
Use Qdrant API-key authentication. Keep QDRANT_API_KEY and NGROK_AUTHTOKEN
outside the notebooks and source control.
