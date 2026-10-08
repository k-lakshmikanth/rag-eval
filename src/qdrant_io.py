from functools import lru_cache
from itertools import product
from qdrant_client.models import (
    VectorParams,
    Distance,
    SparseVectorParams,
    SparseVector,
    MultiVectorConfig,
    MultiVectorComparator,
    PointStruct,
)
from .config import combo_collection
from .embedders import get_dense, get_sparse, get_late

PAYLOAD_COLS = ["chunk_id", "doc_id", "source", "text", "metadata"]


def _payload(row):
    return {c: row[c] for c in PAYLOAD_COLS if c in row}


@lru_cache(maxsize=None)
def _dense_dim(model_id):
    return get_dense(model_id).get_sentence_embedding_dimension()


@lru_cache(maxsize=None)
def _late_dim(model_id):
    return len(next(get_late(model_id).embed(["probe"]))[0])


def combos(cfg):
    return list(product(cfg["dense"], cfg["sparse"], cfg["late"]))


def create_collections(client, cfg):
    for dense_id, sparse_id, late_id in combos(cfg):
        client.recreate_collection(
            collection_name=combo_collection(dense_id, sparse_id, late_id),
            vectors_config={
                "dense": VectorParams(size=_dense_dim(dense_id), distance=Distance.COSINE),
                "late": VectorParams(
                    size=_late_dim(late_id),
                    distance=Distance.COSINE,
                    multivector_config=MultiVectorConfig(
                        comparator=MultiVectorComparator.MAX_SIM
                    ),
                ),
            },
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )


def precompute_vectors(cfg, texts):
    dense_vecs = {m: get_dense(m).encode(texts, normalize_embeddings=True) for m in cfg["dense"]}
    sparse_vecs = {m: list(get_sparse(m).embed(texts)) for m in cfg["sparse"]}
    late_vecs = {m: list(get_late(m).embed(texts)) for m in cfg["late"]}
    return dense_vecs, sparse_vecs, late_vecs


def upsert_combo(client, dense_id, sparse_id, late_id, df, dense_vecs, sparse_vecs, late_vecs):
    dv, sv, lv = dense_vecs[dense_id], sparse_vecs[sparse_id], late_vecs[late_id]
    points = [
        PointStruct(
            id=i,
            vector={
                "dense": dv[i].tolist(),
                "sparse": SparseVector(indices=sv[i].indices.tolist(), values=sv[i].values.tolist()),
                "late": lv[i].tolist(),
            },
            payload=_payload(row),
        )
        for i, row in enumerate(df.to_dict("records"))
    ]
    client.upsert(collection_name=combo_collection(dense_id, sparse_id, late_id), points=points)


def ingest_all(client, cfg, df):
    dense_vecs, sparse_vecs, late_vecs = precompute_vectors(cfg, df["text"].tolist())
    for dense_id, sparse_id, late_id in combos(cfg):
        upsert_combo(client, dense_id, sparse_id, late_id, df, dense_vecs, sparse_vecs, late_vecs)
