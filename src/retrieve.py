from qdrant_client.models import SparseVector
from .config import combo_collection
from .embedders import get_dense, get_sparse, get_late


def _hits(results):
    out = []
    for rank, r in enumerate(results, 1):
        p = r.payload or {}
        out.append(
            {
                "point_id": r.id,
                "chunk_id": p.get("chunk_id"),
                "doc_id": p.get("doc_id"),
                "source": p.get("source"),
                "text": p.get("text"),
                "score": r.score,
                "rank": rank,
            }
        )
    return out


def retrieve_combo(client, dense_id, sparse_id, late_id, question, top_k=8):
    name = combo_collection(dense_id, sparse_id, late_id)

    dense_vec = get_dense(dense_id).encode([question], normalize_embeddings=True)[0]
    dense_hits = _hits(
        client.query_points(collection_name=name, query=dense_vec.tolist(), using="dense", limit=top_k).points
    )

    sv = next(get_sparse(sparse_id).embed([question]))
    sparse_query = SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist())
    sparse_hits = _hits(
        client.query_points(collection_name=name, query=sparse_query, using="sparse", limit=top_k).points
    )

    late_vec = next(get_late(late_id).embed([question]))
    late_hits = _hits(
        client.query_points(collection_name=name, query=late_vec.tolist(), using="late", limit=top_k).points
    )

    return dense_hits, sparse_hits, late_hits


def rrf(result_lists, k=60, top_k=8):
    scores = {}
    records = {}
    for results in result_lists:
        for rank, item in enumerate(results, 1):
            key = str(item["point_id"])
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            records[key] = item
    ranked = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [
        dict(records[key], fused_rank=rank, rrf_score=scores[key])
        for rank, key in enumerate(ranked, 1)
    ]
