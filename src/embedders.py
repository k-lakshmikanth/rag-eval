from functools import lru_cache


@lru_cache(maxsize=None)
def get_dense(model_id):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_id, trust_remote_code=True)


@lru_cache(maxsize=None)
def get_sparse(model_id):
    from fastembed import SparseTextEmbedding
    return SparseTextEmbedding(model_name=model_id)


@lru_cache(maxsize=None)
def get_late(model_id):
    from fastembed import LateInteractionTextEmbedding
    return LateInteractionTextEmbedding(model_name=model_id)
