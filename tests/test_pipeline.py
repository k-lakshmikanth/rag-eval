from src.config import load_config, combo_collection
from src.qdrant_io import combos
from src.retrieve import rrf


def test_config_loads():
    cfg = load_config()
    assert len(cfg["dense"]) == 3
    assert len(cfg["sparse"]) == 3
    assert len(cfg["late"]) == 2
    assert len(combos(cfg)) == 18
    name = combo_collection("jina-embeddings-v4", "Qdrant/bm25", "colbert-ir/colbertv2.0")
    assert name == "jina_embeddings_v4__qdrant_bm25__colbert_ir_colbertv2_0"


def test_rrf_fuses_and_ranks():
    a = [{"point_id": 1, "text": "x"}, {"point_id": 2, "text": "y"}]
    b = [{"point_id": 2, "text": "y"}, {"point_id": 3, "text": "z"}]
    fused = rrf([a, b], k=60, top_k=2)
    assert fused[0]["point_id"] == 2
    assert len(fused) == 2
    assert fused[0]["fused_rank"] == 1


if __name__ == "__main__":
    test_config_loads()
    test_rrf_fuses_and_ranks()
    print("ok")
