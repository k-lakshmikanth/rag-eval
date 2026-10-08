from src.config import load_config, qdrant_client_from_config
from src.qdrant_io import create_collections, combos

if __name__ == "__main__":
    cfg = load_config()
    client = qdrant_client_from_config(cfg)
    create_collections(client, cfg)
    print("Created", len(combos(cfg)), "collections")
    print(client.get_collections())
