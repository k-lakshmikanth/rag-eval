import os
import re
import yaml
from qdrant_client import QdrantClient

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def qdrant_client_from_config(cfg):
    q = cfg["qdrant"]
    url = os.getenv(q["url_env"], q["default_url"])
    api_key = os.getenv(q["api_key_env"]) or None
    return QdrantClient(url=url, api_key=api_key)


def llm_models_from_env(env_var="LLM_MODELS"):
    raw = os.getenv(env_var, "")
    return [x.strip() for x in raw.split(",") if x.strip()]


def slugify(model_id):
    return re.sub(r"[^a-z0-9]+", "_", model_id.lower()).strip("_")


def combo_collection(dense_id, sparse_id, late_id):
    return f"{slugify(dense_id)}__{slugify(sparse_id)}__{slugify(late_id)}"
