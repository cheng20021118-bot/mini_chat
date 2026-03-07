import os
import json
import time
import hashlib
from typing import List, Tuple, Optional

import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer


@st.cache_resource
def get_st_model():
    # keep your original model choice
    return SentenceTransformer("BAAI/bge-small-zh-v1.5")


def _sha256_texts(texts: List[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\n---\n")
    return h.hexdigest()


def _atomic_write_json(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class VectorStore:
    """VectorStore with disk persistence (storage/vector)."""

    def __init__(
        self,
        documents: List[str],
        storage_dir: str = "storage/vector",
        force_rebuild: bool = False,
        log_prefix: str = "[VectorStore]",
    ):
        self.documents = documents
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

        self.model = get_st_model()

        self.index_path = os.path.join(self.storage_dir, "faiss.index")
        self.emb_path = os.path.join(self.storage_dir, "embeddings.npy")
        self.meta_path = os.path.join(self.storage_dir, "meta.json")

        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.Index] = None

        self._build_or_load(force_rebuild=force_rebuild, log_prefix=log_prefix)

    def _build_or_load(self, force_rebuild: bool, log_prefix: str) -> None:
        t0 = time.perf_counter()
        corpus_fp = _sha256_texts(self.documents)

        if (not force_rebuild) and self._artifacts_exist():
            try:
                meta = self._load_meta()
                if self._meta_matches(meta, corpus_fp):
                    self.index = faiss.read_index(self.index_path)
                    self.embeddings = np.load(self.emb_path).astype("float32")
                    dt = (time.perf_counter() - t0) * 1000
                    print(
                        f"{log_prefix} Loaded existing FAISS index in {dt:.1f} ms "
                        f"(docs={meta.get('num_docs')}, dim={meta.get('dim')})."
                    )
                    return
                else:
                    print(f"{log_prefix} Index exists but meta mismatch -> rebuilding.")
            except Exception as e:
                print(f"{log_prefix} Failed to load existing index ({e}) -> rebuilding.")

        # rebuild
        t1 = time.perf_counter()
        self.embeddings = self.model.encode(
            self.documents,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        dim = int(self.embeddings.shape[1])
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

        self._save_all(corpus_fp=corpus_fp, dim=dim)

        dt_build = (time.perf_counter() - t1) * 1000
        dt_total = (time.perf_counter() - t0) * 1000
        print(
            f"{log_prefix} Built FAISS index in {dt_build:.1f} ms "
            f"(total {dt_total:.1f} ms), saved to {self.storage_dir}."
        )

    def _artifacts_exist(self) -> bool:
        return (
            os.path.exists(self.index_path)
            and os.path.exists(self.emb_path)
            and os.path.exists(self.meta_path)
        )

    def _load_meta(self) -> dict:
        with open(self.meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _meta_matches(self, meta: dict, corpus_fp: str) -> bool:
        try:
            if meta.get("version") != 1:
                return False
            if meta.get("embed_model") != "BAAI/bge-small-zh-v1.5":
                return False
            if int(meta.get("num_docs", -1)) != len(self.documents):
                return False
            if meta.get("corpus_fingerprint") != corpus_fp:
                return False
            return True
        except Exception:
            return False

    def _save_all(self, corpus_fp: str, dim: int) -> None:
        # 1) faiss index
        tmp_index = self.index_path + ".tmp"
        faiss.write_index(self.index, tmp_index)
        os.replace(tmp_index, self.index_path)

        # 2) embeddings.npy
        tmp_emb = self.emb_path + ".tmp"
        np.save(tmp_emb, self.embeddings)
        tmp_emb_npy = tmp_emb if tmp_emb.endswith(".npy") else (tmp_emb + ".npy")
        os.replace(tmp_emb_npy, self.emb_path)

        # 3) meta.json
        meta = {
            "version": 1,
            "embed_model": "BAAI/bge-small-zh-v1.5",
            "dim": dim,
            "normalize_embeddings": True,
            "num_docs": len(self.documents),
            "corpus_fingerprint": corpus_fp,
            "built_at_ms": int(time.time() * 1000),
        }
        _atomic_write_json(self.meta_path, meta)

    def get_embedding(self, text: str) -> np.ndarray:
        return self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")[0]

    def retrieve(self, query: str, top_k: int = 8) -> List[Tuple[float, str]]:
        if self.index is None:
            return []

        q_vec = self.get_embedding(query).reshape(1, -1).astype("float32")
        k = min(top_k, int(self.index.ntotal))
        if k <= 0:
            return []

        scores, indices = self.index.search(q_vec, k)
        results: List[Tuple[float, str]] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if 0 <= idx < len(self.documents):
                results.append((float(score), self.documents[idx]))
        return results


@st.cache_resource
def build_vector_store(docs: tuple, storage_dir: str = "storage/vector"):
    return VectorStore(list(docs), storage_dir=storage_dir)
