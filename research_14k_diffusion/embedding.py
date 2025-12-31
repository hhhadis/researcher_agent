from typing import List
import numpy as np

def _hash_vec(tokens: List[str], dim: int = 384) -> np.ndarray:
    rng = np.random.RandomState(42)
    v = np.zeros(dim, dtype=np.float32)
    for t in tokens:
        h = abs(hash(t))
        idx = h % dim
        v[idx] += 1.0
    if np.linalg.norm(v) > 0:
        v = v / np.linalg.norm(v)
    noise = rng.normal(0, 1e-4, size=dim).astype(np.float32)
    return v + noise

def encode(texts: List[str]) -> List[List[float]]:
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embs = model.encode(texts, normalize_embeddings=True)
        return [list(map(float, e)) for e in embs]
    except Exception:
        vecs = []
        print(1)
        for t in texts:
            toks = t.split()
            vecs.append(list(map(float, _hash_vec(toks))))
        return vecs

def aggregate(paragraphs: List[str]) -> List[float]:
    if not paragraphs:
        return list(map(float, _hash_vec([""], 384)))
    embs = encode(paragraphs)
    x = np.mean(np.array(embs, dtype=np.float32), axis=0)
    if np.linalg.norm(x) > 0:
        x = x / np.linalg.norm(x)
    return list(map(float, x))

