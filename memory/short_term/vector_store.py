import logging
import re
import faiss
import numpy as np
import time
from typing import Dict, List
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from colorlog import ColoredFormatter


formatter = ColoredFormatter(
    "%(log_color)s[%(levelname)s] %(message)s",
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'bold_red',
    },
    secondary_log_colors={},
    style='%'
)

handler = logging.StreamHandler()
handler.setFormatter(formatter)

logger = logging.getLogger('MyLogger')
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

@dataclass
class MemoryEntry:
    content: str
    embedding: List[float]
    timestamp: float
    tags: List[str]
    source: str
    scores: Dict[str, float] = field(default_factory=dict)

class VectorMemoryStore:
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2', dim=384, max_size=100):
        self.embedder = SentenceTransformer(embedding_model_name)
        self.dim = dim
        self.max_size = max_size

        self.index = faiss.IndexFlatL2(dim)
        self.entries: List[MemoryEntry] = []
        self.embeddings = []

    def score_importance(self, message: str) -> float:
        # Heuristic scoring: personal language + length + info density
        score = 0.0
        if any(p in message.lower() for p in ["i ", "my ", "me ", "mine "]):
            score += 0.4
        if len(message) > 100:
            score += 0.3
        if len(re.findall(r'\b[A-Z][a-z]+\b', message)) >= 3:  # Named entities (approx)
            score += 0.3
        return min(score, 1.0)

    def score_novelty(self, new_emb: np.ndarray, recent_embs: List[np.ndarray], threshold=0.95) -> float:
        if not recent_embs:
            return 1.0  # fully novel at the start
        sims = cosine_similarity([new_emb], recent_embs)[0]
        max_sim = max(sims)
        novelty = 1 - min(max_sim, threshold)
        return round(novelty, 3)

    def compute_recency_score(self, entry_time, current_time, half_life=600):
        """
        Returns a score between 0 and 1 based on how recent the memory is.
        Uses exponential decay: 0.5^((t_now - t_entry)/half_life)
        """
        age = current_time - entry_time
        decay = 0.5 ** (age / half_life)
        return round(decay, 4)

    def is_duplicate(self, new_embedding, threshold=0.95):
        for emb in self.embeddings:
            sim = cosine_similarity([new_embedding], [emb])[0][0]
            if sim >= threshold:
                return True
        return False

    def add_message(self, message: str, source: str = "chat"):
        embedding = self.embedder.encode([message])[0]

        if self.is_duplicate(new_embedding=embedding):
            logger.warning("Duplicate message, skipping insertion.")
            return

        importance = self.score_importance(message)
        novelty = self.score_novelty(embedding, self.embeddings)

        tags = ['dummy_tag']

        entry = MemoryEntry(
            content=message,
            embedding=embedding.tolist(),
            timestamp=time.time(),
            tags=tags,
            source=source,
            scores={
                "importance": importance,
                "novelty": novelty,
                "recency": 1.0  # always 1.0 at insertion; decays at query-time
            }
        )

        self.entries.append(entry)
        self.embeddings.append(np.array(embedding).astype('float32'))
        self.index.add(np.array([embedding]).astype('float32'))

        if len(self.entries) > self.max_size:
            self._prune()

    def get_relevant_memories(self, query: str, k=5, threshold=0.5, score_weights={"sim": 0.4, "importance": 0.2, "novelty": 0.2, "recency": 0.2}):
        if len(self.entries) == 0:
            return []

        query_vec = self.embedder.encode([query]).astype('float32')
        D, I = self.index.search(query_vec, k)

        query_vec_np = query_vec.reshape(1, -1)
        current_time = time.time()


        scored_results = []
        for idx in I[0]:
            if idx >= len(self.entries):
                continue

            entry = self.entries[idx]
            sim = cosine_similarity(query_vec_np, [self.embeddings[idx]])[0][0]
            if sim < threshold:
                continue

            recency = self.compute_recency_score(entry.timestamp, current_time)
            total_score = (
                score_weights["sim"] * sim +
                score_weights["importance"] * entry.scores.get("importance", 0) +
                score_weights["novelty"] * entry.scores.get("novelty", 0) +
                score_weights["recency"] * recency
            )

            scored_results.append((entry, total_score))

        scored_results.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, score in scored_results]

    def _prune(self):
        self.entries = self.entries[-self.max_size:]
        self.embeddings = self.embeddings[-self.max_size:]
        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(np.array(self.embeddings).astype("float32"))

    def dump_memory(self) -> List[MemoryEntry]:
        return self.entries

    def print_memory_state(self):
        print("===== MEMORY STATE =====")
        for i, entry in enumerate(self.entries[-10:]):
            score_str = ", ".join(f"{k}:{v:.2f}" for k, v in entry.scores.items())
            print(f"{i+1}. [{entry.source}] {entry.content[:60]}... (tags: {entry.tags}, scores: {score_str})")
import logging
import re
import faiss
import numpy as np
import time
import os
from typing import Dict, List
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from colorlog import ColoredFormatter


formatter = ColoredFormatter(
    "%(log_color)s[%(levelname)s] %(message)s",
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'bold_red',
    },
    secondary_log_colors={},
    style='%'
)

handler = logging.StreamHandler()
handler.setFormatter(formatter)

logger = logging.getLogger('MyLogger')
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

@dataclass
class MemoryEntry:
    content: str
    embedding: List[float]
    timestamp: float
    tags: List[str]
    source: str
    scores: Dict[str, float] = field(default_factory=dict)

class VectorMemoryStore:
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2', dim=384, max_size=100):
        self.embedder = SentenceTransformer(embedding_model_name)
        self.dim = dim
        self.max_size = max_size

        self.index = faiss.IndexFlatL2(dim)
        self.entries: List[MemoryEntry] = []
        self.embeddings = []

    def score_importance(self, message: str) -> float:
        # Heuristic scoring: personal language + length + info density
        score = 0.0
        if any(p in message.lower() for p in ["i ", "my ", "me ", "mine "]):
            score += 0.4
        if len(message) > 100:
            score += 0.3
        if len(re.findall(r'\b[A-Z][a-z]+\b', message)) >= 3:  # Named entities (approx)
            score += 0.3
        return min(score, 1.0)

    def score_novelty(self, new_emb: np.ndarray, recent_embs: List[np.ndarray], threshold=0.95) -> float:
        if not recent_embs:
            return 1.0  # fully novel at the start
        sims = cosine_similarity([new_emb], recent_embs)[0]
        max_sim = max(sims)
        novelty = 1 - min(max_sim, threshold)
        return round(novelty, 3)

    def compute_recency_score(self, entry_time, current_time, half_life=600):
        """
        Returns a score between 0 and 1 based on how recent the memory is.
        Uses exponential decay: 0.5^((t_now - t_entry)/half_life)
        """
        age = current_time - entry_time
        decay = 0.5 ** (age / half_life)
        return round(decay, 4)

    def is_duplicate(self, new_embedding, threshold=0.95):
        for emb in self.embeddings:
            sim = cosine_similarity([new_embedding], [emb])[0][0]
            if sim >= threshold:
                return True
        return False

    def add_message(self, message: str, source: str = "chat"):
        embedding = self.embedder.encode([message])[0]

        if self.is_duplicate(new_embedding=embedding):
            logger.warning("Duplicate message, skipping insertion.")
            return

        importance = self.score_importance(message)
        novelty = self.score_novelty(embedding, self.embeddings)

        tags = ['dummy_tag']

        entry = MemoryEntry(
            content=message,
            embedding=embedding.tolist(),
            timestamp=time.time(),
            tags=tags,
            source=source,
            scores={
                "importance": importance,
                "novelty": novelty,
                "recency": 1.0  # always 1.0 at insertion; decays at query-time
            }
        )

        self.entries.append(entry)
        self.embeddings.append(np.array(embedding).astype('float32'))
        self.index.add(np.array([embedding]).astype('float32'))

        if len(self.entries) > self.max_size:
            self._prune()

    def get_relevant_memories(self, query: str, k=5, threshold=0.5, score_weights={"sim": 0.4, "importance": 0.2, "novelty": 0.2, "recency": 0.2}):
        if len(self.entries) == 0:
            return []

        query_vec = self.embedder.encode([query]).astype('float32')
        D, I = self.index.search(query_vec, k)

        query_vec_np = query_vec.reshape(1, -1)
        current_time = time.time()


        scored_results = []
        for idx in I[0]:
            if idx >= len(self.entries):
                continue

            entry = self.entries[idx]
            sim = cosine_similarity(query_vec_np, [self.embeddings[idx]])[0][0]
            if sim < threshold:
                continue

            recency = self.compute_recency_score(entry.timestamp, current_time)
            total_score = (
                score_weights["sim"] * sim +
                score_weights["importance"] * entry.scores.get("importance", 0) +
                score_weights["novelty"] * entry.scores.get("novelty", 0) +
                score_weights["recency"] * recency
            )

            scored_results.append((entry, total_score))

        scored_results.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, score in scored_results]

    def _prune(self):
        self.entries = self.entries[-self.max_size:]
        self.embeddings = self.embeddings[-self.max_size:]
        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(np.array(self.embeddings).astype("float32"))

    def dump_memory(self) -> List[MemoryEntry]:
        return self.entries

    def print_memory_state(self):
        print("===== MEMORY STATE =====")
        for i, entry in enumerate(self.entries[-10:]):
            score_str = ", ".join(f"{k}:{v:.2f}" for k, v in entry.scores.items())
            print(f"{i+1}. [{entry.source}] {entry.content[:60]}... (tags: {entry.tags}, scores: {score_str})")