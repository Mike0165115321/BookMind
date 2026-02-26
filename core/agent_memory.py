"""
Agent Memory — Working memory for the Agentic RAG loop.

Tracks the state across multiple search iterations:
  - Which sub-queries have been searched
  - What chunks have been gathered (deduplicated)
  - Search history with scores per iteration

This is NOT long-term memory (cross-session).
This is working memory — alive only during a single agentic run.

Deduplication Strategy:
  Chunks from different sub-queries may overlap (same book passage).
  We use normalized text fingerprinting to detect and skip duplicates,
  keeping only the highest-scoring version.
"""
from dataclasses import dataclass, field


# ──────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────
@dataclass
class SearchRecord:
    """Record of a single search execution."""
    iteration: int
    query: str
    results: list[tuple[str, float]]  # [(text, score), ...]
    num_new_chunks: int = 0           # How many new (non-duplicate) chunks added


@dataclass
class GatheredChunk:
    """A unique chunk stored in memory."""
    text: str
    score: float
    source_query: str
    iteration: int


class AgentMemory:
    """
    Working memory for an agentic RAG session.

    Responsibilities:
      1. Store search results from each iteration
      2. Deduplicate chunks across iterations
      3. Provide all gathered chunks for final generation
      4. Track search history for UI/debugging
    """

    def __init__(self, original_query: str):
        self.original_query = original_query
        self.search_history: list[SearchRecord] = []
        self._chunks: dict[str, GatheredChunk] = {}  # fingerprint → chunk
        self._searched_queries: set[str] = set()

    # ──────────────────────────────────────────
    # Core Operations
    # ──────────────────────────────────────────
    def add_search_results(
        self,
        query: str,
        results: list[tuple[str, float]],
        iteration: int,
    ) -> int:
        """
        Add search results from one iteration.

        Deduplicates against previously gathered chunks.
        Returns the number of NEW (non-duplicate) chunks added.
        """
        new_count = 0

        for text, score in results:
            fp = self._fingerprint(text)

            if fp in self._chunks:
                # Chunk already exists — keep higher score
                if score > self._chunks[fp].score:
                    self._chunks[fp].score = score
                    self._chunks[fp].source_query = query
                    self._chunks[fp].iteration = iteration
            else:
                # New chunk
                self._chunks[fp] = GatheredChunk(
                    text=text,
                    score=score,
                    source_query=query,
                    iteration=iteration,
                )
                new_count += 1

        # Record this search
        record = SearchRecord(
            iteration=iteration,
            query=query,
            results=results,
            num_new_chunks=new_count,
        )
        self.search_history.append(record)
        self._searched_queries.add(query.lower().strip())

        return new_count

    def get_all_chunks(self) -> list[tuple[str, float]]:
        """
        Get all gathered chunks, sorted by score (highest first).

        Returns:
            List of (text, score) tuples — same format as RAGSearcher.search()
        """
        chunks = sorted(
            self._chunks.values(),
            key=lambda c: c.score,
            reverse=True,
        )
        return [(c.text, c.score) for c in chunks]

    def get_balanced_chunks(self, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Get chunks with BALANCED representation from each source query.

        Instead of global top-K (which may favor one source),
        this does round-robin selection: pick the best chunk from each
        source query in turn, until we have top_k chunks.

        This ensures that if we searched 3 sub-queries, all 3 are
        represented in the final set sent to the LLM.

        Args:
            top_k: Maximum number of chunks to return

        Returns:
            List of (text, score) tuples with balanced source representation
        """
        if not self._chunks:
            return []

        # Group chunks by source_query, sorted by score within each group
        groups: dict[str, list[GatheredChunk]] = {}
        for chunk in self._chunks.values():
            key = chunk.source_query
            if key not in groups:
                groups[key] = []
            groups[key].append(chunk)

        # Sort each group by score (highest first)
        for key in groups:
            groups[key].sort(key=lambda c: c.score, reverse=True)

        # Round-robin selection
        result = []
        group_keys = list(groups.keys())
        pointers = {key: 0 for key in group_keys}

        while len(result) < top_k:
            added_this_round = False
            for key in group_keys:
                if pointers[key] < len(groups[key]):
                    chunk = groups[key][pointers[key]]
                    result.append((chunk.text, chunk.score))
                    pointers[key] += 1
                    added_this_round = True
                    if len(result) >= top_k:
                        break

            if not added_this_round:
                break  # All groups exhausted

        return result

    def get_context_summary(self) -> str:
        """
        Build a summary of what has been gathered so far.

        Used by the Evaluator to decide if more searching is needed.
        Returns a concise text listing gathered information.
        """
        if not self._chunks:
            return "ยังไม่มีข้อมูลที่ค้นหาได้"

        lines = [f"ข้อมูลที่ค้นหาได้แล้ว ({len(self._chunks)} chunks):"]

        for i, chunk in enumerate(self.get_all_chunks()[:10], 1):
            text, score = chunk
            snippet = text[:150].replace("\n", " ")
            lines.append(f"  [{i}] ({score:.2f}) {snippet}...")

        return "\n".join(lines)

    # ──────────────────────────────────────────
    # Query Tracking
    # ──────────────────────────────────────────
    def has_searched(self, query: str) -> bool:
        """Check if a similar query has already been searched."""
        return query.lower().strip() in self._searched_queries

    @property
    def total_chunks(self) -> int:
        """Total number of unique chunks gathered."""
        return len(self._chunks)

    @property
    def total_iterations(self) -> int:
        """Number of search iterations completed."""
        return len(self.search_history)

    # ──────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────
    @staticmethod
    def _fingerprint(text: str) -> str:
        """
        Create a normalized fingerprint for deduplication.

        Strategy: take first 200 chars, normalize whitespace, lowercase.
        This catches exact and near-duplicates without expensive comparison.
        """
        normalized = " ".join(text[:200].lower().split())
        return normalized
"""
"""
