import logging
from typing import List, Dict
import time
from dataclasses import dataclass

from memory.short_term.vector_store import MemoryEntry, VectorMemoryStore

logger = logging.getLogger('MyLogger')

@dataclass
class Summary:
    content: str
    timestamp: float
    source_memories: List[MemoryEntry]
    embedding: List[float]
    tags: List[str]

class SummarizationMemory:
    def __init__(self, vector_store: VectorMemoryStore, summary_threshold: float = 0.8):
        self.vector_store = vector_store
        self.summary_threshold = summary_threshold
        self.summaries: List[Summary] = []
        
    def should_summarize(self, memories: List[MemoryEntry]) -> bool:
        """Determine if memories should be summarized based on similarity and time."""
        if len(memories) < 3:  # Need minimum memories to summarize
            return False
            
        # Check if memories are temporally close
        time_range = memories[-1].timestamp - memories[0].timestamp
        if time_range > 3600:  # More than 1 hour
            return False
            
        # Check if memories are semantically similar
        embeddings = [memory.embedding for memory in memories]
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = self.vector_store.score_novelty(embeddings[i], [embeddings[j]], threshold=1.0)
                similarities.append(1 - sim)  # Convert novelty to similarity
                
        avg_similarity = sum(similarities) / len(similarities)
        return avg_similarity > self.summary_threshold

    def create_summary(self, memories: List[MemoryEntry]) -> Summary:
        """Create a summary from a group of related memories."""
        # Combine memory contents
        combined_content = " ".join(m.content for m in memories)
        
        # Extract common tags
        all_tags = [tag for m in memories for tag in m.tags]
        common_tags = list(set(all_tags))
        
        # Create summary
        summary = Summary(
            content=combined_content,
            timestamp=time.time(),
            source_memories=memories,
            embedding=memories[0].embedding,  # Use first memory's embedding as base
            tags=common_tags
        )
        
        return summary

    def process_memories(self):
        """Process memories and create summaries when appropriate."""
        memories = self.vector_store.dump_memory()
        if not memories:
            return
            
        # Group memories by time windows
        current_window = []
        for memory in memories:
            if not current_window:
                current_window.append(memory)
                continue
                
            # Check if memory belongs to current window
            if memory.timestamp - current_window[0].timestamp <= 3600:  # 1 hour window
                current_window.append(memory)
            else:
                # Process current window
                if self.should_summarize(current_window):
                    summary = self.create_summary(current_window)
                    self.summaries.append(summary)
                    logger.info(f"Created new summary with {len(current_window)} memories")
                
                # Start new window
                current_window = [memory]
        
        # Process final window
        if self.should_summarize(current_window):
            summary = self.create_summary(current_window)
            self.summaries.append(summary)
            logger.info(f"Created new summary with {len(current_window)} memories")

    def get_relevant_summaries(self, query: str, k: int = 3) -> List[Summary]:
        """Retrieve relevant summaries for a query."""
        if not self.summaries:
            return []
            
        # Use vector store to find similar summaries
        query_vec = self.vector_store.embedder.encode([query])[0]
        summary_embeddings = [s.embedding for s in self.summaries]
        
        similarities = []
        for emb in summary_embeddings:
            sim = self.vector_store.score_novelty(query_vec, [emb], threshold=1.0)
            similarities.append(1 - sim)  # Convert novelty to similarity
            
        # Get top k summaries
        top_indices = sorted(range(len(similarities)), 
                        key=lambda i: similarities[i], 
                        reverse=True)[:k]
        
        return [self.summaries[i] for i in top_indices]

    def print_summaries(self):
        """Print all summaries for debugging."""
        print("\n===== MEMORY SUMMARIES =====")
        for i, summary in enumerate(self.summaries):
            print(f"\nSummary {i+1}:")
            print(f"Content: {summary.content[:200]}...")
            print(f"Created: {time.ctime(summary.timestamp)}")
            print(f"Source memories: {len(summary.source_memories)}")
            print(f"Tags: {summary.tags}")
