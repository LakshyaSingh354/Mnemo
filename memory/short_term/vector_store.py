import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

class VectorMemoryStore:
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2'):
        self.embedder = SentenceTransformer(embedding_model_name)
        self.index = faiss.IndexFlatL2(384)  # 384 is dim for MiniLM
        self.messages = []  # Stores actual strings
        self.embeddings = []  # Store to enable saving/reloading

    def add_message(self, message: str):
        embedding = self.embedder.encode([message])[0]
        self.index.add(np.array([embedding]).astype('float32'))
        self.messages.append(message)
        self.embeddings.append(embedding)

    def get_relevant_memories(self, query: str, k=5):
        if len(self.messages) == 0:
            return []

        query_vec = self.embedder.encode([query]).astype('float32')
        D, I = self.index.search(query_vec, k)
        return [self.messages[i] for i in I[0] if i < len(self.messages)]
    

if __name__ == "__main__":
    store = VectorMemoryStore()

    store.add_message("I love Christopher Nolan movies.")
    store.add_message("Interstellar is a sci-fi classic and my favorite Christophar Nolan movie.")
    store.add_message("My favorite actor is Matthew McConaughey.")
    store.add_message("Quantum physics is fascinating.")
    store.add_message("Martin Scorsese is a great director too.")
    store.add_message("Goodfellas is a classic film.")
    store.add_message("I enjoy watching movies with complex plots.")
    store.add_message("The Dark Knight is a masterpiece.")
    store.add_message("I think Inception is a mind-bending film.")
    store.add_message("I love the soundtrack of Interstellar.")
    store.add_message("The cinematography in Interstellar is stunning.")
    store.add_message("Nolan is the director of Interstellar.")

    query = "Do you know who directed Interstellar?"
    memories = store.get_relevant_memories(query)
    
    print("Relevant memories:")
    for m in memories:
        print("-", m)