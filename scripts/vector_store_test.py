from memory.short_term.vector_store import VectorMemoryStore
import time

def main():
    memory = VectorMemoryStore()

    memory.add_message("I love Interstellar.", tags=["movie", "personal"])
    time.sleep(1)
    memory.add_message("I enjoy cooking Italian food.", tags=["hobby", "food"])
    memory.add_message("Matthew McConaughey is my favorite actor.", tags=["actor", "personal"])
    memory.add_message("Matthew McConaughey is my favorite actor.", tags=["actor", "personal"])
    time.sleep(1)
    memory.add_message("Christopher Nolan is my favorite director.", tags=["director", "personal"])
    memory.add_message("Interstellar is the best sci-fi movie ever made.", tags=["opinion", "movie"])
    memory.add_message("I think Nolan's movies always play with time.", tags=["analysis", "director"])

    # Some random messages
    time.sleep(10)
    memory.add_message("The weather is nice today.", tags=["weather"])
    memory.add_message("I need to buy groceries.", tags=["errands"])
    memory.add_message("I love hiking in the mountains.", tags=["hobby", "outdoors"])
    memory.add_message("I watched a great documentary about space.", tags=["documentary", "space"])
    memory.add_message("I enjoy reading books about history.", tags=["hobby", "books"])
    memory.add_message("I am learning to play the guitar.", tags=["hobby", "music"])
    memory.add_message("I love the sound of rain.", tags=["weather", "nature"])
    time.sleep(10)
    memory.add_message("I am planning a trip to the mountains.", tags=["travel", "hobby"])
    memory.add_message("I enjoy cooking Italian food.", tags=["hobby", "food"])
    time.sleep(10)
    memory.add_message("I love Pizza.", tags=["food", "personal"])

    print("\n\n>>> Relevant memories for: 'Who is my favorite actor?'")
    results = memory.get_relevant_memories("Who is my favorite actor?")
    for i, r in enumerate(results):
        print(f"{i+1}. {r.content}")

    print("\n\n>>> Relevant memories for: 'Tell me something about Interstellar'")
    results = memory.get_relevant_memories("Tell me something about Interstellar")
    for i, r in enumerate(results):
        print(f"{i+1}. {r.content}")

    print("\n\n>>> Relevant memories for: 'What is my favorite food?'")
    results = memory.get_relevant_memories("What is my favorite food?")
    for i, r in enumerate(results):
        print(f"{i+1}. {r.content}")

    print("\n\n>>> Memory Debug Dump")
    memory.print_memory_state()

if __name__ == "__main__":
    main()