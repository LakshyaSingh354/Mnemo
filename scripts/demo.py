from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
import os
from memory.short_term.vector_store import VectorMemoryStore

store = VectorMemoryStore()
llm = ChatOllama(model="mistral", temperature=0.7)

SYSTEM_PROMPT = """You are an assistant that remembers past conversations and uses them to give better responses.
Use the relevant past memories provided as context to answer the user's query."""

def build_prompt(memories, user_input):
    prompt = SYSTEM_PROMPT + "\n\n"
    if memories:
        prompt += "Relevant memories:\n"
        for mem in memories:
            prompt += f"- {mem}\n"
    prompt += f"\nUser: {user_input}\nAssistant:"
    return prompt

def main():
    print("💬 Agent Memory Chat. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        memories = store.get_relevant_memories(user_input, k=5)
        full_prompt = build_prompt(memories, user_input)

        response = llm.invoke([HumanMessage(content=full_prompt)])
        reply = response.content.strip()

        print("Agent:", reply)

        store.add_message(user_input)
        store.add_message(reply)

if __name__ == "__main__":
    main()