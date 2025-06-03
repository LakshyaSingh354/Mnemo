import argparse
import os
import dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from memory.short_term.vector_store import VectorMemoryStore
from memory.short_term.chat_buffer import ChatBuffer
from memory.long_term.summarization import SummarizationMemory

dotenv.load_dotenv()

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT_CHAR")
GREETING = os.getenv("GREETING")

def build_prompt(memories, recent_chats, user_input):
    prompt = SYSTEM_PROMPT + "\n\n"
    if memories:
        prompt += "Relevant memories:\n"
        for mem in memories:
            prompt += f"- {mem}\n"

    if recent_chats:
        for chat in recent_chats:
            if chat.role == "user":
                prompt += f"User: {chat.content}\n"
            else:
                prompt += f"Assistant: {chat.content}\n"
    prompt += f"\nUser: {user_input}\nAssistant:"
    return prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-memory", action="store_true", help="Run agent without memory")
    parser.add_argument("--summarization", action="store_true", help="Enable memory summarization")

    args = parser.parse_args()

    use_memory = not args.no_memory
    use_summarization = args.summarization and use_memory

    store = VectorMemoryStore() if use_memory else None
    summarization = SummarizationMemory(store) if use_summarization else None
    buffer = ChatBuffer(max_length=10)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    print("💬 Agent Memory Chat")
    print("🧠 Memory:", "ON" if use_memory else "OFF")
    print("📝 Summarization:", "ON" if use_summarization else "OFF")
    print("\nCommands:")
    print("/exit - Exit chat")
    print("/debug - Show memory state")
    print("/summarize - Trigger memory summarization")
    print("/summaries - Show all summaries")
    print("Type '/exit' to quit.\n")

    print(f"\nAssistant: {GREETING}")
    if use_memory:
        store.add_message(GREETING, source="chat")
    buffer.add_message(role="assistant", content=GREETING)

    while True:
        user_input = input("You: ")
        if user_input.lower() == '/exit':
            print("Exiting chat. Goodbye!")
            break
        elif user_input.lower() == '/debug':
            if use_memory:
                print("\n\n>>> Memory Debug Dump\n\n")
                store.print_memory_state()
            continue
        elif user_input.lower() == '/summarize':
            if use_summarization:
                print("\n>>> Processing memories for summarization...")
                summarization.process_memories()
                print("Summarization complete!")
            else:
                print("Summarization is not enabled!")
            continue
        elif user_input.lower() == '/summaries':
            if use_summarization:
                print("\n>>> Current Summaries:")
                summarization.print_summaries()
            else:
                print("Summarization is not enabled!")
            continue

        memories = []
        if use_memory:
            # Get both relevant memories and summaries
            results = store.get_relevant_memories(user_input, k=5)
            memories = [mem.content for mem in results]

            if use_summarization:
                summaries = summarization.get_relevant_summaries(user_input, k=2)
                for summary in summaries:
                    memories.append(f"[Summary] {summary.content[:200]}...")

        recent_chats = buffer.get_recent()
        full_prompt = build_prompt(memories, recent_chats, user_input)
        response = llm.invoke([HumanMessage(content=full_prompt)])
        reply = response.content.strip()

        print("\nAgent:", reply)

        if use_memory:
            store.add_message(user_input, source="user")
            store.add_message(reply, source="chat")

        buffer.add_message(role="user", content=user_input)
        buffer.add_message(role="assistant", content=reply)

if __name__ == "__main__":
    main()