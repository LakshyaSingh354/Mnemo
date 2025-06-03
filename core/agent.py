from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from memory.short_term.vector_store import VectorMemoryStore
from memory.short_term.chat_buffer import ChatBuffer
from memory.long_term.summarization import SummarizationMemory

class Agent:
    def __init__(self, use_memory=True, use_summarization=False):
        self.use_memory = use_memory
        self.use_summarization = use_summarization
        self._init_components()

        self.SYSTEM_PROMPT = """
        You are a helpful assistant that can answer questions and help with tasks.
        You have access to a memory store and a summarization memory.
        You can use the memory store to store information and the summarization memory to summarize the information.
        You can use the chat buffer to store the recent chats.
        """

    def _init_components(self):
        self.store = VectorMemoryStore() if self.use_memory else None
        self.summarization = (
            SummarizationMemory(self.store)
            if self.use_summarization and self.use_memory
            else None
        )
        self.buffer = ChatBuffer(max_length=10)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

    def reset(self):
        """Reset memory, buffer, and summarization state."""
        self._init_components()

    def add_memory(self, role: str, content: str):
        """Inject memory directly — useful for test setup."""
        if role == "user":
            if self.use_memory:
                self.buffer.add_message(role="user", content=content)
                self.store.add_message(content, source="user")
        elif role == "assistant":
            if self.use_memory:
                self.buffer.add_message(role="assistant", content=content)
                self.store.add_message(content, source="chat")

    def build_prompt(self, memories, recent_chats, user_input):
        prompt = self.SYSTEM_PROMPT + "\n\n"
        if memories:
            prompt += "Relevant memories:\n"
            for mem in memories:
                prompt += f"- {mem}\n"

        for chat in recent_chats:
            prefix = "User" if chat.role == "user" else "Assistant"
            prompt += f"{prefix}: {chat.content}\n"

        prompt += f"\nUser: {user_input}\nAssistant:"
        return prompt

    def run(self, user_input: str) -> str:
        if self.use_memory:
            self.store.add_message(user_input, source="user")

        memories = []
        if self.use_memory:
            results = self.store.get_relevant_memories(user_input, k=5)
            memories = [mem.content for mem in results]

            if self.use_summarization:
                summaries = self.summarization.get_relevant_summaries(user_input, k=2)
                for summary in summaries:
                    memories.append(f"[Summary] {summary.content[:200]}...")

        recent_chats = self.buffer.get_recent()
        prompt = self.build_prompt(memories, recent_chats, user_input)

        response = self.llm.invoke([HumanMessage(content=prompt)])
        reply = response.content.strip()

        if self.use_memory:
            self.store.add_message(reply, source="chat")

        self.buffer.add_message(role="user", content=user_input)
        self.buffer.add_message(role="assistant", content=reply)

        return reply