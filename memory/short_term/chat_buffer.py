from dataclasses import dataclass
import time
from typing import List


@dataclass
class ChatMessage:
    role: str  # "user", "assistant", or "system"
    content: str
    timestamp: float

class ChatBuffer:
    def __init__(self, max_length=20):
        self.max_length = max_length
        self.buffer: List[ChatMessage] = []

    def add_message(self, role: str, content: str):
        entry = ChatMessage(role=role, content=content, timestamp=time.time())
        self.buffer.append(entry)
        if len(self.buffer) > self.max_length:
            self.buffer.pop(0)  # Evict oldest message

    def get_recent(self, n=10):
        return self.buffer[-n:]