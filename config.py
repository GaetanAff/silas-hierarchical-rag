import os
from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class AgentConfig:
    # === LLM MODELS ===
    # SMART: Precise extraction & Final drafting
    SMART_MODEL: str = "qwen3:14b"
    # CHOOSE: Selection of relevant chunks
    CHOOSE_MODEL: str = "qwen3:8b"
    # FAST: Rapid mass scan/summary
    FAST_MODEL: str = "qwen3:0.6b"
    
    # === OLLAMA CONNECTION ===
    BASE_URL: str = "http://localhost:11434"
    TEMPERATURE: float = 0
    
    # === CHUNKING ===
    # Target chunk size in characters
    CHUNK_SIZE: int = 1500
    # Overlap between chunks (avoids cutting an idea mid-sentence)
    CHUNK_OVERLAP: int = 200
    # Minimum chunk size (avoids micro-chunks)
    MIN_CHUNK_SIZE: int = 300
    # Separators for intelligent splitting (priority order)
    CHUNK_SEPARATORS: Tuple[str, ...] = (
        "\n\n\n",   # Triple newline = major section
        "\n\n",     # Double newline = paragraph
        "\n",       # Single newline = line
        ". ",       # End of sentence
        ", ",       # Comma
        " ",        # Space (last resort)
    )
    
    # === SUPPORTED EXTENSIONS ===
    SUPPORTED_EXTENSIONS: Tuple[str, ...] = (
        ".txt", ".md", ".py", ".json", ".csv", 
        ".log", ".yml", ".yaml", ".xml", ".html"
    )
    
    # === DISPLAY ===
    VERBOSE: bool = True
    SHOW_CHUNK_CONTENT: bool = False  # Show chunk content (debug)

cfg = AgentConfig()
