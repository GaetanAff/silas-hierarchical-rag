import os
from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class AgentConfig:
    # === MODÈLES LLM ===
    # SMART: Extraction précise & Rédaction finale
    SMART_MODEL: str = "qwen3:14b"
    # CHOOSE: Sélection des chunks pertinents
    CHOOSE_MODEL: str = "qwen3:8b"
    # FAST: Scan/résumé rapide de masse
    FAST_MODEL: str = "qwen3:0.6b"
    
    # === CONNEXION OLLAMA ===
    BASE_URL: str = "http://localhost:11434"
    TEMPERATURE: float = 0
    
    # === CHUNKING ===
    # Taille cible d'un chunk en caractères
    CHUNK_SIZE: int = 1500
    # Chevauchement entre chunks (évite de couper une idée)
    CHUNK_OVERLAP: int = 200
    # Taille min d'un chunk (évite les micro-chunks)
    MIN_CHUNK_SIZE: int = 300
    # Séparateurs pour découpage intelligent (ordre de priorité)
    CHUNK_SEPARATORS: Tuple[str, ...] = (
        "\n\n\n",   # Triple saut = section majeure
        "\n\n",     # Double saut = paragraphe
        "\n",       # Simple saut = ligne
        ". ",       # Fin de phrase
        ", ",       # Virgule
        " ",        # Espace (dernier recours)
    )
    
    # === EXTENSIONS SUPPORTÉES ===
    SUPPORTED_EXTENSIONS: Tuple[str, ...] = (
        ".txt", ".md", ".py", ".json", ".csv", 
        ".log", ".yml", ".yaml", ".xml", ".html"
    )
    
    # === AFFICHAGE ===
    VERBOSE: bool = True
    SHOW_CHUNK_CONTENT: bool = False  # Afficher le contenu des chunks (debug)

cfg = AgentConfig()
