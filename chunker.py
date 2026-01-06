"""
Chunker intelligent pour Silas V2.
Découpe les documents en segments exploitables sans utiliser de LLM.
Utilise des heuristiques basées sur la structure du texte.
"""

import os
import re
from dataclasses import dataclass
from typing import List, Tuple
from config import cfg


@dataclass
class Chunk:
    """Représente un segment de document."""
    chunk_id: str       # ex: "rapport.md_s3"
    filename: str       # ex: "rapport.md"
    section_idx: int    # ex: 3
    content: str        # Contenu textuel du chunk
    char_start: int     # Position de début dans le fichier original
    char_end: int       # Position de fin


def find_best_split_point(text: str, target_pos: int, separators: Tuple[str, ...]) -> int:
    """
    Trouve le meilleur point de coupure proche de target_pos.
    Cherche le séparateur le plus "fort" dans une fenêtre autour de target_pos.
    """
    window = cfg.CHUNK_OVERLAP  # Zone de recherche
    search_start = max(0, target_pos - window)
    search_end = min(len(text), target_pos + window)
    search_zone = text[search_start:search_end]
    
    for sep in separators:
        # Chercher la dernière occurrence du séparateur dans la zone
        pos = search_zone.rfind(sep)
        if pos != -1:
            return search_start + pos + len(sep)
    
    # Aucun séparateur trouvé, couper au target
    return target_pos


def chunk_text(text: str, filename: str) -> List[Chunk]:
    """
    Découpe un texte en chunks intelligents.
    
    Stratégie:
    1. Si le texte est petit (< CHUNK_SIZE), un seul chunk
    2. Sinon, découpage aux séparateurs naturels (paragraphes, phrases)
    3. Chevauchement pour préserver le contexte
    """
    chunks = []
    text_len = len(text)
    
    # Document court = 1 seul chunk
    if text_len <= cfg.CHUNK_SIZE:
        return [Chunk(
            chunk_id=f"{filename}_s1",
            filename=filename,
            section_idx=1,
            content=text.strip(),
            char_start=0,
            char_end=text_len
        )]
    
    # Document long = découpage intelligent
    current_pos = 0
    section_idx = 1
    
    while current_pos < text_len:
        # Calculer la fin théorique du chunk
        chunk_end = min(current_pos + cfg.CHUNK_SIZE, text_len)
        
        # Si on n'est pas à la fin, chercher un bon point de coupure
        if chunk_end < text_len:
            chunk_end = find_best_split_point(
                text, chunk_end, cfg.CHUNK_SEPARATORS
            )
        
        # Extraire le contenu
        chunk_content = text[current_pos:chunk_end].strip()
        
        # Ignorer les chunks trop petits (sauf le dernier)
        if len(chunk_content) >= cfg.MIN_CHUNK_SIZE or chunk_end >= text_len:
            chunks.append(Chunk(
                chunk_id=f"{filename}_s{section_idx}",
                filename=filename,
                section_idx=section_idx,
                content=chunk_content,
                char_start=current_pos,
                char_end=chunk_end
            ))
            section_idx += 1
        
        # Avancer avec chevauchement
        current_pos = chunk_end - cfg.CHUNK_OVERLAP
        if current_pos <= chunks[-1].char_start if chunks else 0:
            current_pos = chunk_end  # Éviter boucle infinie
    
    return chunks


def chunk_directory(directory: str) -> Tuple[List[Chunk], dict]:
    """
    Découpe tous les fichiers d'un dossier en chunks.
    
    Returns:
        chunks: Liste de tous les chunks
        stats: Statistiques du découpage
    """
    all_chunks = []
    stats = {
        "files_processed": 0,
        "files_skipped": 0,
        "total_chunks": 0,
        "file_details": {}
    }
    
    files = [f for f in os.listdir(directory) 
             if f.endswith(cfg.SUPPORTED_EXTENSIONS)]
    
    for filename in sorted(files):
        filepath = os.path.join(directory, filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                stats["files_skipped"] += 1
                continue
            
            file_chunks = chunk_text(content, filename)
            all_chunks.extend(file_chunks)
            
            stats["files_processed"] += 1
            stats["total_chunks"] += len(file_chunks)
            stats["file_details"][filename] = {
                "chars": len(content),
                "chunks": len(file_chunks)
            }
            
        except Exception as e:
            stats["files_skipped"] += 1
            stats["file_details"][filename] = {"error": str(e)}
    
    return all_chunks, stats


def format_chunk_for_display(chunk: Chunk, max_len: int = 80) -> str:
    """Formate un chunk pour affichage console."""
    preview = chunk.content[:max_len].replace('\n', ' ')
    if len(chunk.content) > max_len:
        preview += "..."
    return f"[{chunk.chunk_id}] ({len(chunk.content)} chars) {preview}"


# === TEST DIRECT ===
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python chunker.py <directory>")
        sys.exit(1)
    
    directory = sys.argv[1]
    chunks, stats = chunk_directory(directory)
    
    print(f"\n📊 STATISTIQUES DE CHUNKING")
    print(f"   Fichiers traités: {stats['files_processed']}")
    print(f"   Fichiers ignorés: {stats['files_skipped']}")
    print(f"   Total chunks: {stats['total_chunks']}")
    
    print(f"\n📄 DÉTAIL PAR FICHIER:")
    for fname, info in stats["file_details"].items():
        if "error" in info:
            print(f"   ❌ {fname}: {info['error']}")
        else:
            print(f"   ✓ {fname}: {info['chars']} chars → {info['chunks']} chunks")
    
    print(f"\n📦 APERÇU DES CHUNKS:")
    for chunk in chunks[:10]:
        print(f"   {format_chunk_for_display(chunk)}")
    
    if len(chunks) > 10:
        print(f"   ... et {len(chunks) - 10} autres chunks")
