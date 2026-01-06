"""
Intelligent Chunker for Silas V2.
Splits documents into exploitable segments without using an LLM.
Uses heuristics based on text structure.
"""

import os
import re
from dataclasses import dataclass
from typing import List, Tuple
from config import cfg


@dataclass
class Chunk:
    """Represents a document segment."""
    chunk_id: str       # ex: "report.md_s3"
    filename: str       # ex: "report.md"
    section_idx: int    # ex: 3
    content: str        # Textual content of the chunk
    char_start: int     # Start position in original file
    char_end: int       # End position


def find_best_split_point(text: str, target_pos: int, separators: Tuple[str, ...]) -> int:
    """
    Finds the best split point close to target_pos.
    Looks for the "strongest" separator in a window around target_pos.
    """
    window = cfg.CHUNK_OVERLAP  # Search zone
    search_start = max(0, target_pos - window)
    search_end = min(len(text), target_pos + window)
    search_zone = text[search_start:search_end]
    
    for sep in separators:
        # Find the last occurrence of the separator in the zone
        pos = search_zone.rfind(sep)
        if pos != -1:
            return search_start + pos + len(sep)
    
    # No separator found, split at target
    return target_pos


def chunk_text(text: str, filename: str) -> List[Chunk]:
    """
    Splits text into intelligent chunks.
    
    Strategy:
    1. If text is small (< CHUNK_SIZE), single chunk
    2. Otherwise, split at natural separators (paragraphs, sentences)
    3. Overlap to preserve context
    """
    chunks = []
    text_len = len(text)
    
    # Short document = single chunk
    if text_len <= cfg.CHUNK_SIZE:
        return [Chunk(
            chunk_id=f"{filename}_s1",
            filename=filename,
            section_idx=1,
            content=text.strip(),
            char_start=0,
            char_end=text_len
        )]
    
    # Long document = intelligent splitting
    current_pos = 0
    section_idx = 1
    
    while current_pos < text_len:
        # Calculate theoretical chunk end
        chunk_end = min(current_pos + cfg.CHUNK_SIZE, text_len)
        
        # If not at end, find a good split point
        if chunk_end < text_len:
            chunk_end = find_best_split_point(
                text, chunk_end, cfg.CHUNK_SEPARATORS
            )
        
        # Extract content
        chunk_content = text[current_pos:chunk_end].strip()
        
        # Ignore chunks that are too small (except the last one)
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
        
        # Advance with overlap
        current_pos = chunk_end - cfg.CHUNK_OVERLAP
        if current_pos <= chunks[-1].char_start if chunks else 0:
            current_pos = chunk_end  # Avoid infinite loop
    
    return chunks


def chunk_directory(directory: str) -> Tuple[List[Chunk], dict]:
    """
    Splits all files in a directory into chunks.
    
    Returns:
        chunks: List of all chunks
        stats: Splitting statistics
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
    """Formats a chunk for console display."""
    preview = chunk.content[:max_len].replace('\n', ' ')
    if len(chunk.content) > max_len:
        preview += "..."
    return f"[{chunk.chunk_id}] ({len(chunk.content)} chars) {preview}"


# === DIRECT TEST ===
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python chunker.py <directory>")
        sys.exit(1)
    
    directory = sys.argv[1]
    chunks, stats = chunk_directory(directory)
    
    print(f"\n📊 CHUNKING STATISTICS")
    print(f"   Processed files: {stats['files_processed']}")
    print(f"   Skipped files:   {stats['files_skipped']}")
    print(f"   Total chunks:    {stats['total_chunks']}")
    
    print(f"\n📄 DETAILS PER FILE:")
    for fname, info in stats["file_details"].items():
        if "error" in info:
            print(f"   ❌ {fname}: {info['error']}")
        else:
            print(f"   ✓ {fname}: {info['chars']} chars → {info['chunks']} chunks")
    
    print(f"\n📦 CHUNK PREVIEW:")
    for chunk in chunks[:10]:
        print(f"   {format_chunk_for_display(chunk)}")
    
    if len(chunks) > 10:
        print(f"   ... and {len(chunks) - 10} other chunks")
