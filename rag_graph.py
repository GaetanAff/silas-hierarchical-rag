"""
Silas V2 - Hierarchical RAG Pipeline.

Flow:
1. CHUNKER   → Splits all docs into segments
2. SCANNER   → Summarizes each chunk (FAST_MODEL)
3. SELECTOR  → Selects relevant chunks (CHOOSE_MODEL)
4. EXTRACTOR → Extracts info from elected chunks (SMART_MODEL)
5. SYNTHESIZER → Writes the final answer (SMART_MODEL)
"""

import os
import ast
import time
from typing import List, TypedDict, Optional
from dataclasses import dataclass

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

from config import cfg
from chunker import Chunk, chunk_directory, format_chunk_for_display
from prompts import (
    SCAN_PROMPT, SELECTOR_PROMPT, EXTRACTOR_PROMPT, 
    SILAS_PERSONA, SYNTHESIZE_PROMPT
)


# === GRAPH STATE ===
class AgentState(TypedDict):
    question: str
    file_directory: str
    # Chunking
    chunks: List[dict]              # List of chunks (serialized)
    chunk_summaries: List[str]      # ["chunk_id: summary", ...]
    # Selection
    selected_chunks: List[str]      # ["doc1.md_s2", "doc2.txt_s1"]
    # Extraction
    extracted_evidence: List[str]   # Extracted passages with source
    # Answer
    final_answer: str
    # Stats
    timings: dict


# === DISPLAY HELPERS ===
def print_header(title: str, emoji: str = "▶"):
    """Displays a step header."""
    print(f"\n{'='*60}")
    print(f"{emoji} {title}")
    print('='*60)


def print_step(msg: str, indent: int = 1):
    """Displays a step line."""
    prefix = "  " * indent
    print(f"{prefix}• {msg}")


def print_progress(current: int, total: int, item: str, result: str = ""):
    """Displays a simple progress bar."""
    pct = (current / total) * 100 if total > 0 else 0
    bar_len = 20
    filled = int(bar_len * current / total) if total > 0 else 0
    bar = "█" * filled + "░" * (bar_len - filled)
    status = f" → {result}" if result else ""
    print(f"\r  [{bar}] {current}/{total} ({pct:.0f}%) {item}{status}", end="", flush=True)


# === INIT LLMs ===
fast_llm = ChatOllama(model=cfg.FAST_MODEL, base_url=cfg.BASE_URL, temperature=cfg.TEMPERATURE)
choose_llm = ChatOllama(model=cfg.CHOOSE_MODEL, base_url=cfg.BASE_URL, temperature=cfg.TEMPERATURE)
smart_llm = ChatOllama(model=cfg.SMART_MODEL, base_url=cfg.BASE_URL, temperature=cfg.TEMPERATURE)


# === NODE 1: CHUNKER ===
def chunk_node(state: AgentState) -> dict:
    """Splits all documents into chunks."""
    print_header("STEP 1: CHUNKING", "✂️")
    start = time.time()
    
    directory = state["file_directory"]
    chunks, stats = chunk_directory(directory)
    
    print_step(f"Directory: {directory}")
    print_step(f"Processed files: {stats['files_processed']}")
    print_step(f"Skipped files: {stats['files_skipped']}")
    print_step(f"Generated chunks: {stats['total_chunks']}")
    
    # Detail per file
    print("\n  File details:")
    for fname, info in stats["file_details"].items():
        if "error" in info:
            print(f"    ❌ {fname}: {info['error']}")
        else:
            ratio = info['chunks']
            print(f"    📄 {fname}: {info['chars']:,} chars → {ratio} chunk(s)")
    
    # Serialize chunks for state
    chunks_data = [
        {
            "chunk_id": c.chunk_id,
            "filename": c.filename,
            "section_idx": c.section_idx,
            "content": c.content,
            "char_start": c.char_start,
            "char_end": c.char_end
        }
        for c in chunks
    ]
    
    elapsed = time.time() - start
    print(f"\n  ⏱️  Duration: {elapsed:.2f}s")
    
    return {
        "chunks": chunks_data,
        "timings": {**state.get("timings", {}), "chunking": elapsed}
    }


# === NODE 2: SCANNER ===
def scan_node(state: AgentState) -> dict:
    """Summarizes each chunk with the fast model."""
    print_header("STEP 2: SCANNING", "🔍")
    start = time.time()
    
    chunks = state["chunks"]
    total = len(chunks)
    
    if total == 0:
        print_step("No chunks to scan")
        return {"chunk_summaries": []}
    
    print_step(f"Model: {cfg.FAST_MODEL}")
    print_step(f"Chunks to scan: {total}")
    print()
    
    summaries = []
    for i, chunk in enumerate(chunks, 1):
        chunk_id = chunk["chunk_id"]
        content = chunk["content"]
        
        # Limit content sent to scanner
        preview = content[:2000] if len(content) > 2000 else content
        
        try:
            prompt = SCAN_PROMPT.format(content=preview)
            response = fast_llm.invoke([HumanMessage(content=prompt)])
            summary = response.content.strip().replace("\n", " ")[:150]
            
            summaries.append(f"[{chunk_id}]: {summary}")
            print_progress(i, total, chunk_id, summary[:40] + "...")
            
        except Exception as e:
            summaries.append(f"[{chunk_id}]: (scan error)")
            print_progress(i, total, chunk_id, f"ERROR: {e}")
    
    print()  # New line after bar
    elapsed = time.time() - start
    print(f"\n  ⏱️  Duration: {elapsed:.2f}s ({elapsed/total:.2f}s/chunk)")
    
    return {
        "chunk_summaries": summaries,
        "timings": {**state.get("timings", {}), "scanning": elapsed}
    }


# === NODE 3: SELECTOR ===
def select_node(state: AgentState) -> dict:
    """Selects relevant chunks."""
    print_header("STEP 3: SELECTION", "🎯")
    start = time.time()
    
    summaries = state["chunk_summaries"]
    question = state["question"]
    
    if not summaries:
        print_step("No summaries available")
        return {"selected_chunks": []}
    
    print_step(f"Model: {cfg.CHOOSE_MODEL}")
    print_step(f"Question: {question[:80]}...")
    print_step(f"Candidate chunks: {len(summaries)}")
    
    # Display summaries
    print("\n  Analyzed summaries:")
    for s in summaries[:10]:  # Limit display
        print(f"    {s[:100]}...")
    if len(summaries) > 10:
        print(f"    ... and {len(summaries) - 10} others")
    
    # Call selection model
    summaries_text = "\n".join(summaries)
    prompt = SELECTOR_PROMPT.format(question=question, summaries=summaries_text)
    
    print("\n  Analyzing...")
    response = choose_llm.invoke([HumanMessage(content=prompt)])
    raw_response = response.content.strip()
    
    # Parse Python list
    selected = []
    try:
        start_bracket = raw_response.find('[')
        end_bracket = raw_response.rfind(']') + 1
        if start_bracket != -1 and end_bracket > start_bracket:
            list_str = raw_response[start_bracket:end_bracket]
            selected = ast.literal_eval(list_str)
    except Exception as e:
        print(f"  ⚠️  Parsing error: {e}")
        print(f"     Raw response: {raw_response[:200]}")
    
    # Validation: check IDs exist
    valid_ids = {c["chunk_id"] for c in state["chunks"]}
    selected = [s for s in selected if s in valid_ids]
    
    elapsed = time.time() - start
    
    print(f"\n  📋 Selected chunks ({len(selected)}):")
    for chunk_id in selected:
        print(f"    ✓ {chunk_id}")
    
    if not selected:
        print("    (no chunks selected)")
    
    print(f"\n  ⏱️  Duration: {elapsed:.2f}s")
    
    return {
        "selected_chunks": selected,
        "timings": {**state.get("timings", {}), "selection": elapsed}
    }


# === NODE 4: EXTRACTOR ===
def extract_node(state: AgentState) -> dict:
    """Extracts information from selected chunks."""
    print_header("STEP 4: EXTRACTION", "⛏️")
    start = time.time()
    
    selected = state["selected_chunks"]
    question = state["question"]
    chunks_map = {c["chunk_id"]: c for c in state["chunks"]}
    
    if not selected:
        print_step("No chunks to analyze")
        return {"extracted_evidence": ["No relevant information found in documents."]}
    
    print_step(f"Model: {cfg.SMART_MODEL}")
    print_step(f"Chunks to extract: {len(selected)}")
    print()
    
    evidence = []
    for i, chunk_id in enumerate(selected, 1):
        chunk = chunks_map.get(chunk_id)
        if not chunk:
            continue
        
        content = chunk["content"]
        
        print_step(f"[{i}/{len(selected)}] Analyzing {chunk_id} ({len(content)} chars)", indent=1)
        
        try:
            prompt = EXTRACTOR_PROMPT.format(
                question=question,
                chunk_id=chunk_id,
                content=content
            )
            response = smart_llm.invoke([HumanMessage(content=prompt)])
            extracted = response.content.strip()
            
            if "NOTHING" not in extracted.upper() and len(extracted) > 10:
                evidence.append(f"--- Source: {chunk_id} ---\n{extracted}")
                preview = extracted[:60].replace('\n', ' ')
                print_step(f"✅ Info found: {preview}...", indent=2)
            else:
                print_step(f"⬜ Nothing relevant", indent=2)
                
        except Exception as e:
            print_step(f"❌ Error: {e}", indent=2)
    
    elapsed = time.time() - start
    print(f"\n  📊 Result: {len(evidence)} relevant extracts")
    print(f"  ⏱️  Duration: {elapsed:.2f}s")
    
    if not evidence:
        evidence = ["No relevant information could be extracted from selected chunks."]
    
    return {
        "extracted_evidence": evidence,
        "timings": {**state.get("timings", {}), "extraction": elapsed}
    }


# === NODE 5: SYNTHESIZER ===
def synthesize_node(state: AgentState) -> dict:
    """Writes the final answer."""
    print_header("STEP 5: SYNTHESIS", "✍️")
    start = time.time()
    
    question = state["question"]
    evidence = state["extracted_evidence"]
    
    print_step(f"Model: {cfg.SMART_MODEL}")
    print_step(f"Evidence to synthesize: {len(evidence)}")
    
    # Build context
    evidence_text = "\n\n".join(evidence)
    
    # Call model
    messages = [
        SystemMessage(content=SILAS_PERSONA),
        HumanMessage(content=SYNTHESIZE_PROMPT.format(
            question=question,
            evidence=evidence_text
        ))
    ]
    
    print_step("Writing answer...")
    response = smart_llm.invoke(messages)
    answer = response.content.strip()
    
    elapsed = time.time() - start
    print(f"\n  ⏱️  Duration: {elapsed:.2f}s")
    
    return {
        "final_answer": answer,
        "timings": {**state.get("timings", {}), "synthesis": elapsed}
    }


# === GRAPH CONSTRUCTION ===
def build_graph() -> StateGraph:
    """Builds the LangGraph workflow."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("chunker", chunk_node)
    workflow.add_node("scanner", scan_node)
    workflow.add_node("selector", select_node)
    workflow.add_node("extractor", extract_node)
    workflow.add_node("synthesizer", synthesize_node)
    
    # Define flow
    workflow.set_entry_point("chunker")
    workflow.add_edge("chunker", "scanner")
    workflow.add_edge("scanner", "selector")
    workflow.add_edge("selector", "extractor")
    workflow.add_edge("extractor", "synthesizer")
    workflow.add_edge("synthesizer", END)
    
    return workflow.compile()


# Global instance
app = build_graph()
