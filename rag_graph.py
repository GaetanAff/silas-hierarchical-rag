"""
Silas V2 - Hierarchical RAG Pipeline.

Flux:
1. CHUNKER   → Découpe tous les docs en segments
2. SCANNER   → Résume chaque chunk (FAST_MODEL)
3. SELECTOR  → Sélectionne les chunks pertinents (CHOOSE_MODEL)
4. EXTRACTOR → Extrait l'info des chunks élus (SMART_MODEL)
5. SYNTHESIZER → Rédige la réponse finale (SMART_MODEL)
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


# === ÉTAT DU GRAPH ===
class AgentState(TypedDict):
    question: str
    file_directory: str
    # Chunking
    chunks: List[dict]              # Liste des chunks (sérialisés)
    chunk_summaries: List[str]      # ["chunk_id: résumé", ...]
    # Sélection
    selected_chunks: List[str]      # ["doc1.md_s2", "doc2.txt_s1"]
    # Extraction
    extracted_evidence: List[str]   # Passages extraits avec source
    # Réponse
    final_answer: str
    # Stats
    timings: dict


# === HELPERS D'AFFICHAGE ===
def print_header(title: str, emoji: str = "▶"):
    """Affiche un header d'étape."""
    print(f"\n{'='*60}")
    print(f"{emoji} {title}")
    print('='*60)


def print_step(msg: str, indent: int = 1):
    """Affiche une ligne d'étape."""
    prefix = "  " * indent
    print(f"{prefix}• {msg}")


def print_progress(current: int, total: int, item: str, result: str = ""):
    """Affiche une barre de progression simple."""
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
    """Découpe tous les documents en chunks."""
    print_header("ÉTAPE 1: CHUNKING", "✂️")
    start = time.time()
    
    directory = state["file_directory"]
    chunks, stats = chunk_directory(directory)
    
    print_step(f"Dossier: {directory}")
    print_step(f"Fichiers traités: {stats['files_processed']}")
    print_step(f"Fichiers ignorés: {stats['files_skipped']}")
    print_step(f"Chunks générés: {stats['total_chunks']}")
    
    # Détail par fichier
    print("\n  Détail par fichier:")
    for fname, info in stats["file_details"].items():
        if "error" in info:
            print(f"    ❌ {fname}: {info['error']}")
        else:
            ratio = info['chunks']
            print(f"    📄 {fname}: {info['chars']:,} chars → {ratio} chunk(s)")
    
    # Sérialiser les chunks pour le state
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
    print(f"\n  ⏱️  Durée: {elapsed:.2f}s")
    
    return {
        "chunks": chunks_data,
        "timings": {**state.get("timings", {}), "chunking": elapsed}
    }


# === NODE 2: SCANNER ===
def scan_node(state: AgentState) -> dict:
    """Résume chaque chunk avec le modèle rapide."""
    print_header("ÉTAPE 2: SCAN", "🔍")
    start = time.time()
    
    chunks = state["chunks"]
    total = len(chunks)
    
    if total == 0:
        print_step("Aucun chunk à scanner")
        return {"chunk_summaries": []}
    
    print_step(f"Modèle: {cfg.FAST_MODEL}")
    print_step(f"Chunks à scanner: {total}")
    print()
    
    summaries = []
    for i, chunk in enumerate(chunks, 1):
        chunk_id = chunk["chunk_id"]
        content = chunk["content"]
        
        # Limiter le contenu envoyé au scanner
        preview = content[:2000] if len(content) > 2000 else content
        
        try:
            prompt = SCAN_PROMPT.format(content=preview)
            response = fast_llm.invoke([HumanMessage(content=prompt)])
            summary = response.content.strip().replace("\n", " ")[:150]
            
            summaries.append(f"[{chunk_id}]: {summary}")
            print_progress(i, total, chunk_id, summary[:40] + "...")
            
        except Exception as e:
            summaries.append(f"[{chunk_id}]: (erreur de scan)")
            print_progress(i, total, chunk_id, f"ERREUR: {e}")
    
    print()  # Nouvelle ligne après la barre
    elapsed = time.time() - start
    print(f"\n  ⏱️  Durée: {elapsed:.2f}s ({elapsed/total:.2f}s/chunk)")
    
    return {
        "chunk_summaries": summaries,
        "timings": {**state.get("timings", {}), "scanning": elapsed}
    }


# === NODE 3: SELECTOR ===
def select_node(state: AgentState) -> dict:
    """Sélectionne les chunks pertinents."""
    print_header("ÉTAPE 3: SÉLECTION", "🎯")
    start = time.time()
    
    summaries = state["chunk_summaries"]
    question = state["question"]
    
    if not summaries:
        print_step("Aucun résumé disponible")
        return {"selected_chunks": []}
    
    print_step(f"Modèle: {cfg.CHOOSE_MODEL}")
    print_step(f"Question: {question[:80]}...")
    print_step(f"Chunks candidats: {len(summaries)}")
    
    # Afficher les résumés
    print("\n  Résumés analysés:")
    for s in summaries[:10]:  # Limiter l'affichage
        print(f"    {s[:100]}...")
    if len(summaries) > 10:
        print(f"    ... et {len(summaries) - 10} autres")
    
    # Appel au modèle de sélection
    summaries_text = "\n".join(summaries)
    prompt = SELECTOR_PROMPT.format(question=question, summaries=summaries_text)
    
    print("\n  Analyse en cours...")
    response = choose_llm.invoke([HumanMessage(content=prompt)])
    raw_response = response.content.strip()
    
    # Parser la liste Python
    selected = []
    try:
        start_bracket = raw_response.find('[')
        end_bracket = raw_response.rfind(']') + 1
        if start_bracket != -1 and end_bracket > start_bracket:
            list_str = raw_response[start_bracket:end_bracket]
            selected = ast.literal_eval(list_str)
    except Exception as e:
        print(f"  ⚠️  Erreur de parsing: {e}")
        print(f"     Réponse brute: {raw_response[:200]}")
    
    # Validation: vérifier que les IDs existent
    valid_ids = {c["chunk_id"] for c in state["chunks"]}
    selected = [s for s in selected if s in valid_ids]
    
    elapsed = time.time() - start
    
    print(f"\n  📋 Chunks sélectionnés ({len(selected)}):")
    for chunk_id in selected:
        print(f"    ✓ {chunk_id}")
    
    if not selected:
        print("    (aucun chunk retenu)")
    
    print(f"\n  ⏱️  Durée: {elapsed:.2f}s")
    
    return {
        "selected_chunks": selected,
        "timings": {**state.get("timings", {}), "selection": elapsed}
    }


# === NODE 4: EXTRACTOR ===
def extract_node(state: AgentState) -> dict:
    """Extrait l'information des chunks sélectionnés."""
    print_header("ÉTAPE 4: EXTRACTION", "⛏️")
    start = time.time()
    
    selected = state["selected_chunks"]
    question = state["question"]
    chunks_map = {c["chunk_id"]: c for c in state["chunks"]}
    
    if not selected:
        print_step("Aucun chunk à analyser")
        return {"extracted_evidence": ["Aucune information pertinente trouvée dans les documents."]}
    
    print_step(f"Modèle: {cfg.SMART_MODEL}")
    print_step(f"Chunks à extraire: {len(selected)}")
    print()
    
    evidence = []
    for i, chunk_id in enumerate(selected, 1):
        chunk = chunks_map.get(chunk_id)
        if not chunk:
            continue
        
        content = chunk["content"]
        
        print_step(f"[{i}/{len(selected)}] Analyse de {chunk_id} ({len(content)} chars)", indent=1)
        
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
                print_step(f"✅ Info trouvée: {preview}...", indent=2)
            else:
                print_step(f"⬜ Rien de pertinent", indent=2)
                
        except Exception as e:
            print_step(f"❌ Erreur: {e}", indent=2)
    
    elapsed = time.time() - start
    print(f"\n  📊 Résultat: {len(evidence)} extraits pertinents")
    print(f"  ⏱️  Durée: {elapsed:.2f}s")
    
    if not evidence:
        evidence = ["Aucune information pertinente n'a pu être extraite des chunks sélectionnés."]
    
    return {
        "extracted_evidence": evidence,
        "timings": {**state.get("timings", {}), "extraction": elapsed}
    }


# === NODE 5: SYNTHESIZER ===
def synthesize_node(state: AgentState) -> dict:
    """Rédige la réponse finale."""
    print_header("ÉTAPE 5: SYNTHÈSE", "✍️")
    start = time.time()
    
    question = state["question"]
    evidence = state["extracted_evidence"]
    
    print_step(f"Modèle: {cfg.SMART_MODEL}")
    print_step(f"Preuves à synthétiser: {len(evidence)}")
    
    # Construire le contexte
    evidence_text = "\n\n".join(evidence)
    
    # Appel au modèle
    messages = [
        SystemMessage(content=SILAS_PERSONA),
        HumanMessage(content=SYNTHESIZE_PROMPT.format(
            question=question,
            evidence=evidence_text
        ))
    ]
    
    print_step("Rédaction en cours...")
    response = smart_llm.invoke(messages)
    answer = response.content.strip()
    
    elapsed = time.time() - start
    print(f"\n  ⏱️  Durée: {elapsed:.2f}s")
    
    return {
        "final_answer": answer,
        "timings": {**state.get("timings", {}), "synthesis": elapsed}
    }


# === CONSTRUCTION DU GRAPH ===
def build_graph() -> StateGraph:
    """Construit le workflow LangGraph."""
    workflow = StateGraph(AgentState)
    
    # Ajout des nodes
    workflow.add_node("chunker", chunk_node)
    workflow.add_node("scanner", scan_node)
    workflow.add_node("selector", select_node)
    workflow.add_node("extractor", extract_node)
    workflow.add_node("synthesizer", synthesize_node)
    
    # Définition du flux
    workflow.set_entry_point("chunker")
    workflow.add_edge("chunker", "scanner")
    workflow.add_edge("scanner", "selector")
    workflow.add_edge("selector", "extractor")
    workflow.add_edge("extractor", "synthesizer")
    workflow.add_edge("synthesizer", END)
    
    return workflow.compile()


# Instance globale
app = build_graph()
