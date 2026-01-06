#!/usr/bin/env python3
"""
Silas V2 - Hierarchical RAG Agent
Point d'entrée CLI avec affichage détaillé.
"""

import sys
import argparse
import warnings
import os
import time
from datetime import datetime

# Suppression des warnings
warnings.filterwarnings("ignore")

from config import cfg
from rag_graph import app


def print_banner():
    """Affiche la bannière de démarrage."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                   SILAS V2 - Hierarchical RAG                 ║
║                    Deep Document Analysis                     ║
╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_config():
    """Affiche la configuration active."""
    print("┌─ Configuration ────────────────────────────────────────────┐")
    print(f"│  🐇 FAST Model (Scan)     : {cfg.FAST_MODEL:<30} │")
    print(f"│  ⚖️  CHOOSE Model (Select) : {cfg.CHOOSE_MODEL:<30} │")
    print(f"│  🧠 SMART Model (Deep)    : {cfg.SMART_MODEL:<30} │")
    print(f"│  📐 Chunk Size            : {cfg.CHUNK_SIZE} chars (overlap: {cfg.CHUNK_OVERLAP})     │")
    print(f"│  🌡️  Temperature           : {cfg.TEMPERATURE:<30} │")
    print("└─────────────────────────────────────────────────────────────┘")


def print_timing_summary(timings: dict, total_time: float):
    """Affiche le résumé des temps d'exécution."""
    print("\n┌─ Temps d'exécution ────────────────────────────────────────┐")
    
    steps = [
        ("Chunking", "chunking"),
        ("Scanning", "scanning"),
        ("Selection", "selection"),
        ("Extraction", "extraction"),
        ("Synthesis", "synthesis"),
    ]
    
    for label, key in steps:
        if key in timings:
            t = timings[key]
            pct = (t / total_time) * 100 if total_time > 0 else 0
            bar_len = 20
            filled = int(bar_len * pct / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"│  {label:<12} {bar} {t:6.2f}s ({pct:5.1f}%) │")
    
    print(f"│  {'─' * 53} │")
    print(f"│  {'TOTAL':<12} {'':20} {total_time:6.2f}s {'':8} │")
    print("└─────────────────────────────────────────────────────────────┘")


def main():
    parser = argparse.ArgumentParser(
        description="Silas V2 - Hierarchical RAG pour analyse documentaire",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python main.py -q "Quelle est la date du rapport?" -d ./docs/
  python main.py -q "Résume les points clés" -d ./projet/ -v
        """
    )
    parser.add_argument("-q", "--question", type=str, required=True, 
                        help="Question à poser aux documents")
    parser.add_argument("-d", "--directory", type=str, required=True, 
                        help="Dossier contenant les documents")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Mode verbeux (affiche plus de détails)")
    
    args = parser.parse_args()

    # Validation du dossier
    if not os.path.isdir(args.directory):
        print(f"❌ Erreur : Le dossier '{args.directory}' est introuvable.")
        sys.exit(1)

    # Compter les fichiers
    supported = cfg.SUPPORTED_EXTENSIONS
    files = [f for f in os.listdir(args.directory) if f.endswith(supported)]
    
    if not files:
        print(f"❌ Erreur : Aucun fichier supporté trouvé dans '{args.directory}'")
        print(f"   Extensions supportées: {', '.join(supported)}")
        sys.exit(1)

    # Affichage initial
    print_banner()
    print_config()
    
    print("\n┌─ Requête ───────────────────────────────────────────────────┐")
    print(f"│  📂 Dossier : {args.directory:<44} │")
    print(f"│  📄 Fichiers: {len(files):<44} │")
    
    # Tronquer la question si trop longue pour l'affichage
    q_display = args.question[:42] + "..." if len(args.question) > 45 else args.question
    print(f"│  ❓ Question: {q_display:<44} │")
    print("└─────────────────────────────────────────────────────────────┘")
    
    # Préparer l'état initial
    initial_state = {
        "question": args.question,
        "file_directory": args.directory,
        "chunks": [],
        "chunk_summaries": [],
        "selected_chunks": [],
        "extracted_evidence": [],
        "final_answer": "",
        "timings": {}
    }

    start_time = time.time()
    
    try:
        # Lancement du pipeline
        result = app.invoke(initial_state)
        
        total_time = time.time() - start_time
        
        # Statistiques
        print_timing_summary(result.get("timings", {}), total_time)
        
        # Stats du traitement
        print("\n┌─ Statistiques ──────────────────────────────────────────────┐")
        print(f"│  Chunks créés     : {len(result.get('chunks', [])):<38} │")
        print(f"│  Chunks scannés   : {len(result.get('chunk_summaries', [])):<38} │")
        print(f"│  Chunks retenus   : {len(result.get('selected_chunks', [])):<38} │")
        print(f"│  Extraits générés : {len(result.get('extracted_evidence', [])):<38} │")
        print("└─────────────────────────────────────────────────────────────┘")
        
        # Réponse finale
        print("\n" + "═" * 61)
        print("                         RÉPONSE FINALE")
        print("═" * 61)
        print()
        print(result["final_answer"])
        print()
        print("═" * 61)
        
    except KeyboardInterrupt:
        print("\n\n👋 Arrêt demandé par l'utilisateur.")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Erreur critique : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
