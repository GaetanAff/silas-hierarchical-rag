#!/usr/bin/env python3
"""
Silas V2 - Hierarchical RAG Agent
CLI entry point with detailed display.
"""

import sys
import argparse
import warnings
import os
import time
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore")

from config import cfg
from rag_graph import app


def print_banner():
    """Displays the startup banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                   SILAS V2 - Hierarchical RAG                 ║
║                    Deep Document Analysis                     ║
╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_config():
    """Displays the active configuration."""
    print("┌─ Configuration ────────────────────────────────────────────┐")
    print(f"│  🐇 FAST Model (Scan)     : {cfg.FAST_MODEL:<30} │")
    print(f"│  ⚖️  CHOOSE Model (Select) : {cfg.CHOOSE_MODEL:<30} │")
    print(f"│  🧠 SMART Model (Deep)    : {cfg.SMART_MODEL:<30} │")
    print(f"│  📐 Chunk Size            : {cfg.CHUNK_SIZE} chars (overlap: {cfg.CHUNK_OVERLAP})     │")
    print(f"│  🌡️  Temperature           : {cfg.TEMPERATURE:<30} │")
    print("└─────────────────────────────────────────────────────────────┘")


def print_timing_summary(timings: dict, total_time: float):
    """Displays the execution time summary."""
    print("\n┌─ Execution Time ───────────────────────────────────────────┐")
    
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
        description="Silas V2 - Hierarchical RAG for document analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py -q "What is the date of the report?" -d ./docs/
  python main.py -q "Summarize the key points" -d ./project/ -v
        """
    )
    parser.add_argument("-q", "--question", type=str, required=True, 
                        help="Question to ask the documents")
    parser.add_argument("-d", "--directory", type=str, required=True, 
                        help="Directory containing the documents")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose mode (shows more details)")
    
    args = parser.parse_args()

    # Directory validation
    if not os.path.isdir(args.directory):
        print(f"❌ Error: The directory '{args.directory}' was not found.")
        sys.exit(1)

    # Count files
    supported = cfg.SUPPORTED_EXTENSIONS
    files = [f for f in os.listdir(args.directory) if f.endswith(supported)]
    
    if not files:
        print(f"❌ Error: No supported files found in '{args.directory}'")
        print(f"   Supported extensions: {', '.join(supported)}")
        sys.exit(1)

    # Initial display
    print_banner()
    print_config()
    
    print("\n┌─ Query ─────────────────────────────────────────────────────┐")
    print(f"│  📂 Folder : {args.directory:<44} │")
    print(f"│  📄 Files  : {len(files):<44} │")
    
    # Truncate question if too long for display
    q_display = args.question[:42] + "..." if len(args.question) > 45 else args.question
    print(f"│  ❓ Question: {q_display:<43} │")
    print("└─────────────────────────────────────────────────────────────┘")
    
    # Prepare initial state
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
        # Run pipeline
        result = app.invoke(initial_state)
        
        total_time = time.time() - start_time
        
        # Statistics
        print_timing_summary(result.get("timings", {}), total_time)
        
        # Processing Stats
        print("\n┌─ Statistics ────────────────────────────────────────────────┐")
        print(f"│  Chunks created   : {len(result.get('chunks', [])):<38} │")
        print(f"│  Chunks scanned   : {len(result.get('chunk_summaries', [])):<38} │")
        print(f"│  Chunks selected  : {len(result.get('selected_chunks', [])):<38} │")
        print(f"│  Evidence extracted: {len(result.get('extracted_evidence', [])):<37} │")
        print("└─────────────────────────────────────────────────────────────┘")
        
        # Final Answer
        print("\n" + "═" * 61)
        print("                         FINAL ANSWER")
        print("═" * 61)
        print()
        print(result["final_answer"])
        print()
        print("═" * 61)
        
    except KeyboardInterrupt:
        print("\n\n👋 Stopped by user.")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Critical Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
