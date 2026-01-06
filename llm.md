# Silas V2 - Hierarchical RAG Agent

## Concept

Silas V2 utilise le pattern **Hierarchical RAG**.
Au lieu d'analyser des fichiers entiers, on découpe en **chunks** pour une précision chirurgicale.

## Différence V1 vs V2

| Aspect | V1 | V2 |
|--------|----|----|
| Unité de traitement | Fichier entier | Chunk (segment) |
| Sélection | Fichiers pertinents | Chunks pertinents |
| Précision | Moyenne (bruit) | Haute (ciblée) |
| Gros documents | Problématique | Optimisé |

## Architecture

```
silas-v2/
├── config.py      # Configuration (modèles + chunking)
├── chunker.py     # Découpage intelligent des documents
├── prompts.py     # Prompts système
├── rag_graph.py   # Pipeline LangGraph
├── main.py        # Point d'entrée CLI
└── llm.md         # Cette documentation
```

## Flux de traitement (5 étapes)

```
[Dossier de docs]
       ↓
   ① CHUNKER (Code Python - pas de LLM)
       → Découpe chaque doc en segments
       → Respecte les frontières naturelles (paragraphes, phrases)
       → Génère: doc1_s1, doc1_s2, doc2_s1...
       ↓
   ② SCANNER (FAST_MODEL: qwen3:0.6b)
       → Résume chaque chunk en 1 phrase
       → Traitement parallèle possible
       ↓
   ③ SELECTOR (CHOOSE_MODEL: qwen3:8b)
       → Reçoit: question + tous les résumés
       → Retourne: ["doc1_s3", "doc2_s7"] (chunks pertinents)
       ↓
   ④ EXTRACTOR (SMART_MODEL: qwen3:14b)
       → Lit UNIQUEMENT les chunks sélectionnés
       → Extrait les passages qui répondent
       ↓
   ⑤ SYNTHESIZER (SMART_MODEL: qwen3:14b)
       → Rédige la réponse finale avec citations
       ↓
[Réponse avec [chunk_id : evidence]]
```

## Chunking Intelligent

Le chunker (`chunker.py`) découpe sans LLM en utilisant des heuristiques:

### Paramètres (config.py)

- `CHUNK_SIZE = 1500` : Taille cible en caractères
- `CHUNK_OVERLAP = 200` : Chevauchement pour préserver le contexte
- `MIN_CHUNK_SIZE = 300` : Évite les micro-chunks inutiles

### Séparateurs (ordre de priorité)

1. `\n\n\n` - Triple saut (section majeure)
2. `\n\n` - Double saut (paragraphe)
3. `\n` - Simple saut (ligne)
4. `. ` - Fin de phrase
5. `, ` - Virgule
6. ` ` - Espace (dernier recours)

### Exemple de découpage

```
Document: rapport.md (4500 chars)
→ rapport.md_s1 (1450 chars) - Introduction
→ rapport.md_s2 (1500 chars) - Analyse
→ rapport.md_s3 (1550 chars) - Conclusions
```

## Les 3 modèles

| Modèle | Alias | Tâche | Justification |
|--------|-------|-------|---------------|
| `qwen3:0.6b` | FAST_MODEL | Scan/résumé | Ultra-rapide, traite des dizaines de chunks |
| `qwen3:8b` | CHOOSE_MODEL | Sélection logique | Bon raisonnement, coût modéré |
| `qwen3:14b` | SMART_MODEL | Extraction + rédaction | Précision maximale |

## Structure de l'état (StateGraph)

```python
class AgentState(TypedDict):
    question: str                   # Question utilisateur
    file_directory: str             # Chemin du dossier
    chunks: List[dict]              # Chunks sérialisés
    chunk_summaries: List[str]      # ["chunk_id: résumé", ...]
    selected_chunks: List[str]      # ["doc1_s3", "doc2_s7"]
    extracted_evidence: List[str]   # Passages extraits
    final_answer: str               # Réponse finale
    timings: dict                   # Temps par étape
```

## Utilisation CLI

```bash
# Usage basique
python main.py -q "Quelle est la conclusion du rapport?" -d ./documents/

# Mode verbeux
python main.py -q "Résume les points clés" -d ./projet/ -v
```

## Sortie console

```
╔═══════════════════════════════════════════════════════════════╗
║                   SILAS V2 - Hierarchical RAG                 ║
╚═══════════════════════════════════════════════════════════════╝

┌─ Configuration ────────────────────────────────────────────┐
│  🐇 FAST Model : qwen3:0.6b                                │
│  ⚖️  CHOOSE     : qwen3:8b                                  │
│  🧠 SMART      : qwen3:14b                                 │
└────────────────────────────────────────────────────────────┘

============================================================
✂️ ÉTAPE 1: CHUNKING
============================================================
  • Fichiers traités: 3
  • Chunks générés: 12

============================================================
🔍 ÉTAPE 2: SCAN
============================================================
  [████████████████████] 12/12 (100%)

... (autres étapes)

┌─ Temps d'exécution ────────────────────────────────────────┐
│  Chunking     ████░░░░░░░░░░░░░░░░   0.02s ( 0.5%)        │
│  Scanning     ████████░░░░░░░░░░░░   3.45s (35.2%)        │
│  Selection    ██░░░░░░░░░░░░░░░░░░   0.89s ( 9.1%)        │
│  Extraction   ████████████░░░░░░░░   4.21s (43.0%)        │
│  Synthesis    ████░░░░░░░░░░░░░░░░   1.19s (12.2%)        │
│  ─────────────────────────────────────────────────────    │
│  TOTAL                               9.76s                │
└────────────────────────────────────────────────────────────┘
```

## Points d'attention

### Modifier le chunking
- `CHUNK_SIZE` dans `config.py` pour ajuster la granularité
- `CHUNK_OVERLAP` pour plus/moins de contexte partagé

### Changer un modèle
- Modifier `FAST_MODEL`, `CHOOSE_MODEL` ou `SMART_MODEL` dans `config.py`

### Ajouter une extension
- Modifier `SUPPORTED_EXTENSIONS` dans `config.py`

### Debug du chunking
```bash
python chunker.py ./mon_dossier/
```

## Optimisations possibles

### Fusion Scanner + Selector
Si le CHOOSE_MODEL est assez bon, on peut fusionner les étapes 2 et 3:
```
"Voici 12 chunks. Lesquels répondent à : {question} ?"
→ ["doc1_s3", "doc2_s7"]
```
Activer via `SELECTOR_DIRECT_PROMPT` dans `prompts.py`.

### Parallélisation
Le scan (étape 2) peut être parallélisé avec `asyncio` pour les gros corpus.

### Cache des résumés
Stocker les résumés de chunks déjà scannés pour éviter de les recalculer.

## Dépendances

```
langchain-ollama
langgraph
```

Ollama doit tourner sur `http://localhost:11434`.
