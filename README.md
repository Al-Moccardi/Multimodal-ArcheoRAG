# 🏛 Pompeii Multimodal Archaeological Framework

A multimodal AI pipeline for intelligent analysis and semantic enrichment of archaeological artifacts, with a case study on the archaeological site of Pompeii.

## Architecture

```
Input (Image + Expert Prompt)
        │
        ▼
┌──────────────────┐
│  Visual Language  │  ← LLaVA 7B via Ollama
│     Model (VLM)   │
└────────┬─────────┘
         │  Multi-object detection + interpretation
         ▼
┌──────────────────┐
│    Agentic        │  ← Keyword matching + LLM fallback
│   Dispatcher      │
└──┬──────┬──────┬─┘
   │      │      │
   ▼      ▼      ▼     1 object = 1 instance
┌──────┐┌──────┐┌──────┐
│Ceram.││Paint.││Arch. │  Each instance has DUAL RAG:
│      ││      ││      │  • Historical Vector DB (context, dating)
│ Hist ││ Hist ││ Hist │  • Cataloguing Vector DB (classification rules)
│ Cat  ││ Cat  ││ Cat  │
└──┬───┘└──┬───┘└──┬───┘
   │       │       │     Merge + LLM Refinement
   ▼       ▼       ▼
┌──────────────────────┐
│   Structured Output   │
│  • JSON Metadata      │  → PostgreSQL (per domain)
│  • COCO Annotations   │  → File storage
│  • Annotated Images   │  → S3 / local storage
└──────────────────────┘
         │
         ▼
┌──────────────────────┐
│ Cross-Reference Index │  Links co-occurring objects
└──────────────────────┘
```

## Quick Start

### 1. Prerequisites

- **Python 3.10+**
- **Ollama** installed and running ([ollama.ai](https://ollama.ai))

### 2. Install Ollama Models

```bash
# Visual Language Model
ollama pull llava:7b

# Text LLM for dispatch + refinement
ollama pull mistral:7b
```

### 3. Install Python Dependencies

```bash
cd pompeii-multimodal
pip install -r requirements.txt
```

### 4. Add Your PDFs to the Knowledge Base

Place your archaeological PDFs in the correct folders:

```
knowledge_base/
├── ceramics/
│   ├── historical/      ← Art history, trade, dating (Peacock, Opait, Olcese...)
│   └── cataloguing/     ← Dressel typology, classification manuals
├── paintings/
│   ├── historical/      ← Cuní, Springer, Bergmann, OAPEN monograph...
│   └── cataloguing/     ← Mau styles, RePAIR dataset, pigment analysis...
└── architecture/
    ├── historical/      ← Strickland thesis, opus craticium, concrete origins...
    └── cataloguing/     ← Opus techniques, PompeiiOnline, House of Arianna...
```

### 5. Index the PDFs

```bash
# Index all domains
python indexer.py

# Or index a specific domain
python indexer.py ceramics
python indexer.py paintings historical
```

### 6. Launch the Application

```bash
python app.py
```

Open your browser at **http://localhost:7860**

## Usage

1. **Upload** an archaeological image (photo from excavation, museum, etc.)
2. **Write** an expert prompt describing the context (excavation area, what to look for)
3. **Click** "Run Analysis"
4. **Review** the annotated image with bounding boxes color-coded by domain
5. **Inspect** the structured JSON metadata for each detected object
6. **Download** COCO JSON, CSV, or metadata JSON files

## Project Structure

```
pompeii-multimodal/
├── app.py                     # Gradio web interface
├── config.py                  # Configuration (models, paths, settings)
├── indexer.py                 # PDF → vector store indexer
├── requirements.txt
├── pipeline/
│   ├── vlm.py                 # Visual Language Model (Ollama + LLaVA)
│   ├── dispatcher.py          # Agentic domain router
│   ├── rag_engine.py          # Dual RAG with ChromaDB
│   ├── refinement.py          # LLM metadata refinement
│   └── annotator.py           # Image annotation + COCO/CSV export
├── knowledge_base/            # Place PDFs here (6 subfolders)
├── vector_stores/             # ChromaDB persistent stores (auto-created)
├── schemas/                   # Domain JSON schemas
└── outputs/                   # Generated results
    ├── metadata/              # Structured JSON per analysis
    ├── annotations/           # COCO JSON + CSV files
    └── images/                # Annotated images with bounding boxes
```

## Domain Schemas

### Ceramics
`typology`, `form`, `fabric`, `technique`, `surface_treatment`, `decoration`,
`estimated_date`, `provenance_region`, `functional_category`, `conservation_state`

### Paintings
`mau_style` (I-IV), `subject_type`, `iconographic_elements`, `color_palette`,
`technique`, `wall_zone`, `estimated_date`, `conservation_state`, `comparanda`

### Architecture
`opus_type`, `structural_element`, `architectural_order`, `materials`,
`construction_phase`, `estimated_date`, `building_type`, `conservation_state`

## Technology Stack

| Component | Technology |
|-----------|-----------|
| VLM | LLaVA 7B via Ollama |
| LLM | Mistral 7B via Ollama |
| Vector Store | ChromaDB (persistent) |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) |
| PDF Processing | pdfplumber |
| Image Annotation | Pillow |
| UI | Gradio |
| Annotation Format | COCO JSON, CSV |

## Configuration

Edit `config.py` to change:
- **VLM_MODEL**: default `llava:7b` (alternatives: `llava:13b`, `bakllava`)
- **LLM_MODEL**: default `mistral:7b` (alternatives: `llama3:8b`, `phi3:mini`)
- **EMBEDDING_MODEL**: default `all-MiniLM-L6-v2`
- **CHUNK_SIZE / CHUNK_OVERLAP**: RAG chunking parameters
- **TOP_K_RETRIEVAL**: number of chunks retrieved per query

## License

This project is for academic research purposes.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{pompeii_multimodal_2025,
  title={A Multimodal Framework for Intelligent Analysis and Semantic
         Enrichment of Archaeological Artifacts},
  year={2025},
  note={Case study: Archaeological site of Pompeii}
}
```
