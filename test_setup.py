"""
Smoke test for the Pompeii Multimodal Framework.
Verifies that all components initialize correctly.

Usage:
    python test_setup.py
"""
import sys
import os

PASS = "✓"
FAIL = "✗"
WARN = "⚠"

results = []


def check(name, func):
    try:
        ok, msg = func()
        status = PASS if ok else FAIL
        results.append((status, name, msg))
        print(f"  [{status}] {name}: {msg}")
    except Exception as e:
        results.append((FAIL, name, str(e)))
        print(f"  [{FAIL}] {name}: {e}")


# ── Tests ──

def test_python_version():
    v = sys.version_info
    ok = v.major >= 3 and v.minor >= 10
    return ok, f"{v.major}.{v.minor}.{v.micro}"


def test_imports():
    errors = []
    for mod in ["gradio", "ollama", "chromadb", "sentence_transformers",
                 "pdfplumber", "PIL", "cv2", "pandas", "numpy"]:
        try:
            __import__(mod)
        except ImportError:
            errors.append(mod)
    if errors:
        return False, f"Missing: {', '.join(errors)}"
    return True, "All imports OK"


def test_pipeline_imports():
    try:
        from pipeline.vlm import VLMAnalyzer
        from pipeline.dispatcher import AgenticDispatcher
        from pipeline.rag_engine import DualRAGEngine
        from pipeline.refinement import MetadataRefiner
        from pipeline.annotator import ImageAnnotator
        return True, "All pipeline modules loaded"
    except ImportError as e:
        return False, str(e)


def test_ollama_connection():
    import ollama
    from config import OLLAMA_HOST
    try:
        client = ollama.Client(host=OLLAMA_HOST)
        models = client.list()
        names = [m.get("name", m.get("model", "?")) for m in models.get("models", [])]
        return True, f"Connected. Models: {', '.join(names[:5])}"
    except Exception as e:
        return False, f"Cannot connect to Ollama at {OLLAMA_HOST}: {e}"


def test_ollama_vlm():
    import ollama
    from config import VLM_MODEL, OLLAMA_HOST
    try:
        client = ollama.Client(host=OLLAMA_HOST)
        models = client.list()
        names = [m.get("name", m.get("model", "?")) for m in models.get("models", [])]
        # Check if VLM model is available (match by prefix)
        found = any(VLM_MODEL.split(":")[0] in n for n in names)
        if found:
            return True, f"{VLM_MODEL} available"
        return False, f"{VLM_MODEL} not found. Run: ollama pull {VLM_MODEL}"
    except Exception:
        return False, "Cannot check models"


def test_ollama_llm():
    import ollama
    from config import LLM_MODEL, OLLAMA_HOST
    try:
        client = ollama.Client(host=OLLAMA_HOST)
        models = client.list()
        names = [m.get("name", m.get("model", "?")) for m in models.get("models", [])]
        found = any(LLM_MODEL.split(":")[0] in n for n in names)
        if found:
            return True, f"{LLM_MODEL} available"
        return False, f"{LLM_MODEL} not found. Run: ollama pull {LLM_MODEL}"
    except Exception:
        return False, "Cannot check models"


def test_embedding_model():
    from sentence_transformers import SentenceTransformer
    from config import EMBEDDING_MODEL
    try:
        model = SentenceTransformer(EMBEDDING_MODEL)
        vec = model.encode("test amphora Dressel")
        return True, f"{EMBEDDING_MODEL} loaded, dim={len(vec)}"
    except Exception as e:
        return False, str(e)


def test_chromadb():
    from pipeline.rag_engine import DualRAGEngine
    try:
        engine = DualRAGEngine()
        stats = engine.get_store_stats()
        total = sum(stats.values())
        empty = sum(1 for v in stats.values() if v == 0)
        return True, f"{len(stats)} stores initialized, {total} total chunks ({empty} empty)"
    except Exception as e:
        return False, str(e)


def test_knowledge_base_folders():
    from config import DOMAINS
    found = 0
    pdfs = 0
    for domain, paths in DOMAINS.items():
        for db_type, path in paths.items():
            if os.path.isdir(path):
                found += 1
                pdf_count = len(list(
                    f for f in os.listdir(path) if f.lower().endswith(".pdf")
                ))
                pdfs += pdf_count
    return True, f"{found}/6 folders exist, {pdfs} PDFs found"


def test_output_dirs():
    from config import OUTPUT_METADATA, OUTPUT_ANNOTATIONS, OUTPUT_IMAGES
    dirs = [OUTPUT_METADATA, OUTPUT_ANNOTATIONS, OUTPUT_IMAGES]
    ok = all(os.path.isdir(d) for d in dirs)
    return ok, "All output directories exist" if ok else "Some output dirs missing"


# ── Run ──

print("=" * 55)
print("  Pompeii Multimodal Framework — Setup Test")
print("=" * 55)

print("\n[Python & Dependencies]")
check("Python version", test_python_version)
check("Python packages", test_imports)
check("Pipeline modules", test_pipeline_imports)

print("\n[Ollama & Models]")
check("Ollama connection", test_ollama_connection)
check("VLM model", test_ollama_vlm)
check("LLM model", test_ollama_llm)

print("\n[AI Components]")
check("Embedding model", test_embedding_model)
check("ChromaDB stores", test_chromadb)

print("\n[File System]")
check("Knowledge base folders", test_knowledge_base_folders)
check("Output directories", test_output_dirs)

# ── Summary ──
print(f"\n{'=' * 55}")
passed = sum(1 for s, _, _ in results if s == PASS)
failed = sum(1 for s, _, _ in results if s == FAIL)
total = len(results)
print(f"  Results: {passed}/{total} passed, {failed} failed")

if failed == 0:
    print("  🎉 All checks passed! Ready to go.")
    print(f"\n  Next: place PDFs in knowledge_base/ and run:")
    print(f"    python indexer.py")
    print(f"    python app.py")
else:
    print(f"\n  Fix the {failed} issue(s) above before running the app.")

print("=" * 55)
