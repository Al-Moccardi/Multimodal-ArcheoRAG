"""
LLM Refinement — Stage 3: CLASSIFICATION + CATALOGUING.

This is where the actual archaeological expertise happens.
The LLM receives:
  1. Rich visual description from VLM (Stage 1)
  2. Historical context chunks from RAG
  3. Cataloguing reference chunks from RAG (type descriptions, diagnostic features)
  4. Dimension estimates from bbox

Its job: MATCH the visual description against known types in the RAG context,
then produce structured catalogue metadata.

This is the "archaeologist's brain" — it compares what was seen (VLM)
with what is known (RAG) to identify and classify.
"""

import json
import logging
import requests
from datetime import datetime
from pathlib import Path

from config import LLM_MODEL, OUTPUT_METADATA

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"


# ─── Incremental IDs ────────────────────────────────────────────

COUNTER_FILE = OUTPUT_METADATA / "catalogue_counters.json"

DOMAIN_PREFIXES = {
    "ceramics":     "PMP-CER",
    "paintings":    "PMP-PAI",
    "architecture": "PMP-ARC",
}


def _load_counters() -> dict:
    if COUNTER_FILE.exists():
        try:
            return json.loads(COUNTER_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {d: 0 for d in DOMAIN_PREFIXES}


def _save_counters(counters: dict):
    COUNTER_FILE.write_text(json.dumps(counters, indent=2))


def next_catalogue_id(domain: str) -> str:
    counters = _load_counters()
    key = domain if domain in counters else "ceramics"
    counters[key] = counters.get(key, 0) + 1
    _save_counters(counters)
    prefix = DOMAIN_PREFIXES.get(domain, "PMP-UNK")
    return f"{prefix}-{counters[key]:05d}"


# ─── Dimension estimation ────────────────────────────────────────

def estimate_dimensions(bbox: dict, domain: str) -> dict:
    px_w = bbox.get("width", 0)
    px_h = bbox.get("height", 0)
    if px_w == 0 or px_h == 0:
        return {"estimated": False}

    scale = 0.8  # mm per pixel heuristic
    est_w_mm = round(px_w * scale)
    est_h_mm = round(px_h * scale)

    result = {
        "estimated": True,
        "method": "bbox_pixel_ratio",
        "bbox_px": {"width": px_w, "height": px_h},
    }
    if domain == "ceramics":
        result.update({"est_height_mm": est_h_mm, "est_width_mm": est_w_mm, "est_rim_diameter_mm": est_w_mm})
    else:
        result.update({"est_height_cm": round(est_h_mm / 10, 1), "est_width_cm": round(est_w_mm / 10, 1)})
    return result


# ─── Domain schemas ──────────────────────────────────────────────

DOMAIN_SCHEMAS = {
    "ceramics": {
        "catalogue_id": "auto",
        "classification": {
            "typology": "e.g., Dressel 1, Dressel 2-4, Morel 2621",
            "form": "amphora / bowl / plate / lamp / jug / dolium / unguentarium",
            "type_confidence": "high / medium / low",
            "diagnostic_features_matched": "which visual features matched the type"
        },
        "fabric": {
            "color": "Munsell or description",
            "inclusions": "volcanic / calcareous / micaceous",
            "texture": "coarse / medium / fine",
            "hardness": "soft / medium / hard"
        },
        "technique": "wheel-thrown / mold-made / hand-built / coil-built",
        "surface_treatment": "slip / glaze / painted / burnished / plain",
        "dimensions": {
            "height_mm": "estimate", "width_mm": "estimate",
            "rim_diameter_mm": "estimate", "base_diameter_mm": "estimate",
            "wall_thickness_mm": "estimate"
        },
        "weight_g": "estimate from dimensions",
        "dating": {"period": "", "date_range": "", "method": "", "evidence": ""},
        "provenance": {"production_area": "", "workshop": "", "find_context": "Pompeii"},
        "conservation_state": "",
        "fragment_percentage": "",
        "decoration": "",
        "stamps_inscriptions": "",
        "comparanda": "",
        "functional_interpretation": "",
        "notes": ""
    },
    "paintings": {
        "catalogue_id": "auto",
        "classification": {
            "mau_style": "I / II / III / IV / transitional",
            "style_name": "incrustation / architectural / ornamental / intricate",
            "type_confidence": "high / medium / low",
            "diagnostic_features_matched": "which visual features matched the style"
        },
        "subject": {
            "category": "mythological / landscape / still life / portrait / decorative",
            "description": "",
            "figures": "",
            "narrative_source": ""
        },
        "technique": {"primary": "buon fresco / secco / mixed", "layers": "", "preparation": ""},
        "palette": {"dominant_colors": [], "pigments": "", "scheme": "polychrome / monochrome"},
        "composition": {"layout": "", "perspective": "", "registers": ""},
        "wall_zone": "",
        "dimensions": {"height_cm": "estimate", "width_cm": "estimate"},
        "dating": {"period": "", "date_range": "", "method": "", "evidence": ""},
        "location": {"building": "", "room": "", "wall": "", "position": "in situ / detached"},
        "conservation_state": "",
        "damage": "",
        "comparanda": "",
        "notes": ""
    },
    "architecture": {
        "catalogue_id": "auto",
        "classification": {
            "opus_type": "incertum / quasi-reticulatum / reticulatum / testaceum / mixtum / vittatum / signinum / quadratum / caementicium / craticium",
            "type_confidence": "high / medium / low",
            "diagnostic_features_matched": "which visual features matched the opus type"
        },
        "materials": {"primary": "", "mortar": "", "secondary": ""},
        "architectural_order": "doric / ionic / corinthian / tuscan / composite / none",
        "element_type": "wall / column / pilaster / arch / vault / threshold / foundation / floor",
        "construction_technique": {"facing": "", "core": "", "bonding": ""},
        "dimensions": {"height_cm": "estimate", "width_cm": "estimate", "thickness_cm": "", "block_size_cm": ""},
        "building_phase": {"phase": "", "modifications": "", "stratigraphic_relations": ""},
        "dating": {"period": "", "date_range": "", "method": "", "evidence": ""},
        "structural_condition": "",
        "seismic_damage": "",
        "comparanda": "",
        "functional_interpretation": "",
        "notes": ""
    },
}


# ─── Refiner ─────────────────────────────────────────────────────

class MetadataRefiner:
    """
    Stage 3: Classification + Cataloguing.
    Compares VLM visual description against RAG reference types.
    """

    def __init__(self, model: str = None):
        self.model = model or LLM_MODEL

    def refine(self, obj: dict, domain: str, rag_results: dict) -> dict:
        """
        Classify and catalogue one object.

        The LLM acts as an archaeologist:
          - Reads the visual description (what was seen)
          - Reads the RAG context (what is known about types)
          - Matches features to known types
          - Produces structured metadata
        """
        cat_id = next_catalogue_id(domain)
        schema = DOMAIN_SCHEMAS.get(domain, DOMAIN_SCHEMAS["ceramics"])
        bbox = obj.get("bbox", {})
        dim_est = estimate_dimensions(bbox, domain)

        # Collect all visual info
        vis_desc = obj.get("visual_description", obj.get("description", ""))
        if not isinstance(vis_desc, str):
            vis_desc = json.dumps(vis_desc) if isinstance(vis_desc, (dict, list)) else str(vis_desc)

        key_features = obj.get("key_features", [])
        if isinstance(key_features, list):
            features_text = ", ".join(str(f) for f in key_features)
        else:
            features_text = str(key_features)

        user_hint = obj.get("user_label_hint", "")

        historical = "\n\n".join(rag_results.get("historical_context", []))
        cataloguing = "\n\n".join(rag_results.get("cataloguing_rules", []))

        prompt = self._build_prompt(
            domain=domain,
            schema=schema,
            vis_desc=vis_desc,
            features_text=features_text,
            user_hint=user_hint,
            dim_est=dim_est,
            historical=historical or "[No historical context — use your expertise]",
            cataloguing=cataloguing or "[No cataloguing rules — use standard practice]",
        )

        metadata = self._call_llm(prompt)

        # Inject pipeline fields
        metadata["catalogue_id"] = cat_id
        metadata["_domain"] = domain
        metadata["_vlm_visual_description"] = vis_desc
        metadata["_vlm_key_features"] = key_features
        metadata["_user_label_hint"] = user_hint
        metadata["_bbox"] = bbox
        metadata["_bbox_source"] = obj.get("bbox_source", "vlm")
        metadata["_vlm_confidence"] = obj.get("confidence", 0.0)
        metadata["_dimension_estimate"] = dim_est
        metadata["_generated_at"] = datetime.now().isoformat()
        metadata["_model"] = self.model

        return metadata

    def _build_prompt(
        self, domain: str, schema: dict, vis_desc: str, features_text: str,
        user_hint: str, dim_est: dict, historical: str, cataloguing: str,
    ) -> str:
        schema_json = json.dumps(schema, indent=2, ensure_ascii=False)
        dim_json = json.dumps(dim_est, indent=2)

        hint_section = ""
        if user_hint:
            hint_section = f"""
=== USER HINT ===
The archaeologist suggested this might be: {user_hint}
Consider this as a hypothesis to verify against the visual evidence and references."""

        return f"""You are a senior Pompeian archaeologist doing TYPOLOGICAL CLASSIFICATION.

You have TWO sources of information:

SOURCE 1 — VISUAL DESCRIPTION (what the camera saw):
"{vis_desc}"

Key visual features observed: {features_text}
{hint_section}

SOURCE 2 — REFERENCE DATABASE (what is known about {domain} types):

--- Historical context ---
{historical[:4000]}

--- Cataloguing reference (type descriptions, diagnostic features) ---
{cataloguing[:4000]}

=== YOUR TASK ===
STEP 1 — MATCH: Compare the visual features in Source 1 against the type descriptions
in Source 2. Which known type best matches what was observed? Explain which specific
features match (e.g., "triangular rim + ribbon handles → Dressel 1").

STEP 2 — CLASSIFY: Assign the most likely classification. State your confidence
(high/medium/low) and list the diagnostic features that matched.

STEP 3 — CATALOGUE: Fill the complete metadata schema below.

=== DIMENSION ESTIMATES FROM IMAGE ===
{dim_json}
Adjust these based on your knowledge of typical {domain} dimensions.

=== SCHEMA TO FILL (every field, never null, use "est." prefix for estimates) ===
{schema_json}

CRITICAL RULES:
1. The "classification" section is THE MOST IMPORTANT. Match visual features to types.
2. In "diagnostic_features_matched", list WHICH visual features led to the classification.
3. NEVER leave a field null or empty. Estimate with "est." prefix if unsure.
4. For dating, use typological dating based on your classification.
5. For comparanda, cite specific parallels from the reference database if possible.
6. Put your reasoning and any uncertainties in "notes".

Respond with ONLY valid JSON. No markdown, no backticks."""

    def _call_llm(self, prompt: str) -> dict:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.15, "num_predict": 4096},
        }
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
            resp.raise_for_status()
            raw = resp.json().get("response", "{}")
            return self._parse_json(raw)
        except requests.ConnectionError:
            logger.error("Ollama not reachable for refinement.")
            return {"error": "Ollama not reachable"}
        except Exception as e:
            logger.error(f"Refinement LLM error: {e}")
            return {"error": str(e)}

    @staticmethod
    def _parse_json(raw: str) -> dict:
        raw = raw.strip()
        if "```" in raw:
            for part in raw.split("```"):
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    raw = part
                    break
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass
        return {"raw_response": raw[:500]}
