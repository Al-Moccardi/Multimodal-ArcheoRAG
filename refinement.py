"""
LLM Refinement module.
Merges historical context + cataloguing schema into structured JSON metadata.
The output conforms to domain-specific schemas.
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Domain-specific JSON schemas
# ──────────────────────────────────────────────

DOMAIN_SCHEMAS = {
    "ceramics": {
        "type": "ceramics",
        "fields": {
            "typology": "Dressel/Morel type classification (e.g., Dressel 2-4)",
            "form": "Vessel form (amphora, plate, bowl, lamp, etc.)",
            "fabric": "Clay fabric description (color, inclusions, texture)",
            "technique": "Manufacturing technique (wheel-thrown, mold-made, etc.)",
            "surface_treatment": "Slip, glaze, burnish, paint",
            "dimensions": {"rim_diameter_mm": None, "base_diameter_mm": None, "height_mm": None, "wall_thickness_mm": None},
            "dating": {"period": None, "date_range": None, "dating_evidence": None},
            "provenance": {"production_area": None, "find_context": None},
            "conservation_state": "Good / Fair / Poor / Fragmentary",
            "comparanda": "References to similar published examples",
            "notes": "Additional observations"
        }
    },
    "paintings": {
        "type": "paintings",
        "fields": {
            "style": "Mau style classification (I-IV)",
            "style_name": "Incrustation / Architectural / Ornamental / Intricate",
            "subject": "Iconographic subject (mythological, landscape, still life, etc.)",
            "technique": "Fresco / secco / mixed",
            "palette": ["List of identified pigments/colors"],
            "composition": "Description of compositional layout",
            "dimensions": {"height_cm": None, "width_cm": None},
            "wall_zone": "Upper / Middle / Lower / Predella / Lunette",
            "dating": {"period": None, "date_range": None, "dating_evidence": None},
            "location": {"building": None, "room": None, "wall": None},
            "conservation_state": "Good / Fair / Poor / Fragmentary",
            "comparanda": "References to similar published frescoes",
            "notes": "Additional observations"
        }
    },
    "architecture": {
        "type": "architecture",
        "fields": {
            "opus_type": "Opus classification (incertum, reticulatum, testaceum, mixtum, etc.)",
            "materials": ["List of building materials identified"],
            "architectural_order": "Doric / Ionic / Corinthian / Composite / None",
            "element_type": "Wall / Column / Arch / Vault / Threshold / Foundation",
            "construction_technique": "Description of building method",
            "dimensions": {"height_cm": None, "width_cm": None, "thickness_cm": None},
            "building_phase": "Construction phase within the building's history",
            "dating": {"period": None, "date_range": None, "dating_evidence": None},
            "structural_condition": "Intact / Partially collapsed / Ruined",
            "modifications": "Evidence of repairs, rebuilding, or reuse",
            "comparanda": "References to similar published structures",
            "notes": "Additional observations"
        }
    }
}


class LLMRefinement:
    """
    Merges dual-RAG context into structured metadata.
    
    Takes:
      - Object label + description from VLM
      - Historical context (from historical vector DB)
      - Cataloguing info (from cataloguing vector DB)
      - Expert prompt
      - Domain-specific JSON schema
      
    Produces:
      - Structured JSON metadata conforming to the domain schema
    """

    def __init__(self, model_name: str = "mistral-7b-instruct"):
        self.model_name = model_name
        self.model = None

    def _load_model(self):
        """Lazy-load the refinement LLM."""
        if self.model is not None:
            return

        logger.info(f"Loading refinement LLM: {self.model_name}...")

        # ── Option A: Local model via transformers ──
        # from transformers import AutoTokenizer, AutoModelForCausalLM
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        # ── Option B: vLLM for fast inference ───────
        # from vllm import LLM
        # self.model = LLM(model=self.model_name)

        # ── Option C: OpenAI API ────────────────────
        # from openai import OpenAI
        # self.model = OpenAI()

        logger.info("Refinement LLM loaded (placeholder).")

    def refine(
        self,
        domain: str,
        object_label: str,
        object_description: str,
        historical_context: str,
        cataloguing_info: str,
        expert_prompt: str,
        bbox_source: str = "vlm"
    ) -> dict:
        """
        Produce structured metadata by merging dual-RAG context.
        
        Args:
            domain: Archaeological domain (ceramics, paintings, architecture).
            object_label: Object label from VLM.
            object_description: Object description from VLM.
            historical_context: Retrieved text from historical vector DB.
            cataloguing_info: Retrieved text from cataloguing vector DB.
            expert_prompt: Original expert prompt.
            bbox_source: "user" or "vlm" — included in metadata.
            
        Returns:
            Structured JSON metadata dict conforming to the domain schema.
        """
        self._load_model()

        schema = DOMAIN_SCHEMAS.get(domain, DOMAIN_SCHEMAS["ceramics"])

        prompt = self._build_refinement_prompt(
            domain=domain,
            schema=schema,
            object_label=object_label,
            object_description=object_description,
            historical_context=historical_context,
            cataloguing_info=cataloguing_info,
            expert_prompt=expert_prompt
        )

        # ── Call LLM ────────────────────────────
        # response = self.model.chat.completions.create(
        #     model=self.model_name,
        #     messages=[{"role": "user", "content": prompt}],
        #     response_format={"type": "json_object"},
        #     temperature=0.1,
        #     max_tokens=2048
        # )
        # metadata = json.loads(response.choices[0].message.content)

        # ── Placeholder ─────────────────────────
        logger.info(f"Refinement LLM: generating {domain} metadata for '{object_label}'...")
        metadata = self._placeholder_metadata(domain, object_label, object_description, schema)

        # Add pipeline metadata
        metadata["_pipeline"] = {
            "domain": domain,
            "bbox_source": bbox_source,
            "vlm_label": object_label,
            "schema_version": "1.0"
        }

        return metadata

    def _build_refinement_prompt(
        self,
        domain: str,
        schema: dict,
        object_label: str,
        object_description: str,
        historical_context: str,
        cataloguing_info: str,
        expert_prompt: str
    ) -> str:
        schema_json = json.dumps(schema["fields"], indent=2, ensure_ascii=False)

        return f"""You are an expert archaeological cataloguer specializing in {domain}.

Your task: produce structured metadata for an archaeological artifact by merging
information from TWO sources:

1) HISTORICAL CONTEXT (art history, dating, trade routes, cultural significance):
{historical_context}

2) CATALOGUING RULES (classification manual, field definitions, normative schema):
{cataloguing_info}

ARTIFACT INFORMATION:
- Label: {object_label}
- Visual description: {object_description}
- Expert notes: {expert_prompt}

OUTPUT SCHEMA (fill every field, use null if unknown):
{schema_json}

RULES:
- Every field must be populated based on evidence from the two sources above.
- If the historical context suggests a date but the cataloguing manual specifies 
  a different format, use the cataloguing format.
- "comparanda" must cite specific published parallels if found in the sources.
- Be conservative: if uncertain, say so in the "notes" field.

Respond with ONLY valid JSON conforming to the schema above."""

    def _placeholder_metadata(
        self, domain: str, label: str, description: str, schema: dict
    ) -> dict:
        """Generate placeholder metadata for testing."""
        base = {}
        for key, value in schema["fields"].items():
            if isinstance(value, dict):
                base[key] = {k: "unknown" for k in value}
            elif isinstance(value, list):
                base[key] = []
            else:
                base[key] = f"[placeholder for {key}]"
        
        base["notes"] = f"Auto-generated placeholder for {label}: {description}"
        return base
