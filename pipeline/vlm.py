"""
Visual Language Model — Stage 1: PERCEPTION ONLY.

The VLM's job is to DESCRIBE what it sees, NOT to classify.
It should never output typological labels like "Dressel 2-4" or "opus reticulatum".
Instead it outputs rich visual descriptions that downstream RAG + LLM can match.

Example output:
  BAD:  "This is a Dressel 2-4 amphora"
  GOOD: "Ceramic vessel fragment with rounded rim profile, bifid handles attached
         at the neck, cylindrical neck, ovoid body. Fabric is reddish-brown with
         visible dark volcanic inclusions. Surface shows traces of a pale slip."
"""

import json
import base64
import logging
import requests
from pathlib import Path
from PIL import Image

from config import VLM_MODEL

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"


# ─── Diagnostic feature checklist per domain ─────────────────────
# These tell the VLM WHAT to look for, without telling it what things ARE.

OBSERVATION_GUIDE = {
    "ceramics": """For ceramic/pottery objects, describe:
- RIM: shape (triangular, rounded, flared, thickened, everted, rolled), diameter
- HANDLES: type (bifid, ribbon, rod, strap, lug, vertical, horizontal), attachment point
- NECK: shape (cylindrical, conical, absent), length
- BODY: shape (ovoid, globular, piriform, cylindrical, carinated)
- BASE: type (flat, pointed spike, ring foot, disc, concave)
- FABRIC: color, visible inclusions (volcanic, sand, mica, limestone), texture (coarse/fine)
- SURFACE: treatment (slip color, glaze, painted bands, stamped, incised, plain, burnished)
- SIZE: relative (small <15cm, medium 15-50cm, large >50cm)
- CONDITION: complete, fragmentary, which parts are preserved
- STAMPS/INSCRIPTIONS: any visible text, letters, symbols""",

    "paintings": """For wall paintings/frescoes, describe:
- BACKGROUND: dominant color (red, black, white, yellow, blue)
- ZONES: how many horizontal registers, what divides them
- ARCHITECTURAL ELEMENTS: painted columns, pediments, cornices, window frames
- FIGURES: human, divine, animal — posture, clothing, attributes
- OBJECTS: garlands, candelabra, masks, vessels, food, instruments
- LANDSCAPES: architecture, gardens, seascapes, rural scenes
- GEOMETRIC/ORNAMENTAL: borders, frames, arabesques, scrollwork
- ILLUSIONISM: flat or 3D depth, trompe-l'oeil effects, perspective lines
- TECHNIQUE: fresco visible layers, sinopia, giornate seams
- CONDITION: fading, lacunae, salt damage, detachment""",

    "architecture": """For architectural elements/masonry, describe:
- BLOCKS: shape (regular squares, irregular polygons, triangular, rectangular bricks)
- PATTERN: how blocks are arranged (grid, diagonal net, random, herringbone, courses)
- MORTAR: visible between blocks? thickness? color?
- MATERIAL: stone type (grey tuff, yellow tuff, lava, limestone, marble), bricks (red, triangular)
- CORNERS: how are edges/quoins treated (brick chains, large blocks)
- SURFACE: rendered/plastered or exposed masonry
- ARCHES/VAULTS: pointed, semicircular, flat lintel
- COLUMNS: shape (round, square), surface (smooth, fluted), capital style
- DIMENSIONS: block sizes relative to each other, wall thickness
- REPAIRS: different construction phases visible, patches, fillings""",
}

# Generic guide when domain is unknown
GENERIC_GUIDE = """Describe in detail:
- MATERIAL: what is it made of (ceramic, stone, plaster, metal, glass)
- SHAPE: overall form, profile, cross-section
- COLOR: dominant color, patterns, variations
- TEXTURE: smooth, rough, coarse, fine
- SIZE: relative size compared to surroundings
- SURFACE: any decoration, marks, inscriptions, damage
- CONDITION: complete, fragmentary, worn, broken
- CONTEXT: what surrounds it, what it's resting on"""


class VLMAnalyzer:
    """
    Stage 1: Visual perception.
    Outputs DESCRIPTIONS, not classifications.
    """

    def __init__(self, model: str = None):
        self.model = model or VLM_MODEL

    def analyze_image(
        self,
        image_path: str,
        expert_prompt: str,
        user_bboxes: list[dict] | None = None,
    ) -> dict:
        """
        Analyze image → produce rich visual descriptions.
        Does NOT classify or assign typological labels.
        """
        if user_bboxes and len(user_bboxes) > 0:
            return self._describe_user_regions(image_path, expert_prompt, user_bboxes)
        else:
            return self._describe_full_scene(image_path, expert_prompt)

    # ──────────────────────────────────────────
    #  Mode A: Full scene
    # ──────────────────────────────────────────

    def _describe_full_scene(self, image_path: str, expert_prompt: str) -> dict:
        """VLM describes everything visible — no classification."""
        logger.info("VLM Stage 1: full scene description (no classification)")

        prompt = f"""You are an archaeological photographer's assistant. Your job is to DESCRIBE
what you see in this image in precise visual detail. Do NOT classify or name artifact types.
Do NOT use typological labels. Only describe physical characteristics.

Expert context: {expert_prompt}

{GENERIC_GUIDE}

Respond with ONLY valid JSON (no markdown):
{{
  "scene_description": "Overall description of the scene",
  "detected_objects": [
    {{
      "visual_description": "Detailed physical description of this object — shape, material, color, texture, size, condition, decorations",
      "material_category": "ceramic / stone / plaster / metal / glass / organic / mixed",
      "bbox": {{"x": 0, "y": 0, "width": 100, "height": 100}},
      "confidence": 0.85
    }}
  ]
}}

IMPORTANT: In "visual_description", describe ONLY what you SEE. Do not guess names or types.
Describe shapes, colors, textures, patterns, materials, sizes, conditions."""

        raw = self._call_ollama(image_path, prompt)
        result = self._parse_response(raw)

        for obj in result.get("detected_objects", []):
            obj["bbox_source"] = "vlm"
            obj["label"] = obj.get("material_category", "object")  # generic placeholder
            obj.setdefault("description", obj.get("visual_description", ""))
            self._normalize_object(obj)

        return result

    # ──────────────────────────────────────────
    #  Mode B: User bounding boxes
    # ──────────────────────────────────────────

    def _describe_user_regions(
        self, image_path: str, expert_prompt: str, user_bboxes: list[dict]
    ) -> dict:
        """Describe each user-defined region in detail."""
        logger.info(f"VLM Stage 1: describing {len(user_bboxes)} user regions")

        image = Image.open(image_path).convert("RGB")
        all_objects = []

        for i, bbox in enumerate(user_bboxes):
            x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
            user_label = bbox.get("label", "")

            # Pick the right observation guide based on user label hint
            guide = self._pick_guide(user_label)

            crop = image.crop((x, y, x + w, y + h))
            crop_path = Path(image_path).parent / f"_crop_{i}.jpg"
            crop.save(str(crop_path), quality=90)

            label_hint = ""
            if user_label:
                label_hint = f"\nThe expert thinks this might be related to: {user_label}. But describe only what you SEE."

            prompt = f"""You are an archaeological photographer's assistant examining a cropped region.
DESCRIBE what you see in precise visual detail. Do NOT classify or assign type names.
Only describe physical characteristics: shapes, colors, textures, patterns, materials.
{label_hint}

Expert context: {expert_prompt}

{guide}

Respond with ONLY valid JSON (no markdown):
{{
  "visual_description": "Detailed physical description — shapes, colors, textures, patterns, materials, sizes, condition",
  "material_category": "ceramic / stone / plaster / metal / glass / organic / mixed",
  "key_features": ["list", "of", "distinctive", "visual", "features"],
  "condition": "complete / fragmentary / damaged / worn",
  "confidence": 0.9
}}

REMEMBER: Describe ONLY what you see. No type names, no classifications."""

            raw = self._call_ollama(str(crop_path), prompt)
            obj = self._parse_single(raw, i)

            obj["bbox"] = {"x": x, "y": y, "width": w, "height": h}
            obj["bbox_source"] = "user"
            obj["user_label_hint"] = user_label  # keep the user's hint for RAG
            obj["label"] = user_label if user_label else obj.get("material_category", f"object_{i}")
            obj.setdefault("description", obj.get("visual_description", ""))
            self._normalize_object(obj)

            all_objects.append(obj)

            try:
                crop_path.unlink()
            except OSError:
                pass

        scene_desc = f"Detailed description of {len(user_bboxes)} regions."
        return {"scene_description": scene_desc, "detected_objects": all_objects}

    # ──────────────────────────────────────────
    #  Guide selector
    # ──────────────────────────────────────────

    def _pick_guide(self, user_label: str) -> str:
        """Pick the observation guide based on user label hint."""
        if not user_label:
            return GENERIC_GUIDE
        l = user_label.lower()
        if any(k in l for k in ["amphora", "pottery", "ceramic", "vessel", "bowl", "plate", "jug", "dressel", "lucerna", "dolium"]):
            return OBSERVATION_GUIDE["ceramics"]
        if any(k in l for k in ["fresco", "painting", "style", "mural", "scene"]):
            return OBSERVATION_GUIDE["paintings"]
        if any(k in l for k in ["opus", "wall", "column", "arch", "vault", "masonry", "foundation"]):
            return OBSERVATION_GUIDE["architecture"]
        return GENERIC_GUIDE

    # ──────────────────────────────────────────
    #  Ollama
    # ──────────────────────────────────────────

    def _call_ollama(self, image_path: str, prompt: str) -> str:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
            "options": {"temperature": 0.15, "num_predict": 2048},
        }

        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json().get("response", "")
        except requests.ConnectionError:
            raise RuntimeError("Ollama not reachable at localhost:11434. Start with: ollama serve")
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            raise

    # ──────────────────────────────────────────
    #  Parsing
    # ──────────────────────────────────────────

    def _parse_response(self, raw: str) -> dict:
        try:
            data = json.loads(self._extract_json(raw))
            data.setdefault("detected_objects", [])
            data.setdefault("scene_description", "")
            for obj in data["detected_objects"]:
                obj.setdefault("bbox", {"x": 0, "y": 0, "width": 50, "height": 50})
                obj.setdefault("confidence", 0.5)
            return data
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"VLM parse failed: {e}")
            return {"scene_description": raw[:200], "detected_objects": []}

    def _parse_single(self, raw: str, index: int) -> dict:
        try:
            obj = json.loads(self._extract_json(raw))
            obj.setdefault("visual_description", "")
            obj.setdefault("confidence", 0.8)
            return obj
        except (json.JSONDecodeError, KeyError):
            return {"visual_description": raw[:300] if raw else "", "confidence": 0.5}

    @staticmethod
    def _normalize_object(obj: dict):
        for key in ("description", "visual_description", "label", "material_category"):
            val = obj.get(key)
            if val is not None and not isinstance(val, str):
                obj[key] = json.dumps(val) if isinstance(val, (dict, list)) else str(val)
        if "bbox" not in obj or not isinstance(obj.get("bbox"), dict):
            obj["bbox"] = {"x": 0, "y": 0, "width": 50, "height": 50}
        conf = obj.get("confidence")
        if conf is None:
            obj["confidence"] = 0.5
        elif not isinstance(conf, (int, float)):
            try:
                obj["confidence"] = float(conf)
            except (ValueError, TypeError):
                obj["confidence"] = 0.5
        # Ensure key_features is a list of strings
        kf = obj.get("key_features")
        if kf and isinstance(kf, list):
            obj["key_features"] = [str(f) for f in kf]
        elif kf and isinstance(kf, str):
            obj["key_features"] = [kf]
        else:
            obj["key_features"] = []

    @staticmethod
    def _extract_json(text: str) -> str:
        text = text.strip()
        if "```" in text:
            for part in text.split("```"):
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    return part
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            return text[start:end]
        return text
