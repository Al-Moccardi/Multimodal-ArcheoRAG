"""
Agentic Dispatcher — Stage 2: DOMAIN ROUTING.

Routes objects to domains based on visual descriptions from Stage 1.
Since the VLM no longer classifies, the dispatcher uses:
  1. material_category from VLM (ceramic / stone / plaster / etc.)
  2. Visual description keywords
  3. User label hints (if user drew the bbox with a label)
"""

import logging
from config import DOMAINS

logger = logging.getLogger(__name__)

# Keywords found in VISUAL DESCRIPTIONS (not typological names)
DESCRIPTION_KEYWORDS = {
    "ceramics": [
        "ceramic", "pottery", "clay", "terracotta", "earthenware",
        "rim", "handle", "vessel", "sherd", "fragment",
        "fabric", "slip", "glaze", "wheel", "coil",
        "amphora", "bowl", "plate", "jug", "lamp", "pot",
        "fired", "kiln", "stamp", "inscription",
        "reddish", "orange", "buff",  # common ceramic fabric colors
    ],
    "paintings": [
        "fresco", "painting", "mural", "plaster", "pigment",
        "painted", "color", "scene", "figure", "landscape",
        "panel", "border", "frame", "decoration", "ornament",
        "red background", "black background", "white background",
        "architectural motif", "garland", "candelabra", "medallion",
        "mythological", "cupid", "venus", "illusionistic",
        "intonaco", "arriccio",
    ],
    "architecture": [
        "wall", "masonry", "block", "brick", "stone",
        "mortar", "concrete", "tuff", "limestone", "lava",
        "column", "capital", "arch", "vault", "threshold",
        "foundation", "floor", "drain", "pilaster",
        "regular pattern", "diagonal net", "irregular",
        "courses", "quoin", "rendered",
        "structural", "load-bearing",
    ],
}

# material_category → domain mapping
MATERIAL_DOMAIN = {
    "ceramic": "ceramics",
    "plaster": "paintings",
    "stone":   "architecture",
    "metal":   "ceramics",      # metal artifacts often catalogued with ceramics
    "glass":   "ceramics",      # glass vessels too
    "organic": "ceramics",      # bone, wood artifacts
    "mixed":   "architecture",  # mixed materials often structural
}


class AgenticDispatcher:
    """Routes objects to domains using descriptions, not typological labels."""

    def __init__(self, domains: list[str] | None = None):
        self.domains = domains or DOMAINS

    def dispatch(self, objects: list[dict]) -> dict[str, list[dict]]:
        """Assign each object to a domain based on visual description."""
        result = {d: [] for d in self.domains}

        for obj in objects:
            domain = self._classify(obj)
            obj["assigned_domain"] = domain
            result[domain].append(obj)

        return result

    def _classify(self, obj: dict) -> str:
        """Classify based on description + material + user hints."""

        # Collect all text safely
        parts = []
        for key in ("visual_description", "description", "label",
                     "material_category", "user_label_hint", "domain"):
            val = obj.get(key, "")
            if val is None:
                continue
            parts.append(str(val) if not isinstance(val, str) else val)
        text = " ".join(parts).lower()

        scores = {d: 0 for d in self.domains}

        # 1. Keyword scoring from visual description
        for domain, keywords in DESCRIPTION_KEYWORDS.items():
            if domain in self.domains:
                for kw in keywords:
                    if kw in text:
                        scores[domain] += 1

        # 2. material_category boost
        mat = str(obj.get("material_category", "")).lower().strip()
        mapped = MATERIAL_DOMAIN.get(mat)
        if mapped and mapped in self.domains:
            scores[mapped] += 5  # strong signal

        # 3. User label hint boost (if user labeled the bbox)
        hint = str(obj.get("user_label_hint", "")).lower()
        if hint:
            for domain, keywords in DESCRIPTION_KEYWORDS.items():
                if domain in self.domains:
                    for kw in keywords:
                        if kw in hint:
                            scores[domain] += 3  # user knows best

        # 4. key_features boost
        features = obj.get("key_features", [])
        if isinstance(features, list):
            feat_text = " ".join(str(f) for f in features).lower()
            for domain, keywords in DESCRIPTION_KEYWORDS.items():
                if domain in self.domains:
                    for kw in keywords:
                        if kw in feat_text:
                            scores[domain] += 2

        best = max(scores, key=scores.get)
        if scores[best] == 0:
            logger.warning(f"Dispatcher: no signal, defaulting to ceramics")
            return "ceramics"

        logger.debug(f"Dispatcher: scores={scores} → {best}")
        return best
