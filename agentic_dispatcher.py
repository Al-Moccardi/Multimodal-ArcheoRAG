"""
Agentic Dispatcher.
Routes each detected object to the correct archaeological domain.
Uses LLM reasoning to classify objects when simple heuristics are insufficient.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Domain keyword maps
# ──────────────────────────────────────────────

DOMAIN_KEYWORDS = {
    "ceramics": [
        "amphora", "pottery", "sherd", "ceramic", "vessel", "terracotta",
        "fragment", "rim", "handle", "base", "coarse ware", "fine ware",
        "slip", "glaze", "kiln", "dressel", "morel", "lucerna", "lamp",
        "dolium", "pithos", "unguentarium", "plate", "bowl", "cup",
        "jug", "flask", "olla", "teglia", "coccio", "vaso", "anfora"
    ],
    "paintings": [
        "fresco", "painting", "mural", "wall painting", "pigment",
        "plaster", "stucco", "panel", "scene", "mythological",
        "landscape", "portrait", "decoration", "ornament", "motif",
        "first style", "second style", "third style", "fourth style",
        "incrustation", "architectural style", "ornamental", "intricate",
        "mau", "affresco", "pittura", "dipinto", "intonaco"
    ],
    "architecture": [
        "wall", "masonry", "opus", "reticulatum", "incertum", "testaceum",
        "mixtum", "caementicium", "latericium", "signinum", "quadratum",
        "column", "capital", "architrave", "pediment", "vault", "arch",
        "brick", "tile", "tufa", "travertine", "marble", "limestone",
        "foundation", "threshold", "cornice", "pilaster", "entablature",
        "domus", "insula", "atrium", "peristyle", "impluvium",
        "muratura", "colonna", "struttura", "pavimento"
    ]
}


class AgenticDispatcher:
    """
    Routes detected objects to the correct archaeological domain.
    
    Strategy:
    1. Try keyword matching first (fast, deterministic).
    2. If ambiguous, use LLM-based classification (slower, more accurate).
    3. Falls back to "ceramics" if truly uncertain (most common artifact type).
    """

    def __init__(self, domains: list[str], llm_client=None):
        self.domains = domains
        self.llm_client = llm_client  # Optional: for LLM-based routing

    def route(self, label: str, description: str, expert_prompt: str) -> str:
        """
        Determine which domain an object belongs to.
        
        Args:
            label: Object label from VLM.
            description: Object description from VLM.
            expert_prompt: The expert's original prompt.
            
        Returns:
            Domain name (e.g., "ceramics", "paintings", "architecture").
        """
        # Combine all text for matching
        text = f"{label} {description} {expert_prompt}".lower()

        # ── Step 1: Keyword matching ────────────
        scores = {}
        for domain, keywords in DOMAIN_KEYWORDS.items():
            if domain not in self.domains:
                continue
            score = sum(1 for kw in keywords if kw in text)
            scores[domain] = score

        # Clear winner?
        max_score = max(scores.values()) if scores else 0
        winners = [d for d, s in scores.items() if s == max_score and s > 0]

        if len(winners) == 1:
            logger.debug(f"Dispatcher: keyword match → {winners[0]} (score={max_score})")
            return winners[0]

        # ── Step 2: LLM classification (if available) ──
        if self.llm_client and len(winners) != 1:
            domain = self._llm_classify(label, description, expert_prompt)
            if domain:
                return domain

        # ── Step 3: Fallback ────────────────────
        if len(winners) > 1:
            logger.warning(
                f"Dispatcher: ambiguous ({winners}), choosing first match."
            )
            return winners[0]
        
        logger.warning("Dispatcher: no keyword match, defaulting to ceramics.")
        return "ceramics"

    def _llm_classify(
        self, label: str, description: str, expert_prompt: str
    ) -> Optional[str]:
        """
        Use an LLM to classify the object into a domain.
        Used when keyword matching is ambiguous.
        """
        prompt = f"""Classify this archaeological object into exactly one domain.

Domains: {', '.join(self.domains)}

Object label: {label}
Description: {description}
Expert context: {expert_prompt}

Respond with ONLY the domain name, nothing else."""

        try:
            # Replace with your LLM call:
            # response = self.llm_client.generate(prompt)
            # domain = response.strip().lower()
            # if domain in self.domains:
            #     return domain
            pass
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
        
        return None
