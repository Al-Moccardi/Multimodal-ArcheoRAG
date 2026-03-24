"""
Archaeological Artifact Analysis Pipeline
==========================================
A Multimodal Framework for Intelligent Analysis and Semantic Enrichment
of Archaeological Artifacts using VLM + Agentic RAG.

Input:  Image + Expert Prompt + Optional Bounding Boxes
Output: Per-object structured metadata (JSON), COCO annotations, annotated images
"""

import json
import uuid
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from vlm_module import VLMModule, VLMResult, BoundingBox
from agentic_dispatcher import AgenticDispatcher
from rag_engine import DualRAGEngine
from refinement import LLMRefinement
from output_manager import OutputManager, COCOAnnotation
from config import PipelineConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Data classes
# ──────────────────────────────────────────────

@dataclass
class PipelineInput:
    """Input to the pipeline."""
    image_path: str
    expert_prompt: str
    bounding_boxes: Optional[list[BoundingBox]] = None  # Optional user-defined boxes

    def has_user_boxes(self) -> bool:
        return self.bounding_boxes is not None and len(self.bounding_boxes) > 0


@dataclass
class DetectedObject:
    """A single detected object in the image."""
    object_id: str
    label: str
    description: str
    bbox: BoundingBox
    confidence: float
    bbox_source: str  # "user" or "vlm"


@dataclass
class AnalysisResult:
    """Final result for a single object."""
    object_id: str
    domain: str
    label: str
    bbox: BoundingBox
    bbox_source: str
    metadata: dict
    historical_context: str
    cataloguing_info: str
    confidence: float


@dataclass
class PipelineOutput:
    """Complete pipeline output for one image."""
    image_id: str
    image_path: str
    results: list[AnalysisResult] = field(default_factory=list)
    cross_references: dict = field(default_factory=dict)


# ──────────────────────────────────────────────
#  Main Pipeline
# ──────────────────────────────────────────────

class ArchaeologicalPipeline:
    """
    Main pipeline orchestrator.
    
    Flow:
        1. INPUT: Image + Prompt + Optional Bounding Boxes
        2. VLM: Interprets the scene, detects objects (or uses user boxes)
        3. DISPATCHER: Routes each object to the correct domain
        4. DUAL RAG: Queries historical + cataloguing DBs per domain
        5. REFINEMENT: Merges context + schema into structured JSON
        6. OUTPUT: Metadata JSON, COCO annotations, annotated images
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        logger.info("Initializing Archaeological Pipeline...")
        
        # Stage 1: Visual Language Model
        self.vlm = VLMModule(
            model_name=self.config.vlm_model,
            device=self.config.device
        )
        
        # Stage 2: Agentic Dispatcher
        self.dispatcher = AgenticDispatcher(
            domains=self.config.domains
        )
        
        # Stage 3: Dual RAG engines (one per domain)
        self.rag_engines: dict[str, DualRAGEngine] = {}
        for domain in self.config.domains:
            self.rag_engines[domain] = DualRAGEngine(
                domain=domain,
                historical_db_path=self.config.get_historical_db_path(domain),
                cataloguing_db_path=self.config.get_cataloguing_db_path(domain),
                embedding_model=self.config.embedding_model
            )
        
        # Stage 4: LLM Refinement
        self.refinement = LLMRefinement(
            model_name=self.config.llm_model
        )
        
        # Stage 5: Output Manager
        self.output_manager = OutputManager(
            output_dir=self.config.output_dir
        )
        
        logger.info("Pipeline initialized successfully.")

    # ──────────────────────────────────────────
    #  Main entry point
    # ──────────────────────────────────────────

    def run(self, pipeline_input: PipelineInput) -> PipelineOutput:
        """
        Execute the full pipeline on a single image.
        
        Args:
            pipeline_input: Image path, expert prompt, and optional bounding boxes.
            
        Returns:
            PipelineOutput with per-object results, annotations, and cross-references.
        """
        image_id = str(uuid.uuid4())[:8]
        logger.info(f"[{image_id}] Starting pipeline for: {pipeline_input.image_path}")

        # ── Step 1: VLM Interpretation ──────────────────────
        logger.info(f"[{image_id}] Step 1: VLM interpretation...")
        
        vlm_result: VLMResult = self.vlm.interpret(
            image_path=pipeline_input.image_path,
            prompt=pipeline_input.expert_prompt,
            user_bboxes=pipeline_input.bounding_boxes
        )

        # ── Step 2: Resolve bounding boxes ──────────────────
        detected_objects = self._resolve_bounding_boxes(
            vlm_result=vlm_result,
            user_bboxes=pipeline_input.bounding_boxes
        )
        
        logger.info(
            f"[{image_id}] Detected {len(detected_objects)} objects "
            f"(bbox source: {'user' if pipeline_input.has_user_boxes() else 'vlm'})"
        )

        # ── Step 3-5: Process each object ───────────────────
        results: list[AnalysisResult] = []
        
        for obj in detected_objects:
            result = self._process_single_object(
                image_id=image_id,
                obj=obj,
                expert_prompt=pipeline_input.expert_prompt
            )
            if result:
                results.append(result)

        # ── Step 6: Generate outputs ────────────────────────
        pipeline_output = PipelineOutput(
            image_id=image_id,
            image_path=pipeline_input.image_path,
            results=results
        )

        # Build cross-references (objects from same image)
        pipeline_output.cross_references = self._build_cross_references(results)

        # Write all outputs
        self.output_manager.save_all(
            pipeline_output=pipeline_output,
            image_path=pipeline_input.image_path
        )

        logger.info(
            f"[{image_id}] Pipeline complete. "
            f"{len(results)} objects catalogued across "
            f"{len(set(r.domain for r in results))} domains."
        )
        
        return pipeline_output

    # ──────────────────────────────────────────
    #  Internal methods
    # ──────────────────────────────────────────

    def _resolve_bounding_boxes(
        self,
        vlm_result: VLMResult,
        user_bboxes: Optional[list[BoundingBox]]
    ) -> list[DetectedObject]:
        """
        Resolve bounding boxes: use user-provided if available,
        otherwise use VLM-detected boxes.
        """
        detected = []

        if user_bboxes and len(user_bboxes) > 0:
            # USER-PROVIDED BBOXES
            # The VLM has already interpreted each user box region.
            for i, bbox in enumerate(user_bboxes):
                # Find the VLM interpretation for this bbox region
                interpretation = vlm_result.get_interpretation_for_bbox(bbox, index=i)
                
                detected.append(DetectedObject(
                    object_id=f"obj_{i:03d}",
                    label=interpretation.get("label", f"object_{i}"),
                    description=interpretation.get("description", ""),
                    bbox=bbox,
                    confidence=interpretation.get("confidence", 1.0),
                    bbox_source="user"
                ))
        else:
            # VLM-DETECTED BBOXES
            for i, detection in enumerate(vlm_result.detections):
                detected.append(DetectedObject(
                    object_id=f"obj_{i:03d}",
                    label=detection.label,
                    description=detection.description,
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    bbox_source="vlm"
                ))

        return detected

    def _process_single_object(
        self,
        image_id: str,
        obj: DetectedObject,
        expert_prompt: str
    ) -> Optional[AnalysisResult]:
        """
        Process a single detected object through dispatcher → RAG → refinement.
        """
        logger.info(f"[{image_id}] Processing object: {obj.label} ({obj.object_id})")

        # ── Dispatch: determine domain ──────────────────────
        domain = self.dispatcher.route(
            label=obj.label,
            description=obj.description,
            expert_prompt=expert_prompt
        )
        
        if domain not in self.rag_engines:
            logger.warning(f"[{image_id}] Unknown domain '{domain}' for {obj.label}. Skipping.")
            return None
        
        logger.info(f"[{image_id}]   → Routed to domain: {domain}")

        # ── Dual RAG: historical + cataloguing ──────────────
        rag_engine = self.rag_engines[domain]
        
        query = self._build_rag_query(obj, expert_prompt)
        
        historical_context = rag_engine.query_historical(query)
        cataloguing_info = rag_engine.query_cataloguing(query)
        
        logger.info(
            f"[{image_id}]   → RAG retrieved: "
            f"{len(historical_context)} historical chunks, "
            f"{len(cataloguing_info)} cataloguing chunks"
        )

        # ── Merge + Refinement ──────────────────────────────
        metadata = self.refinement.refine(
            domain=domain,
            object_label=obj.label,
            object_description=obj.description,
            historical_context=historical_context,
            cataloguing_info=cataloguing_info,
            expert_prompt=expert_prompt,
            bbox_source=obj.bbox_source
        )

        return AnalysisResult(
            object_id=obj.object_id,
            domain=domain,
            label=obj.label,
            bbox=obj.bbox,
            bbox_source=obj.bbox_source,
            metadata=metadata,
            historical_context=historical_context,
            cataloguing_info=cataloguing_info,
            confidence=obj.confidence
        )

    def _build_rag_query(self, obj: DetectedObject, expert_prompt: str) -> str:
        """Build the RAG query from object info + expert prompt."""
        return (
            f"Object: {obj.label}. "
            f"Description: {obj.description}. "
            f"Expert context: {expert_prompt}"
        )

    def _build_cross_references(self, results: list[AnalysisResult]) -> dict:
        """
        Build cross-reference index linking objects from the same image.
        Useful for co-occurrence analysis (e.g., amphora + fresco in same room).
        """
        cross_refs = {
            "total_objects": len(results),
            "domains_found": list(set(r.domain for r in results)),
            "objects": [
                {
                    "object_id": r.object_id,
                    "domain": r.domain,
                    "label": r.label,
                    "bbox": r.bbox.to_dict(),
                    "bbox_source": r.bbox_source
                }
                for r in results
            ],
            "co_occurrences": self._compute_co_occurrences(results)
        }
        return cross_refs

    def _compute_co_occurrences(self, results: list[AnalysisResult]) -> list[dict]:
        """Compute pairwise spatial co-occurrences between objects."""
        co_occurrences = []
        for i, r1 in enumerate(results):
            for r2 in results[i + 1:]:
                co_occurrences.append({
                    "pair": [r1.object_id, r2.object_id],
                    "domains": [r1.domain, r2.domain],
                    "labels": [r1.label, r2.label],
                    "spatial_overlap": r1.bbox.iou(r2.bbox)
                })
        return co_occurrences


# ──────────────────────────────────────────────
#  CLI entry point
# ──────────────────────────────────────────────

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Archaeological Artifact Analysis Pipeline"
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--prompt", required=True, help="Expert domain prompt")
    parser.add_argument(
        "--bboxes", default=None,
        help='Optional bounding boxes as JSON: [{"x":10,"y":20,"w":100,"h":150,"label":"amphora"}, ...]'
    )
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--output-dir", default="./output", help="Output directory")
    
    args = parser.parse_args()
    
    # Parse optional bounding boxes
    user_bboxes = None
    if args.bboxes:
        raw_boxes = json.loads(args.bboxes)
        user_bboxes = [
            BoundingBox(
                x=b["x"], y=b["y"],
                width=b["w"], height=b["h"],
                label=b.get("label")
            )
            for b in raw_boxes
        ]

    # Load config
    config = PipelineConfig(output_dir=args.output_dir)
    if args.config:
        config = PipelineConfig.from_yaml(args.config)
    
    # Build input
    pipeline_input = PipelineInput(
        image_path=args.image,
        expert_prompt=args.prompt,
        bounding_boxes=user_bboxes
    )
    
    # Run
    pipeline = ArchaeologicalPipeline(config=config)
    output = pipeline.run(pipeline_input)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Pipeline complete: {output.image_id}")
    print(f"Objects found: {len(output.results)}")
    for r in output.results:
        print(f"  [{r.domain}] {r.label} — bbox_source={r.bbox_source}")
    print(f"Output directory: {config.output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
