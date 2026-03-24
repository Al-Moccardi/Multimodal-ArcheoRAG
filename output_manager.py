"""
Output Manager.
Generates all pipeline outputs:
  - Structured metadata JSON (per object, per domain DB)
  - COCO-format annotations JSON
  - CSV annotations
  - Annotated images with bounding boxes
  - Cross-reference index
"""

import csv
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from main import PipelineOutput, AnalysisResult

from vlm_module import BoundingBox

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  COCO Annotation format
# ──────────────────────────────────────────────

@dataclass
class COCOAnnotation:
    """Single COCO-format annotation."""
    id: int
    image_id: str
    category_id: int
    category_name: str
    bbox: list          # [x, y, width, height]
    area: int
    iscrowd: int = 0
    attributes: dict = None

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "image_id": self.image_id,
            "category_id": self.category_id,
            "bbox": self.bbox,
            "area": self.area,
            "iscrowd": self.iscrowd
        }
        if self.attributes:
            d["attributes"] = self.attributes
        return d


# ──────────────────────────────────────────────
#  Domain color scheme for annotated images
# ──────────────────────────────────────────────

DOMAIN_COLORS = {
    "ceramics":     (29, 158, 117),   # teal
    "paintings":    (186, 117, 23),    # amber
    "architecture": (55, 138, 221),    # blue
}

DOMAIN_CATEGORY_IDS = {
    "ceramics": 1,
    "paintings": 2,
    "architecture": 3,
}


class OutputManager:
    """
    Manages all pipeline outputs.
    
    Directory structure:
        output/
        ├── {image_id}/
        │   ├── metadata/
        │   │   ├── obj_000_ceramics.json
        │   │   ├── obj_001_paintings.json
        │   │   └── ...
        │   ├── annotations/
        │   │   ├── coco_annotations.json
        │   │   └── annotations.csv
        │   ├── images/
        │   │   └── annotated.png
        │   └── cross_reference.json
    """

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)

    def save_all(self, pipeline_output: "PipelineOutput", image_path: str):
        """Save all outputs for a pipeline run."""
        image_id = pipeline_output.image_id
        base_dir = self.output_dir / image_id
        
        # Create directories
        (base_dir / "metadata").mkdir(parents=True, exist_ok=True)
        (base_dir / "annotations").mkdir(parents=True, exist_ok=True)
        (base_dir / "images").mkdir(parents=True, exist_ok=True)

        # 1. Structured metadata JSON (one per object)
        self._save_metadata(pipeline_output.results, base_dir / "metadata")

        # 2. COCO annotations
        self._save_coco_annotations(
            pipeline_output, image_path, base_dir / "annotations"
        )

        # 3. CSV annotations
        self._save_csv_annotations(
            pipeline_output.results, base_dir / "annotations"
        )

        # 4. Annotated image
        self._save_annotated_image(
            pipeline_output.results, image_path, base_dir / "images"
        )

        # 5. Cross-reference index
        self._save_cross_references(
            pipeline_output.cross_references, base_dir
        )

        logger.info(f"All outputs saved to: {base_dir}")

    # ──────────────────────────────────────────
    #  1. Structured metadata JSON
    # ──────────────────────────────────────────

    def _save_metadata(self, results: list, metadata_dir: Path):
        """Save one JSON file per detected object."""
        for result in results:
            filename = f"{result.object_id}_{result.domain}.json"
            filepath = metadata_dir / filename

            output = {
                "object_id": result.object_id,
                "domain": result.domain,
                "label": result.label,
                "bbox": result.bbox.to_dict(),
                "bbox_source": result.bbox_source,
                "confidence": result.confidence,
                "metadata": result.metadata,
                "generated_at": datetime.now().isoformat()
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(results)} metadata files.")

    # ──────────────────────────────────────────
    #  2. COCO annotations
    # ──────────────────────────────────────────

    def _save_coco_annotations(
        self, pipeline_output: "PipelineOutput",
        image_path: str, annotations_dir: Path
    ):
        """Save annotations in COCO JSON format."""
        try:
            img = Image.open(image_path)
            img_w, img_h = img.size
        except Exception:
            img_w, img_h = 0, 0

        coco = {
            "info": {
                "description": "Archaeological Artifact Annotations",
                "version": "1.0",
                "date_created": datetime.now().isoformat(),
                "pipeline": "archaeological-multimodal-framework"
            },
            "images": [{
                "id": pipeline_output.image_id,
                "file_name": Path(image_path).name,
                "width": img_w,
                "height": img_h
            }],
            "categories": [
                {"id": v, "name": k}
                for k, v in DOMAIN_CATEGORY_IDS.items()
            ],
            "annotations": []
        }

        for i, result in enumerate(pipeline_output.results):
            annotation = COCOAnnotation(
                id=i,
                image_id=pipeline_output.image_id,
                category_id=DOMAIN_CATEGORY_IDS.get(result.domain, 0),
                category_name=result.domain,
                bbox=result.bbox.to_coco(),
                area=result.bbox.area,
                attributes={
                    "label": result.label,
                    "bbox_source": result.bbox_source,
                    "confidence": result.confidence,
                    "domain": result.domain
                }
            )
            coco["annotations"].append(annotation.to_dict())

        filepath = annotations_dir / "coco_annotations.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(coco, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved COCO annotations: {filepath}")

    # ──────────────────────────────────────────
    #  3. CSV annotations
    # ──────────────────────────────────────────

    def _save_csv_annotations(self, results: list, annotations_dir: Path):
        """Save annotations as CSV."""
        filepath = annotations_dir / "annotations.csv"

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "object_id", "domain", "label",
                "bbox_x", "bbox_y", "bbox_w", "bbox_h",
                "bbox_source", "confidence"
            ])
            for result in results:
                writer.writerow([
                    result.object_id,
                    result.domain,
                    result.label,
                    result.bbox.x,
                    result.bbox.y,
                    result.bbox.width,
                    result.bbox.height,
                    result.bbox_source,
                    f"{result.confidence:.3f}"
                ])

        logger.info(f"Saved CSV annotations: {filepath}")

    # ──────────────────────────────────────────
    #  4. Annotated image
    # ──────────────────────────────────────────

    def _save_annotated_image(
        self, results: list, image_path: str, images_dir: Path
    ):
        """Draw bounding boxes on the image and save."""
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Cannot open image for annotation: {e}")
            return

        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except OSError:
            font = ImageFont.load_default()
            font_small = font

        for result in results:
            bbox = result.bbox
            color = DOMAIN_COLORS.get(result.domain, (200, 200, 200))

            # Draw bbox rectangle
            draw.rectangle(
                [bbox.x, bbox.y, bbox.x2, bbox.y2],
                outline=color, width=3
            )

            # Draw label background
            label_text = f"{result.label} [{result.domain}]"
            source_text = f"bbox: {result.bbox_source} | conf: {result.confidence:.2f}"

            text_bbox = draw.textbbox((0, 0), label_text, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]

            label_y = max(bbox.y - text_h - 22, 0)
            draw.rectangle(
                [bbox.x, label_y, bbox.x + text_w + 10, label_y + text_h + 20],
                fill=color
            )
            draw.text((bbox.x + 5, label_y + 2), label_text, fill=(255, 255, 255), font=font)
            draw.text((bbox.x + 5, label_y + text_h + 4), source_text, fill=(255, 255, 255), font=font_small)

        filepath = images_dir / "annotated.png"
        img.save(filepath, quality=95)
        logger.info(f"Saved annotated image: {filepath}")

    # ──────────────────────────────────────────
    #  5. Cross-reference index
    # ──────────────────────────────────────────

    def _save_cross_references(self, cross_refs: dict, base_dir: Path):
        """Save cross-reference index."""
        filepath = base_dir / "cross_reference.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(cross_refs, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved cross-reference index: {filepath}")
