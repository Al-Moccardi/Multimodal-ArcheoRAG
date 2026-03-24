"""
Image Annotator and Export module.
Draws bounding boxes, exports COCO JSON, CSV, and metadata JSON.
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from config import OUTPUT_METADATA, OUTPUT_ANNOTATIONS, OUTPUT_IMAGES

logger = logging.getLogger(__name__)

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


class ImageAnnotator:
    """Draws bounding boxes on images and exports annotations."""

    # ──────────────────────────────────────────
    #  Annotated image
    # ──────────────────────────────────────────

    def annotate_image(
        self,
        image_path: str,
        objects: list[dict],
        all_metadata: list[dict],
    ) -> str:
        """
        Draw bounding boxes on image and save.

        Returns:
            Path to the annotated image.
        """
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16
            )
            font_sm = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12
            )
        except OSError:
            font = ImageFont.load_default()
            font_sm = font

        for i, obj in enumerate(objects):
            bbox = obj.get("bbox", {})
            x = bbox.get("x", 0)
            y = bbox.get("y", 0)
            w = bbox.get("width", 50)
            h = bbox.get("height", 50)

            domain = obj.get("assigned_domain", obj.get("domain", "ceramics"))
            color = DOMAIN_COLORS.get(domain, (180, 180, 180))
            label = obj.get("label", f"obj_{i}")
            bbox_src = obj.get("bbox_source", "vlm")
            conf = obj.get("confidence", 0.0)

            # Rectangle
            draw.rectangle([x, y, x + w, y + h], outline=color, width=3)

            # Label background
            text_main = f"{label} [{domain}]"
            text_sub = f"bbox:{bbox_src} conf:{conf:.2f}"

            tb = draw.textbbox((0, 0), text_main, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]

            ly = max(y - th - 22, 0)
            draw.rectangle([x, ly, x + tw + 10, ly + th + 20], fill=color)
            draw.text((x + 5, ly + 2), text_main, fill=(255, 255, 255), font=font)
            draw.text((x + 5, ly + th + 4), text_sub, fill=(255, 255, 255), font=font_sm)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = str(OUTPUT_IMAGES / f"annotated_{ts}.png")
        img.save(out_path, quality=95)
        logger.info(f"Annotated image saved: {out_path}")
        return out_path

    # ──────────────────────────────────────────
    #  COCO JSON
    # ──────────────────────────────────────────

    def export_coco_json(
        self,
        image_path: str,
        objects: list[dict],
        all_metadata: list[dict],
    ) -> str:
        """Export annotations in COCO format. Returns file path."""
        try:
            img = Image.open(image_path)
            iw, ih = img.size
        except Exception:
            iw, ih = 0, 0

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_id = ts

        coco = {
            "info": {
                "description": "Pompeii Archaeological Annotations",
                "version": "1.0",
                "date_created": datetime.now().isoformat(),
            },
            "images": [
                {
                    "id": image_id,
                    "file_name": Path(image_path).name,
                    "width": iw,
                    "height": ih,
                }
            ],
            "categories": [
                {"id": v, "name": k}
                for k, v in DOMAIN_CATEGORY_IDS.items()
            ],
            "annotations": [],
        }

        for i, obj in enumerate(objects):
            bbox = obj.get("bbox", {})
            domain = obj.get("assigned_domain", obj.get("domain", "ceramics"))
            coco["annotations"].append(
                {
                    "id": i,
                    "image_id": image_id,
                    "category_id": DOMAIN_CATEGORY_IDS.get(domain, 0),
                    "bbox": [
                        bbox.get("x", 0),
                        bbox.get("y", 0),
                        bbox.get("width", 0),
                        bbox.get("height", 0),
                    ],
                    "area": bbox.get("width", 0) * bbox.get("height", 0),
                    "iscrowd": 0,
                    "attributes": {
                        "label": obj.get("label", ""),
                        "bbox_source": obj.get("bbox_source", "vlm"),
                        "confidence": obj.get("confidence", 0),
                        "domain": domain,
                    },
                }
            )

        out_path = str(OUTPUT_ANNOTATIONS / f"coco_{ts}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(coco, f, indent=2, ensure_ascii=False)
        return out_path

    # ──────────────────────────────────────────
    #  CSV
    # ──────────────────────────────────────────

    def export_csv(
        self,
        image_path: str,
        objects: list[dict],
        all_metadata: list[dict],
    ) -> str:
        """Export annotations as CSV. Returns file path."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = str(OUTPUT_ANNOTATIONS / f"annotations_{ts}.csv")

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "object_index", "domain", "label",
                "bbox_x", "bbox_y", "bbox_w", "bbox_h",
                "bbox_source", "confidence",
            ])
            for i, obj in enumerate(objects):
                bbox = obj.get("bbox", {})
                writer.writerow([
                    i,
                    obj.get("assigned_domain", obj.get("domain", "")),
                    obj.get("label", ""),
                    bbox.get("x", 0),
                    bbox.get("y", 0),
                    bbox.get("width", 0),
                    bbox.get("height", 0),
                    obj.get("bbox_source", "vlm"),
                    f"{obj.get('confidence', 0):.3f}",
                ])
        return out_path

    # ──────────────────────────────────────────
    #  Metadata JSON — Catalogue-ready
    # ──────────────────────────────────────────

    def save_metadata(self, all_metadata: list[dict]) -> str:
        """
        Save catalogue-ready metadata:
          1. One JSON per object (catalogue card): metadata/{catalogue_id}.json
          2. Session batch file: metadata/batch_{timestamp}.json
          3. Append to persistent catalogue: metadata/catalogue_master.json
        
        Returns path to the batch file (for Gradio download).
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Individual catalogue cards
        for meta in all_metadata:
            cat_id = meta.get("catalogue_id", f"unknown_{ts}")
            card_path = OUTPUT_METADATA / f"{cat_id}.json"
            with open(card_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

        # 2. Session batch
        batch_path = str(OUTPUT_METADATA / f"batch_{ts}.json")
        with open(batch_path, "w", encoding="utf-8") as f:
            json.dump(all_metadata, f, indent=2, ensure_ascii=False, default=str)

        # 3. Append to persistent master catalogue
        master_path = OUTPUT_METADATA / "catalogue_master.json"
        existing = []
        if master_path.exists():
            try:
                existing = json.loads(master_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing = []
        
        # Add new entries (avoid duplicates by catalogue_id)
        existing_ids = {e.get("catalogue_id") for e in existing}
        for meta in all_metadata:
            if meta.get("catalogue_id") not in existing_ids:
                existing.append(meta)
        
        with open(master_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Saved {len(all_metadata)} catalogue cards. Master has {len(existing)} total entries.")

        return batch_path
