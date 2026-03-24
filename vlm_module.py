"""
Visual Language Model module.
Handles image interpretation and object detection.
Supports optional user-provided bounding boxes.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional
from PIL import Image

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Bounding Box
# ──────────────────────────────────────────────

@dataclass
class BoundingBox:
    """
    Bounding box in pixel coordinates.
    Format: (x, y) is top-left corner, width and height define the box.
    """
    x: int
    y: int
    width: int
    height: int
    label: Optional[str] = None

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_dict(self) -> dict:
        return {
            "x": self.x, "y": self.y,
            "width": self.width, "height": self.height,
            "label": self.label
        }

    def to_coco(self) -> list:
        """COCO format: [x, y, width, height]."""
        return [self.x, self.y, self.width, self.height]

    def to_xyxy(self) -> list:
        """[x1, y1, x2, y2] format."""
        return [self.x, self.y, self.x2, self.y2]

    def iou(self, other: "BoundingBox") -> float:
        """Compute Intersection over Union with another box."""
        ix1 = max(self.x, other.x)
        iy1 = max(self.y, other.y)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        intersection = (ix2 - ix1) * (iy2 - iy1)
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0

    def crop_from(self, image: Image.Image) -> Image.Image:
        """Crop this bounding box region from a PIL image."""
        return image.crop((self.x, self.y, self.x2, self.y2))

    @classmethod
    def from_coco(cls, coco_bbox: list, label: str = None) -> "BoundingBox":
        """Create from COCO format [x, y, w, h]."""
        return cls(x=int(coco_bbox[0]), y=int(coco_bbox[1]),
                   width=int(coco_bbox[2]), height=int(coco_bbox[3]),
                   label=label)

    @classmethod
    def from_xyxy(cls, coords: list, label: str = None) -> "BoundingBox":
        """Create from [x1, y1, x2, y2] format."""
        return cls(x=int(coords[0]), y=int(coords[1]),
                   width=int(coords[2] - coords[0]),
                   height=int(coords[3] - coords[1]),
                   label=label)


# ──────────────────────────────────────────────
#  VLM Detection Result
# ──────────────────────────────────────────────

@dataclass
class Detection:
    """A single object detection from the VLM."""
    label: str
    description: str
    bbox: BoundingBox
    confidence: float


@dataclass
class VLMResult:
    """Complete VLM output for one image."""
    scene_description: str
    detections: list[Detection] = field(default_factory=list)
    bbox_interpretations: dict = field(default_factory=dict)

    def get_interpretation_for_bbox(self, bbox: BoundingBox, index: int = 0) -> dict:
        """Get VLM interpretation for a specific user-provided bbox."""
        key = f"bbox_{index}"
        if key in self.bbox_interpretations:
            return self.bbox_interpretations[key]
        return {
            "label": bbox.label or f"object_{index}",
            "description": "",
            "confidence": 1.0
        }


# ──────────────────────────────────────────────
#  VLM Module
# ──────────────────────────────────────────────

class VLMModule:
    """
    Visual Language Model wrapper.
    
    Two operating modes:
    
    A) NO USER BBOXES → Full scene interpretation + object detection
       The VLM analyzes the entire image, identifies objects,
       generates bounding boxes and descriptions.
    
    B) USER BBOXES PROVIDED → Targeted region interpretation
       The VLM receives the image + the user-defined bounding boxes.
       For each bbox, it interprets the cropped region and produces
       a description. No detection needed — locations are known.
    """

    def __init__(self, model_name: str = "llava-v1.6-mistral-7b", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None

    def _load_model(self):
        """Lazy-load the VLM model."""
        if self.model is not None:
            return

        logger.info(f"Loading VLM: {self.model_name}...")

        # ──────────────────────────────────────
        # OPTION A: LLaVA / local model
        # ──────────────────────────────────────
        # from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        # self.processor = LlavaNextProcessor.from_pretrained(self.model_name)
        # self.model = LlavaNextForConditionalGeneration.from_pretrained(
        #     self.model_name, torch_dtype=torch.float16, device_map="auto"
        # )

        # ──────────────────────────────────────
        # OPTION B: Florence-2 (good at grounding)
        # ──────────────────────────────────────
        # from transformers import AutoProcessor, AutoModelForCausalLM
        # self.processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     "microsoft/Florence-2-large", torch_dtype=torch.float16, trust_remote_code=True
        # ).to(self.device)

        # ──────────────────────────────────────
        # OPTION C: OpenAI GPT-4o Vision API
        # ──────────────────────────────────────
        # from openai import OpenAI
        # self.model = OpenAI()

        logger.info("VLM loaded (placeholder mode — swap in your model).")

    def interpret(
        self,
        image_path: str,
        prompt: str,
        user_bboxes: Optional[list[BoundingBox]] = None
    ) -> VLMResult:
        """
        Interpret an archaeological image.
        
        Args:
            image_path: Path to the input image.
            prompt: Expert domain prompt.
            user_bboxes: Optional list of user-defined bounding boxes.
            
        Returns:
            VLMResult with scene description, detections, and bbox interpretations.
        """
        self._load_model()
        image = Image.open(image_path).convert("RGB")

        if user_bboxes and len(user_bboxes) > 0:
            return self._interpret_with_user_bboxes(image, prompt, user_bboxes)
        else:
            return self._interpret_full_scene(image, prompt)

    def _interpret_full_scene(self, image: Image.Image, prompt: str) -> VLMResult:
        """
        MODE A: Full scene — VLM detects objects and interprets them.
        
        The VLM receives the full image and a prompt like:
        "Identify all archaeological artifacts in this image from a Pompeii excavation.
         For each object, provide: label, description, bounding box coordinates."
        """
        detection_prompt = self._build_detection_prompt(prompt)

        # ─── Call VLM ───────────────────────────
        # Replace this with actual model inference:
        #
        # For Florence-2:
        #   inputs = self.processor(text="<OD>", images=image, return_tensors="pt").to(self.device)
        #   generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        #   results = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        #   parsed = self.processor.post_process_generation(results, task="<OD>", image_size=image.size)
        #
        # For GPT-4o:
        #   response = self.model.chat.completions.create(
        #       model="gpt-4o",
        #       messages=[{"role": "user", "content": [
        #           {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        #           {"type": "text", "text": detection_prompt}
        #       ]}],
        #       response_format={"type": "json_object"}
        #   )

        # ─── Placeholder response ───────────────
        logger.info("VLM inference (full scene detection)...")
        vlm_response = self._placeholder_detection(image, prompt)

        return vlm_response

    def _interpret_with_user_bboxes(
        self,
        image: Image.Image,
        prompt: str,
        user_bboxes: list[BoundingBox]
    ) -> VLMResult:
        """
        MODE B: User bboxes — VLM interprets each cropped region.
        
        For each user-defined bounding box:
        1. Crop the image region
        2. Send crop + prompt to VLM
        3. Get interpretation (label, description, domain hints)
        """
        bbox_interpretations = {}
        all_descriptions = []

        for i, bbox in enumerate(user_bboxes):
            # Crop the region
            crop = bbox.crop_from(image)

            interpretation_prompt = self._build_interpretation_prompt(prompt, bbox)

            # ─── Call VLM on crop ───────────────
            # Replace with actual inference on the cropped region:
            #
            # inputs = self.processor(images=crop, text=interpretation_prompt, return_tensors="pt")
            # output = self.model.generate(**inputs, max_new_tokens=512)
            # text = self.processor.decode(output[0], skip_special_tokens=True)

            # ─── Placeholder ────────────────────
            logger.info(f"VLM inference on user bbox {i}: {bbox.label or 'unlabeled'}")
            interpretation = self._placeholder_interpretation(bbox, i)

            bbox_interpretations[f"bbox_{i}"] = interpretation
            all_descriptions.append(interpretation.get("description", ""))

        scene_description = (
            f"User-guided analysis of {len(user_bboxes)} regions. "
            + " | ".join(all_descriptions)
        )

        return VLMResult(
            scene_description=scene_description,
            detections=[],  # No VLM detections — user provided the boxes
            bbox_interpretations=bbox_interpretations
        )

    # ──────────────────────────────────────────
    #  Prompt builders
    # ──────────────────────────────────────────

    def _build_detection_prompt(self, expert_prompt: str) -> str:
        return f"""You are an expert archaeological artifact analyzer.

Analyze this archaeological image and identify all distinct artifacts or features.

Expert context: {expert_prompt}

For each detected object, provide a JSON array with:
- "label": short name (e.g., "amphora_fragment", "fresco_panel", "opus_reticulatum")
- "description": detailed archaeological description
- "bbox": [x, y, width, height] in pixel coordinates
- "confidence": detection confidence 0-1
- "domain_hint": one of "ceramics", "paintings", "architecture"

Respond ONLY with valid JSON."""

    def _build_interpretation_prompt(self, expert_prompt: str, bbox: BoundingBox) -> str:
        region_label = f" (labeled: {bbox.label})" if bbox.label else ""
        return f"""You are an expert archaeological artifact analyzer.

Analyze this cropped region from an archaeological excavation{region_label}.

Expert context: {expert_prompt}

Provide a JSON object with:
- "label": short archaeological name for this object
- "description": detailed description (material, technique, condition, style)
- "confidence": how confident you are in the identification 0-1
- "domain_hint": one of "ceramics", "paintings", "architecture"

Respond ONLY with valid JSON."""

    # ──────────────────────────────────────────
    #  Placeholders (replace with real inference)
    # ──────────────────────────────────────────

    def _placeholder_detection(self, image: Image.Image, prompt: str) -> VLMResult:
        """Placeholder: simulates VLM detection output."""
        w, h = image.size
        return VLMResult(
            scene_description=f"Archaeological scene ({w}x{h}px). Prompt: {prompt[:80]}...",
            detections=[
                Detection(
                    label="amphora_fragment",
                    description="Fragment of a Roman amphora, possibly Dressel 2-4 type",
                    bbox=BoundingBox(x=50, y=100, width=200, height=300, label="amphora_fragment"),
                    confidence=0.87
                ),
                Detection(
                    label="fresco_panel",
                    description="Wall painting fragment with architectural motifs, possible Second Style",
                    bbox=BoundingBox(x=300, y=50, width=350, height=400, label="fresco_panel"),
                    confidence=0.74
                ),
            ]
        )

    def _placeholder_interpretation(self, bbox: BoundingBox, index: int) -> dict:
        """Placeholder: simulates VLM interpretation of a user bbox."""
        return {
            "label": bbox.label or f"artifact_{index}",
            "description": f"Archaeological artifact in user-defined region {index}",
            "confidence": 0.95,
            "domain_hint": "ceramics"
        }
