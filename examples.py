"""
Example usage of the Archaeological Pipeline.
Demonstrates both modes: auto-detection and user-provided bounding boxes.
"""

from main import ArchaeologicalPipeline, PipelineInput
from vlm_module import BoundingBox
from config import PipelineConfig


# ══════════════════════════════════════════════
#  EXAMPLE 1: Auto-detection (no user bboxes)
# ══════════════════════════════════════════════

def example_auto_detection():
    """
    The expert provides an image and a prompt.
    The VLM detects all objects and generates bounding boxes automatically.
    """
    config = PipelineConfig(
        vlm_model="llava-v1.6-mistral-7b",
        output_dir="./output/auto_detection"
    )

    pipeline = ArchaeologicalPipeline(config)

    result = pipeline.run(PipelineInput(
        image_path="./images/pompeii_room_VII_4_31.jpg",
        expert_prompt=(
            "This is a photograph from Room 31 of House VII.4 in Pompeii, "
            "Regio VII, Insula 4. The image shows a corner of the room with "
            "visible wall paintings and ceramic fragments on the floor. "
            "Identify all archaeological artifacts and architectural features. "
            "The room is dated to the final phase before the eruption of 79 AD."
        ),
        bounding_boxes=None  # VLM will detect objects
    ))

    print(f"\nAuto-detection found {len(result.results)} objects:")
    for r in result.results:
        print(f"  [{r.domain}] {r.label} (bbox by {r.bbox_source})")


# ══════════════════════════════════════════════
#  EXAMPLE 2: User-provided bounding boxes
# ══════════════════════════════════════════════

def example_user_bboxes():
    """
    The expert draws bounding boxes manually (e.g., via a web UI)
    and provides them along with the image.
    The VLM only interprets each region — no detection needed.
    """
    config = PipelineConfig(
        vlm_model="llava-v1.6-mistral-7b",
        output_dir="./output/user_bboxes"
    )

    pipeline = ArchaeologicalPipeline(config)

    result = pipeline.run(PipelineInput(
        image_path="./images/pompeii_room_VII_4_31.jpg",
        expert_prompt=(
            "Pompeii, House VII.4, Room 31. Final phase (62-79 AD). "
            "I have identified three regions of interest in this photograph."
        ),
        bounding_boxes=[
            BoundingBox(x=50, y=100, width=200, height=300, label="amphora_fragment"),
            BoundingBox(x=300, y=50, width=350, height=400, label="wall_fresco"),
            BoundingBox(x=100, y=450, width=500, height=150, label="opus_reticulatum"),
        ]
    ))

    print(f"\nUser-guided analysis found {len(result.results)} objects:")
    for r in result.results:
        print(f"  [{r.domain}] {r.label} (bbox by {r.bbox_source})")


# ══════════════════════════════════════════════
#  EXAMPLE 3: Mixed — partial user annotation
# ══════════════════════════════════════════════

def example_partial_annotation():
    """
    The expert annotates only the objects they are sure about.
    Other regions are still analyzed — the user boxes take priority.
    """
    config = PipelineConfig(output_dir="./output/partial")
    pipeline = ArchaeologicalPipeline(config)

    # Expert marks only the amphora — the rest is auto-detected
    result = pipeline.run(PipelineInput(
        image_path="./images/pompeii_room_VII_4_31.jpg",
        expert_prompt=(
            "I have circled a Dressel 2-4 amphora fragment in this image. "
            "Please classify it and also identify any other visible artifacts."
        ),
        bounding_boxes=[
            BoundingBox(x=120, y=200, width=180, height=250, label="dressel_2_4"),
        ]
    ))

    for r in result.results:
        print(f"  [{r.domain}] {r.label} — bbox_source={r.bbox_source}")


# ══════════════════════════════════════════════
#  EXAMPLE 4: CLI usage
# ══════════════════════════════════════════════

def show_cli_usage():
    """Print CLI usage examples."""
    print("""
CLI Usage Examples:
═══════════════════

# Auto-detection (no bounding boxes):
python main.py \\
    --image ./images/pompeii_room.jpg \\
    --prompt "Pompeii House VII.4 Room 31, final phase 62-79 AD" \\
    --output-dir ./output

# With user-defined bounding boxes:
python main.py \\
    --image ./images/pompeii_room.jpg \\
    --prompt "Pompeii House VII.4 Room 31, final phase 62-79 AD" \\
    --bboxes '[{"x":50,"y":100,"w":200,"h":300,"label":"amphora"},{"x":300,"y":50,"w":350,"h":400,"label":"fresco"}]' \\
    --output-dir ./output

# With config file:
python main.py \\
    --image ./images/pompeii_room.jpg \\
    --prompt "Identify all visible artifacts" \\
    --config ./config.yaml \\
    --output-dir ./output
""")


# ══════════════════════════════════════════════
#  EXAMPLE 5: Batch processing
# ══════════════════════════════════════════════

def example_batch_processing():
    """Process multiple images, some with bboxes, some without."""
    config = PipelineConfig(output_dir="./output/batch")
    pipeline = ArchaeologicalPipeline(config)

    batch = [
        PipelineInput(
            image_path="./images/img_001.jpg",
            expert_prompt="Ceramics storage room, Regio I",
            bounding_boxes=None  # auto-detect
        ),
        PipelineInput(
            image_path="./images/img_002.jpg",
            expert_prompt="Wall with Fourth Style decoration",
            bounding_boxes=[
                BoundingBox(x=0, y=0, width=800, height=600, label="fourth_style_panel")
            ]
        ),
        PipelineInput(
            image_path="./images/img_003.jpg",
            expert_prompt="Intersection showing multiple opus types",
            bounding_boxes=[
                BoundingBox(x=10, y=10, width=300, height=500, label="opus_reticulatum"),
                BoundingBox(x=320, y=10, width=300, height=500, label="opus_incertum"),
            ]
        ),
    ]

    all_results = []
    for inp in batch:
        result = pipeline.run(inp)
        all_results.append(result)
        print(f"  {inp.image_path}: {len(result.results)} objects")

    print(f"\nBatch complete: {sum(len(r.results) for r in all_results)} total objects.")


# ──────────────────────────────────────────────

if __name__ == "__main__":
    show_cli_usage()
    print("Running Example 1: Auto-detection...")
    example_auto_detection()
    print("\nRunning Example 2: User bounding boxes...")
    example_user_bboxes()
