"""
Pompeii Multimodal Archaeological Framework — Gradio Interface.

- Click-and-draw bounding boxes on the image
- Macro domain labels only (Ceramics / Paintings / Architecture)
- Fancy HTML pipeline log for human-in-the-loop evaluation

Usage:
    python app.py
"""
import os
import json
import uuid
import tempfile
import gradio as gr
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from config import OUTPUT_METADATA, OUTPUT_ANNOTATIONS, OUTPUT_IMAGES
from pipeline.vlm import VLMAnalyzer
from pipeline.dispatcher import AgenticDispatcher
from pipeline.rag_engine import DualRAGEngine
from pipeline.refinement import MetadataRefiner
from pipeline.annotator import ImageAnnotator


# ─── Initialize Pipeline ────────────────────────────────────────
print("[boot] Initializing pipeline...")

vlm = VLMAnalyzer()
dispatcher = AgenticDispatcher()
rag = DualRAGEngine()
refiner = MetadataRefiner()
annotator = ImageAnnotator()

print("[boot] Ready.\n")


# ─── Macro domain labels ────────────────────────────────────────
MACRO_LABELS = {
    "ceramics":     {"emoji": "🏺", "color": (29, 158, 117),  "name": "Ceramics"},
    "paintings":    {"emoji": "🎨", "color": (186, 117, 23),  "name": "Paintings"},
    "architecture": {"emoji": "🏛️", "color": (55, 138, 221),  "name": "Architecture"},
}

DOMAIN_COLORS = {d: v["color"] for d, v in MACRO_LABELS.items()}
DOMAIN_COLORS["auto"] = (255, 80, 80)
DOMAIN_COLORS["unknown"] = (160, 160, 160)


def label_to_domain(label: str) -> str:
    if not label:
        return "auto"
    l = label.lower().strip()
    if l in MACRO_LABELS:
        return l
    return "unknown"


# ─── Font ────────────────────────────────────────────────────────
def _load_font(size):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()

FONT_BIG = _load_font(15)
FONT_SMALL = _load_font(12)


# ─── Image rendering ────────────────────────────────────────────

def render_image(image_path, boxes, first_click_xy=None):
    if not image_path:
        return None

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for i, box in enumerate(boxes):
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        label = box.get("label", "")
        domain = label_to_domain(label)
        color = DOMAIN_COLORS.get(domain, (160, 160, 160))

        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)

        info = MACRO_LABELS.get(label.lower(), {})
        emoji = info.get("emoji", "🔴")
        display_name = info.get("name", label or "auto")
        tag = f" {i+1}. {emoji} {display_name} "

        tb = draw.textbbox((0, 0), tag, font=FONT_BIG)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ly = max(y - th - 4, 0)
        draw.rectangle([x, ly, x + tw + 2, ly + th + 4], fill=color)
        draw.text((x + 1, ly + 2), tag, fill=(255, 255, 255), font=FONT_BIG)

    if first_click_xy:
        px, py = first_click_xy
        r = 8
        draw.ellipse([px - r, py - r, px + r, py + r],
                      fill=(255, 60, 60), outline=(255, 255, 255), width=2)
        draw.text((px + 12, py - 8), "click 2nd corner...",
                  fill=(255, 60, 60), font=FONT_SMALL)

    return img


def format_box_list(boxes):
    if not boxes:
        return "**No boxes drawn** — the VLM will auto-detect all objects."
    lines = []
    for i, b in enumerate(boxes):
        label = b.get("label", "")
        info = MACRO_LABELS.get(label.lower(), {})
        emoji = info.get("emoji", "🔴")
        name = info.get("name", label or "auto")
        lines.append(f"{emoji} **{i+1}.** {name} — ({b['x']}, {b['y']}) {b['width']}×{b['height']}px")
    return f"**{len(boxes)} box(es):**\n\n" + "\n\n".join(lines)


# ─── Click state + label handlers ────────────────────────────────

def _dd_reset():
    return ""

# All handlers return 7 outputs:
# [drawing_image, boxes_state, pending_state, bbox_status, label_row, custom_label_input, custom_label_input_value]
# But we simplified: label_row visibility + custom_label reset

def _make_outputs(preview, boxes, pending, status, show_label_row, custom_val=""):
    return (preview, boxes, pending, status,
            gr.update(visible=show_label_row), custom_val)


def on_image_upload(image_path):
    if not image_path:
        return _make_outputs(None, [], None, "Upload an image to begin.", False)
    img = Image.open(image_path)
    w, h = img.size
    return _make_outputs(
        img, [], None,
        f"**Image loaded: {w}×{h}px** — Click on the image to set the 1st corner of a box.",
        False,
    )


def on_image_click(image_path, boxes, pending, evt: gr.SelectData):
    if not image_path:
        return _make_outputs(None, boxes, None, "Upload an image first.", False)

    cx, cy = evt.index[0], evt.index[1]

    if pending and pending.get("phase") == "label":
        preview = render_image(image_path, boxes, first_click_xy=(cx, cy))
        return _make_outputs(
            preview, boxes,
            {"phase": "corner1", "x": cx, "y": cy},
            f"**Previous box discarded.** 1st corner at ({cx}, {cy}) — click opposite corner.",
            False,
        )

    if pending is None or pending.get("phase") != "corner1":
        preview = render_image(image_path, boxes, first_click_xy=(cx, cy))
        return _make_outputs(
            preview, boxes,
            {"phase": "corner1", "x": cx, "y": cy},
            f"**1st corner: ({cx}, {cy})** — Now click the opposite corner.",
            False,
        )

    x1, y1 = pending["x"], pending["y"]
    x_min, x_max = min(x1, cx), max(x1, cx)
    y_min, y_max = min(y1, cy), max(y1, cy)
    w, h = x_max - x_min, y_max - y_min

    if w < 5 or h < 5:
        preview = render_image(image_path, boxes)
        return _make_outputs(preview, boxes, None, "⚠ Box too small. Try again.", False)

    temp_box = {"x": x_min, "y": y_min, "width": w, "height": h, "label": ""}
    preview = render_image(image_path, boxes + [temp_box])
    return _make_outputs(
        preview, boxes,
        {"phase": "label", "x": x_min, "y": y_min, "w": w, "h": h},
        f"**Box: ({x_min}, {y_min}) {w}×{h}px** — Assign a category below or **Skip** for auto.",
        True,
    )


def assign_macro_label(image_path, boxes, pending, macro_domain):
    """User clicks one of the 3 macro domain buttons."""
    if not pending or pending.get("phase") != "label":
        return _make_outputs(render_image(image_path, boxes), boxes, None, format_box_list(boxes), False)

    new_box = {"x": pending["x"], "y": pending["y"],
               "width": pending["w"], "height": pending["h"], "label": macro_domain}
    boxes = boxes + [new_box]
    preview = render_image(image_path, boxes)
    status = format_box_list(boxes) + "\n\n*Click on the image to draw another box.*"
    return _make_outputs(preview, boxes, None, status, False)


def confirm_custom_label(image_path, boxes, pending, custom_label):
    """User typed a custom label."""
    if not pending or pending.get("phase") != "label":
        return _make_outputs(render_image(image_path, boxes), boxes, None, format_box_list(boxes), False)

    label = custom_label.strip() if custom_label else ""
    new_box = {"x": pending["x"], "y": pending["y"],
               "width": pending["w"], "height": pending["h"], "label": label}
    boxes = boxes + [new_box]
    preview = render_image(image_path, boxes)
    status = format_box_list(boxes) + "\n\n*Click on the image to draw another box.*"
    return _make_outputs(preview, boxes, None, status, False)


def skip_label(image_path, boxes, pending):
    if not pending or pending.get("phase") != "label":
        return _make_outputs(render_image(image_path, boxes), boxes, None, format_box_list(boxes), False)

    new_box = {"x": pending["x"], "y": pending["y"],
               "width": pending["w"], "height": pending["h"], "label": ""}
    boxes = boxes + [new_box]
    preview = render_image(image_path, boxes)
    status = format_box_list(boxes) + "\n\n*Click on the image to draw another box.*"
    return _make_outputs(preview, boxes, None, status, False)


def undo_last_box(image_path, boxes):
    if not boxes:
        img = render_image(image_path, []) if image_path else None
        return _make_outputs(img, [], None, "Nothing to undo.", False)
    boxes = boxes[:-1]
    preview = render_image(image_path, boxes)
    return _make_outputs(preview, boxes, None, format_box_list(boxes), False)


def clear_all_boxes(image_path):
    preview = render_image(image_path, []) if image_path else None
    return _make_outputs(preview, [], None, "All boxes cleared. Click on the image to start.", False)


# ─── Fancy HTML log builder ──────────────────────────────────────

class PipelineLogger:
    """Builds a styled HTML log for human-in-the-loop evaluation."""

    STEP_COLORS = {
        1: "#7F77DD",  # purple  - VLM
        2: "#D85A30",  # coral   - Dispatch
        3: "#1D9E75",  # teal    - RAG
        4: "#378ADD",  # blue    - Refinement
        5: "#639922",  # green   - Export
    }

    def __init__(self):
        self.entries = []
        self.current_step = 0

    def step(self, num: int, title: str):
        self.current_step = num
        color = self.STEP_COLORS.get(num, "#888")
        self.entries.append(
            f'<div style="margin:12px 0 6px;padding:8px 12px;border-radius:8px;'
            f'background:{color}18;border-left:4px solid {color};">'
            f'<span style="font-weight:600;color:{color};font-size:14px;">STEP {num}/5 — {title}</span></div>'
        )

    def info(self, text: str, indent: int = 0):
        pad = 16 + indent * 20
        self.entries.append(
            f'<div style="padding:2px 0 2px {pad}px;font-size:13px;color:var(--body-text-color,#444);'
            f'font-family:monospace;">{text}</div>'
        )

    def object_card(self, idx: int, total: int, label: str, domain: str,
                     bbox_src: str, hist_n: int, cat_n: int, meta_n: int):
        """Render a single object as a mini card."""
        color = {"ceramics": "#1D9E75", "paintings": "#BA7517", "architecture": "#378ADD"}.get(domain, "#888")
        emoji = {"ceramics": "🏺", "paintings": "🎨", "architecture": "🏛️"}.get(domain, "📦")
        self.entries.append(f'''
<div style="margin:8px 0;padding:10px 14px;border-radius:8px;border:1px solid {color}40;background:{color}08;">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <span style="font-weight:600;color:{color};font-size:14px;">{emoji} [{idx}/{total}] {label}</span>
    <span style="font-size:11px;padding:2px 8px;border-radius:10px;background:{color}20;color:{color};font-weight:500;">{domain}</span>
  </div>
  <div style="display:flex;gap:16px;margin-top:6px;font-size:12px;color:#888;">
    <span>bbox: <b>{bbox_src}</b></span>
    <span>RAG: <b>{hist_n}</b> hist + <b>{cat_n}</b> cat</span>
    <span>metadata: <b>{meta_n}</b> fields</span>
  </div>
</div>''')

    def success(self, text: str):
        self.entries.append(
            f'<div style="margin:12px 0;padding:10px 14px;border-radius:8px;'
            f'background:#63992218;border:1px solid #63992240;color:#3B6D11;font-weight:600;font-size:14px;">'
            f'✓ {text}</div>'
        )

    def error(self, text: str):
        self.entries.append(
            f'<div style="margin:8px 0;padding:10px 14px;border-radius:8px;'
            f'background:#E24B4A18;border:1px solid #E24B4A40;color:#A32D2D;font-size:13px;">'
            f'✗ {text}</div>'
        )

    def divider(self):
        self.entries.append('<hr style="border:none;border-top:1px solid #eee;margin:8px 0;">')

    def summary_table(self, objects: list, user_bboxes: bool):
        """Final summary as a styled table."""
        domains = {}
        for o in objects:
            d = o.get("assigned_domain", o.get("domain", "?"))
            domains[d] = domains.get(d, 0) + 1

        mode = "User-drawn boxes" if user_bboxes else "VLM auto-detection"
        rows = "".join(
            f'<tr><td style="padding:4px 12px;">{d}</td><td style="padding:4px 12px;font-weight:600;">{n}</td></tr>'
            for d, n in sorted(domains.items())
        )
        self.entries.append(f'''
<div style="margin:12px 0;padding:14px;border-radius:8px;background:var(--block-background-fill,#fafaf8);border:1px solid var(--border-color-primary,#eee);">
  <div style="font-weight:600;font-size:14px;margin-bottom:8px;">Pipeline Summary</div>
  <table style="width:100%;font-size:13px;border-collapse:collapse;">
    <tr style="border-bottom:1px solid #eee;">
      <td style="padding:4px 12px;color:#888;">Mode</td>
      <td style="padding:4px 12px;font-weight:600;">{mode}</td>
    </tr>
    <tr style="border-bottom:1px solid #eee;">
      <td style="padding:4px 12px;color:#888;">Total objects</td>
      <td style="padding:4px 12px;font-weight:600;">{len(objects)}</td>
    </tr>
    {rows}
  </table>
</div>''')

    def render(self) -> str:
        return "\n".join(self.entries)


# ─── Core Pipeline ───────────────────────────────────────────────

def run_pipeline(image_path, expert_prompt, boxes, progress=gr.Progress()):
    if not image_path:
        return None, "{}", "<p>⚠ Upload an image.</p>", None, None, None

    if not expert_prompt or not expert_prompt.strip():
        expert_prompt = (
            "Analyze this archaeological image from Pompeii. "
            "Identify all visible artifacts and architectural elements."
        )

    user_bboxes = boxes if boxes else None
    log = PipelineLogger()

    try:
        # ── VLM ──
        log.step(1, "Visual Language Model")
        log.info(f"Model: <b>{vlm.model}</b>")
        log.info(f"Prompt: {expert_prompt[:100]}...")

        if user_bboxes:
            n_labeled = sum(1 for b in user_bboxes if b.get("label"))
            n_auto = len(user_bboxes) - n_labeled
            log.info(f"Input: <b>{len(user_bboxes)}</b> user-drawn boxes ({n_labeled} labeled, {n_auto} auto)")
        else:
            log.info("Input: full scene — VLM will auto-detect objects")

        progress(0.15, desc="VLM analyzing...")
        vlm_result = vlm.analyze_image(image_path, expert_prompt, user_bboxes=user_bboxes)

        objects = vlm_result.get("detected_objects", [])
        log.info(f"Scene: {vlm_result.get('scene_description', '')[:120]}")
        log.info(f"Detected: <b>{len(objects)} object(s)</b>")

        if not objects:
            log.error("No objects detected in the image.")
            return None, "{}", log.render(), None, None, None

        # ── Dispatch ──
        progress(0.30, desc="Dispatching...")
        log.step(2, "Agentic Dispatch")
        dispatched = dispatcher.dispatch(objects)
        for domain, objs in dispatched.items():
            if objs:
                emoji = MACRO_LABELS.get(domain, {}).get("emoji", "📦")
                log.info(f"{emoji} <b>{domain}</b>: {len(objs)} object(s)")

        # ── RAG + Refinement ──
        progress(0.45, desc="RAG + refinement...")
        log.step(3, "Dual RAG Retrieval")
        log.step(4, "LLM Refinement")

        all_metadata = []
        total = len(objects)
        for i, obj in enumerate(objects):
            domain = obj.get("assigned_domain", obj.get("domain", "ceramics"))
            obj_label = obj.get("label", "object")
            bbox_src = obj.get("bbox_source", "vlm")

            progress(0.45 + (0.35 * i / max(total, 1)), desc=f"{obj_label}...")

            # Build RAG query from visual description + features (not label)
            vis_desc = obj.get("visual_description", obj.get("description", ""))
            if not isinstance(vis_desc, str):
                vis_desc = str(vis_desc)
            key_feats = " ".join(str(f) for f in obj.get("key_features", []))
            user_hint = obj.get("user_label_hint", "")
            query = f"{vis_desc} {key_feats} {user_hint} {domain} Pompeii"

            rag_results = rag.query(domain, query)
            h_n = len(rag_results.get("historical_context", []))
            c_n = len(rag_results.get("cataloguing_rules", []))

            metadata = refiner.refine(obj, domain, rag_results)
            cat_id = metadata.get("catalogue_id", "?")
            all_metadata.append(metadata)

            log.object_card(
                idx=i + 1, total=total, label=f"{cat_id} — {obj_label}",
                domain=domain, bbox_src=bbox_src,
                hist_n=h_n, cat_n=c_n, meta_n=len(metadata),
            )

        # ── Export ──
        progress(0.85, desc="Exporting...")
        log.step(5, "Export & Annotation")

        annotated = annotator.annotate_image(image_path, objects, all_metadata)
        coco = annotator.export_coco_json(image_path, objects, all_metadata)
        csv_p = annotator.export_csv(image_path, objects, all_metadata)
        meta_p = annotator.save_metadata(all_metadata)

        log.info(f"Annotated image: <code>{Path(annotated).name}</code>")
        log.info(f"COCO JSON: <code>{Path(coco).name}</code>")
        log.info(f"CSV: <code>{Path(csv_p).name}</code>")
        log.info(f"Metadata: <code>{Path(meta_p).name}</code>")

        log.divider()
        log.summary_table(objects, bool(user_bboxes))
        log.success(f"Pipeline complete — {len(objects)} objects processed")

        progress(1.0, desc="Done!")

        return (
            annotated,
            json.dumps(all_metadata, indent=2, default=str),
            log.render(),
            coco, csv_p, meta_p,
        )

    except Exception as e:
        log.error(f"Pipeline error: {str(e)}")
        import traceback
        log.info(f"<pre>{traceback.format_exc()}</pre>")
        return None, "{}", log.render(), None, None, None


def get_kb_status():
    stats = rag.get_store_stats()
    lines = ["Knowledge Base Status", "=" * 40]
    for store, count in sorted(stats.items()):
        icon = "✓" if count > 0 else "○"
        lines.append(f"  [{icon}] {store}: {count} chunks")
    lines.append("")
    lines.append("Place PDFs in knowledge_bases/ → python indexer.py")
    return "\n".join(lines)


# ─── Gradio UI ───────────────────────────────────────────────────

DESCRIPTION = """
# 🏛 Pompeii Multimodal Archaeological Framework

### How to use:
1. **Upload** an archaeological image
2. **Click twice** on the image to draw a bounding box (1st corner → 2nd corner)
3. **Pick a category**: 🏺 Ceramics, 🎨 Paintings, 🏛️ Architecture — or **Skip** for auto
4. Repeat for all objects of interest
5. Write your **expert prompt** → click **Run Analysis**

> 💡 **No boxes?** Just upload + prompt → the VLM auto-detects everything.
"""

with gr.Blocks(title="Pompeii Multimodal Framework") as app:

    boxes_state = gr.State([])
    pending_state = gr.State(None)

    gr.Markdown(DESCRIPTION)

    with gr.Row():

        # ═══ LEFT ═══════════════════════════════
        with gr.Column(scale=1):

            input_image = gr.Image(
                label="1 · Upload image",
                type="filepath",
                height=100,
                sources=["upload", "clipboard"],
            )

            gr.Markdown("### 2 · Draw bounding boxes *(click on the image)*")

            drawing_image = gr.Image(
                label="Click: 1st corner → 2nd corner → assign category",
                type="pil", height=400, interactive=False,
            )

            bbox_status = gr.Markdown("Upload an image to start.")

            # ── Macro label buttons (shown after 2nd click) ──
            with gr.Group(visible=False) as label_row:
                gr.Markdown("**What is this region?**")
                with gr.Row():
                    btn_ceramics = gr.Button("🏺 Ceramics", size="lg", variant="secondary", scale=1)
                    btn_paintings = gr.Button("🎨 Paintings", size="lg", variant="secondary", scale=1)
                    btn_architecture = gr.Button("🏛️ Architecture", size="lg", variant="secondary", scale=1)
                with gr.Row():
                    custom_label_input = gr.Textbox(
                        label="Or type a custom label",
                        placeholder="e.g. mosaic, bronze_statue, inscription",
                        scale=3,
                    )
                    confirm_custom_btn = gr.Button("✓ Assign", variant="primary", size="sm", scale=1)
                    skip_btn = gr.Button("⏭ Skip (auto)", size="sm", scale=1)

            with gr.Row():
                undo_btn = gr.Button("↩ Undo last", size="sm")
                clear_btn = gr.Button("🗑 Clear all", size="sm")

            gr.Markdown("### 3 · Expert prompt")
            expert_prompt = gr.Textbox(
                label="",
                placeholder="e.g., Room 31, House VII.4, Pompeii. Final phase before 79 AD.",
                lines=3,
            )

            run_btn = gr.Button("▶ Run Analysis", variant="primary", size="lg")

            with gr.Accordion("KB Status", open=False):
                kb_box = gr.Textbox(value=get_kb_status, label="", lines=10, interactive=False)
                gr.Button("↻ Refresh", size="sm").click(fn=get_kb_status, outputs=kb_box)

        # ═══ RIGHT ══════════════════════════════
        with gr.Column(scale=1):

            output_image = gr.Image(label="Annotated Result", height=400)

            with gr.Tabs():
                with gr.TabItem("Metadata (JSON)"):
                    output_meta = gr.Code(label="Structured Metadata", language="json", lines=22)
                with gr.TabItem("Pipeline Evaluation"):
                    output_log = gr.HTML(
                        label="Pipeline Log",
                        value='<div style="padding:20px;text-align:center;color:#aaa;">Run the pipeline to see results here.</div>',
                    )

            gr.Markdown("### 📥 Download")
            with gr.Row():
                dl_coco = gr.File(label="COCO JSON")
                dl_csv = gr.File(label="CSV")
                dl_meta = gr.File(label="Metadata")

    # ═══ Events ═════════════════════════════════

    all_outputs = [drawing_image, boxes_state, pending_state, bbox_status, label_row, custom_label_input]

    input_image.change(fn=on_image_upload, inputs=[input_image], outputs=all_outputs)
    drawing_image.select(fn=on_image_click, inputs=[input_image, boxes_state, pending_state], outputs=all_outputs)

    # 3 macro domain buttons
    for btn, domain in [(btn_ceramics, "ceramics"), (btn_paintings, "paintings"), (btn_architecture, "architecture")]:
        btn.click(
            fn=lambda img, bx, pnd, d=domain: assign_macro_label(img, bx, pnd, d),
            inputs=[input_image, boxes_state, pending_state],
            outputs=all_outputs,
        )

    confirm_custom_btn.click(
        fn=confirm_custom_label,
        inputs=[input_image, boxes_state, pending_state, custom_label_input],
        outputs=all_outputs,
    )
    skip_btn.click(fn=skip_label, inputs=[input_image, boxes_state, pending_state], outputs=all_outputs)
    undo_btn.click(fn=undo_last_box, inputs=[input_image, boxes_state], outputs=all_outputs)
    clear_btn.click(fn=clear_all_boxes, inputs=[input_image], outputs=all_outputs)

    run_btn.click(
        fn=run_pipeline,
        inputs=[input_image, expert_prompt, boxes_state],
        outputs=[output_image, output_meta, output_log, dl_coco, dl_csv, dl_meta],
    )

    gr.Examples(
        examples=[
            [None, "Ceramic fragments from Regio V. Identify amphora types."],
            [None, "Classify the wall painting according to Mau's four styles."],
            [None, "Identify the opus type and estimate the construction period."],
        ],
        inputs=[input_image, expert_prompt],
        label="Example Prompts",
    )


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
    )
