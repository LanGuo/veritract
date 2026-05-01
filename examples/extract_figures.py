"""
Example: extract structured data from PDF figures using a multimodal model.

Each figure is passed as an image to extract_raw. Grounding runs against the
full document text so values visible in figures but discussed in the body
(rather than only in the caption) can be verified.

Requires: pip install 'veritract[pdf]'
Requires: Ollama running with a multimodal model, e.g. gemma4:e4b

Saves figure_extraction_results.html for visual verification.
"""
import base64, io
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

from veritract import LLMClient
from veritract.extraction import extract_raw, ground

llm = LLMClient(model="gemma4:e4b", temperature=0.0, seed=42)

FIGURE_SCHEMA = {
    "type": "object",
    "properties": {
        "figure_type":     {"type": "string"},
        "models_compared": {"type": "string"},
        "metric":          {"type": "string"},
        "best_result":     {"type": "string"},
        "key_finding":     {"type": "string"},
    },
    "required": ["figure_type", "models_compared", "metric", "best_result", "key_finding"],
}

PROMPT = """You are reading a figure from an ML research paper.
Rules:
- Extract only what is directly visible in the figure or stated in the caption.
- Copy exact text labels as they appear. Do not paraphrase or use prior knowledge.
- If a field is not visible or stated, return an empty string.

Fields:
* figure_type: Type of visual — bar chart, line plot, scatter plot, architecture diagram, table, etc.
* models_compared: Model names or systems shown in the figure (comma-separated as labelled).
* metric: What is being measured or plotted (exact axis label or caption description).
* best_result: The highest or best numerical result visible, with the model name it belongs to.
* key_finding: The main takeaway stated in the caption or clearly shown in the figure."""


def _pil_to_b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _load_pdf(path: str):
    """Convert PDF to markdown and extract figures with captions."""
    opts = PdfPipelineOptions()
    opts.generate_picture_images = True
    opts.images_scale = 2.0
    conv = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    doc = conv.convert(str(path))
    full_text = doc.document.export_to_markdown()
    figures = []
    for pic in doc.document.pictures:
        img = pic.get_image(doc.document)
        caption = pic.caption_text(doc.document) or ""
        if img is None or img.width < 200:
            continue
        figures.append((img, caption))
    return full_text, figures


def extract_figures(path: str, llm, *, schema=None, prompt=None):
    """
    Extract structured fields from every figure in a PDF.

    Grounding runs against the full document text so values visible in a figure
    but discussed anywhere in the paper body can be verified — not just values
    mentioned in the caption.

    Returns list of (image, caption, ExtractionResult).
    """
    schema = schema or FIGURE_SCHEMA
    prompt = prompt or PROMPT
    doc_id = Path(path).name
    full_text, figures = _load_pdf(path)

    results = []
    for img, caption in figures:
        b64 = _pil_to_b64(img)
        # Prompt sees the caption; grounding searches the full document.
        raw = extract_raw(
            caption or f"Figure from {doc_id}",
            schema, llm,
            prompt=prompt,
            images=[b64],
            source_type="figure",
            doc_id=doc_id,
        )
        # Override source_text so grounding searches the full doc, not just the caption.
        raw.source_text = full_text
        result = ground(raw, llm=None, mode="fuzzy")
        results.append((img, caption, result))
    return results


def save_html(all_results: dict[str, list], output_path: str = "figure_extraction_results.html"):
    """Save figures and extraction results to an HTML file for visual verification."""
    def _result_rows(result):
        rows = []
        for field, gf in result.extracted.items():
            if not gf["value"]:
                continue
            ptype = gf["span"]["provenance_type"] if gf["span"] else "no-span"
            rows.append((field, gf["value"], ptype))
        for q in result.quarantined:
            if q["value"]:
                rows.append((q["field_name"], q["value"], "QUARANTINED"))
        return rows

    sections = []
    for name, fig_results in all_results.items():
        fig_blocks = []
        for i, (img, caption, result) in enumerate(fig_results):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()

            rows = _result_rows(result)
            rows_html = "\n".join(
                f"<tr class='{ptype.lower().replace('-','_')}'>"
                f"<td>{field}</td><td>{value}</td><td class='ptype'>{ptype}</td></tr>"
                for field, value, ptype in rows
            ) or "<tr><td colspan='3'><em>no fields extracted</em></td></tr>"

            caption_html = (
                f"<p class='caption'>{caption}</p>" if caption
                else "<p class='caption dim'>no caption</p>"
            )
            fig_blocks.append(f"""
        <div class="figure-card">
          <div class="fig-header">Figure {i+1} &nbsp;·&nbsp; {img.width}×{img.height}px</div>
          <div class="fig-body">
            <img src="data:image/png;base64,{b64}" style="max-width:100%;border:1px solid #ddd;border-radius:4px;">
            {caption_html}
            <table>
              <thead><tr><th>field</th><th>extracted value</th><th>provenance</th></tr></thead>
              <tbody>{rows_html}</tbody>
            </table>
          </div>
        </div>""")
        sections.append(f"<section><h2>{name}</h2>{''.join(fig_blocks)}</section>")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>veritract — figure extraction results</title>
<style>
  body{{font-family:system-ui,sans-serif;max-width:960px;margin:40px auto;padding:0 20px;color:#222}}
  h1{{font-size:1.4rem;border-bottom:2px solid #222;padding-bottom:8px}}
  h2{{font-size:1.1rem;margin-top:2.5rem;color:#444;border-bottom:1px solid #ccc;padding-bottom:4px}}
  .figure-card{{margin:1.5rem 0;border:1px solid #e0e0e0;border-radius:6px;overflow:hidden}}
  .fig-header{{background:#f5f5f5;padding:6px 12px;font-size:.85rem;font-weight:600;color:#555}}
  .fig-body{{padding:12px}}
  .caption{{font-size:.82rem;color:#555;margin:8px 0;font-style:italic}}
  .dim{{color:#aaa}}
  table{{width:100%;border-collapse:collapse;margin-top:10px;font-size:.82rem}}
  th{{background:#f0f0f0;text-align:left;padding:5px 8px;border-bottom:2px solid #ccc}}
  td{{padding:4px 8px;border-bottom:1px solid #eee;vertical-align:top;word-break:break-word}}
  .ptype{{font-family:monospace;font-size:.78rem;white-space:nowrap}}
  tr.direct td{{background:#f0fff0}}
  tr.paraphrased td{{background:#f5f9ff}}
  tr.no_span td{{background:#fffef0}}
  tr.quarantined td{{background:#fff5f5}}
  .legend{{font-size:.78rem;margin-bottom:1rem}}
  .legend span{{display:inline-block;padding:2px 8px;border-radius:3px;margin-right:6px}}
</style></head>
<body>
<h1>veritract — figure extraction · gemma4:e4b · grounded against full document text</h1>
<div class="legend">
  <span style="background:#f0fff0">direct</span>
  <span style="background:#f5f9ff">paraphrased</span>
  <span style="background:#fffef0">no-span (vision-only, not in doc text)</span>
  <span style="background:#fff5f5">QUARANTINED</span>
</div>
{''.join(sections)}
</body></html>"""

    Path(output_path).write_text(html)
    print(f"Saved → {Path(output_path).resolve()}")


if __name__ == "__main__":
    pdfs = {
        "Gemma 3":   "/tmp/llm_reports/gemma3.pdf",
        "Kimi K2.5": "/tmp/llm_reports/kimi_k2_5.pdf",
        "Qwen 3":    "/tmp/llm_reports/qwen3.pdf",
    }
    all_results = {}
    for name, path in pdfs.items():
        print(f"Processing {name}...")
        fig_results = extract_figures(path, llm)
        print(f"  {len(fig_results)} figures")
        all_results[name] = fig_results

    save_html(all_results)
