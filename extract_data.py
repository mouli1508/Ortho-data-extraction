#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Docling + EasyOCR extraction for large Korean / English medical PDFs on an NVIDIA L4.
Optimised for speed (skip OCR on born‑digital pages) and memory (batched rendering).
"""

from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime

import torch  # sanity‑check GPU visibility
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    EasyOcrOptions,
    AcceleratorOptions,
    AcceleratorDevice,
    TableFormerMode,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.settings import settings

from docling_core.types.doc import ImageRefMode

# ── Configuration ────────────────────────────────────────────────────────────
PDF_SOURCE_DIR = Path("medical_pdfs/bookData")
OUTPUT_DIR = Path("extracted_data/finalExtracted")

EXPORT_JSON = True    # dump structured JSON alongside Markdown
RECURSIVE   = False   # True → recurse into sub‑folders
# ─────────────────────────────────────────────────────────────────────────────


def log_result(result, pdf_name: str) -> None:
    """Print Docling page‑level warnings / errors (if any)."""
    if getattr(result, "warnings", None):
        print(f"⚠️  {pdf_name}: {len(result.warnings)} warnings")
        for w in result.warnings:
            print("    •", w)
    if getattr(result, "errors", None):
        print(f"🛑  {pdf_name}: {len(result.errors)} errors")
        for e in result.errors:
            print("    •", e)


def extract_pdf_data(pdf_path: Path,
                     output_root: Path,
                     converter: DocumentConverter) -> None:
    """Extract text, tables, and images from one PDF."""
    print(f"🔄  Processing: {pdf_path.relative_to(PDF_SOURCE_DIR)}")

    pdf_output_dir = output_root / pdf_path.stem
    # images_dir     = pdf_output_dir / "images"
    # images_dir.mkdir(parents=True, exist_ok=True)
    pdf_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = converter.convert(source=str(pdf_path))

        total_secs = (
            result.timings.get("pipeline_total").times[0]
            if "pipeline_total" in result.timings else None
        )

        # Markdown (human‑readable) export
        md_path      = pdf_output_dir / f"{pdf_path.stem}.md"
        # md_content = result.document.export_to_markdown()
        # md_path.write_text(md_content, encoding="utf-8")
        # result.document.save_as_markdown(md_path) # Using save_as_markdown for consistency
        result.document.save_as_markdown(
            md_path,
            artifacts_dir=pdf_output_dir / "images",   # write PNGs here
            image_mode=ImageRefMode.REFERENCED         # use normal ![](images/xxx.png) links
        )


        # JSON (machine‑readable) export
        if EXPORT_JSON:
            json_path = pdf_output_dir / f"{pdf_path.stem}.json"
            # result.document.save_as_json(json_path, image_mode=ImageRefMode.EMBEDDED)
            result.document.save_as_json(
                json_path,
                image_mode=ImageRefMode.REFERENCED,
                artifacts_dir=pdf_output_dir / "images"   # Docling will create & populate
            )

        log_result(result, pdf_path.name)
        print(f"✅  Done ({total_secs:.2f}s)" if total_secs else "✅  Done")

    except Exception as exc:
        print(f"❌  Failed on {pdf_path.name}: {exc}")


def build_converter() -> DocumentConverter:
    """Configure a Docling converter for bilingual OCR on CUDA."""
    ocr_cfg = EasyOcrOptions(
        lang=["ko", "en"],
        force_full_page_ocr=False,   # let Docling reuse embedded text when present
    )

    pipe_opts = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        ocr_options=ocr_cfg,
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CUDA),
        pages_per_batch=30,          # prevents GPU OOM on 3000‑page tomes
    )
    # High‑accuracy table mode (toggle to FAST if speed > fidelity)
    pipe_opts.table_structure_options.mode = TableFormerMode.ACCURATE

    # pipe_opts.num_cpu_workers = 4

    fmt_opts = {InputFormat.PDF: PdfFormatOption(pipeline_options=pipe_opts)}
    return DocumentConverter(format_options=fmt_opts)


def main() -> None:
    """Entry point: set up converter, iterate over PDFs, extract."""
    # Hard failure if CUDA isn’t visible (avoids silent CPU fallback)
    assert torch.cuda.is_available(), "CUDA not detected — check PyTorch/CUDA install"

    if not PDF_SOURCE_DIR.exists():
        raise FileNotFoundError(f"Source directory not found: {PDF_SOURCE_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    settings.debug.profile_pipeline_timings = True

    converter = build_converter()
    print("🚀  Converter ready – starting extraction…\n")

    pdf_iter = (
        PDF_SOURCE_DIR.rglob("*.pdf") if RECURSIVE
        else PDF_SOURCE_DIR.glob("*.pdf")
    )
    for pdf_file in pdf_iter:
        extract_pdf_data(pdf_file, OUTPUT_DIR, converter)

    print("\n✅  Extraction completed at",
          datetime.now().strftime("%Y‑%m‑%d %H:%M:%S"))


if __name__ == "__main__":
    # Uncomment if multiple GPUs and you want to pin to a specific one
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
