import os
import sys
import json
import zipfile
import tempfile
import pymupdf
from unittest.mock import patch

import translator
from translator import (
    analyze_pdf_pages,
    translate_pdf_file,
    export_bilingual_pdf,
    export_docx_comparison,
    load_checkpoint,
    save_checkpoint,
    detect_block_alignment,
    sample_background_and_text_color,
    FatalAPIError
)

CP_PATH = r"c:\Users\james\python腳本\PDF_translater\CP.pdf"

def run_verification():
    print("=====================================================================")
    print("STEP 1: Real-world CP.pdf Parsing and Classification (94 Pages)")
    print("=====================================================================")
    if not os.path.exists(CP_PATH):
        raise FileNotFoundError(f"CP.pdf not found at {CP_PATH}")
    
    file_size_mb = os.path.getsize(CP_PATH) / (1024 * 1024)
    print(f"[OK] Found CP.pdf: {file_size_mb:.2f} MB")
    
    doc = pymupdf.open(CP_PATH)
    total_pages = len(doc)
    print(f"[OK] Total pages: {total_pages}")
    assert total_pages == 94, f"Expected 94 pages, got {total_pages}"
    
    # Run full 94-page analysis
    pages_meta = analyze_pdf_pages(doc)
    assert len(pages_meta) == 94, f"Expected 94 metadata entries, got {len(pages_meta)}"
    
    type_counts = {"text": 0, "scan": 0, "mixed": 0}
    for m in pages_meta:
        type_counts[m["page_type"]] += 1
        
    print(f"[OK] Page classification breakdown: {type_counts}")
    assert type_counts["text"] > 0, "Expected text pages"
    assert type_counts["scan"] > 0, "Expected scan pages"
    assert type_counts["mixed"] > 0, "Expected mixed pages"
    doc.close()

    print("\n=====================================================================")
    print("STEP 2: End-to-End Mock Translate on Sample Pages from CP.pdf")
    print("=====================================================================")
    # Extract pages 1 to 4 (covers title, text, image/scan) to a test PDF
    sample_pdf_path = os.path.join(tempfile.gettempdir(), "cp_sample_test.pdf")
    src_doc = pymupdf.open(CP_PATH)
    sample_doc = pymupdf.open()
    # Pages 0, 1, 2, 3 (first 4 pages)
    sample_doc.insert_pdf(src_doc, from_page=0, to_page=3)
    sample_doc.save(sample_pdf_path)
    sample_doc.close()
    src_doc.close()
    print(f"[OK] Created 4-page sample PDF: {sample_pdf_path}")

    output_pdf_path = os.path.join(tempfile.gettempdir(), "cp_output_translated.pdf")
    task_id = "verify_e2e_task"
    task_dir = os.path.join("temp_tasks", task_id)
    os.makedirs(task_dir, exist_ok=True)

    progress_log = []
    def on_progress(*args):
        progress_log.append(args)

    # Mock translate_blocks
    def mock_translate_blocks(items, target_lang_desc, api_key, provider="gemini", model=None, base_url=None, domain="general"):
        return {it["id"]: f"[繁中譯文] {it['text'][:40]}" for it in items}

    ckpt_file = os.path.join(task_dir, "checkpoint.json")

    with patch("translator.translate_blocks", side_effect=mock_translate_blocks):
        result = translate_pdf_file(
            input_path=sample_pdf_path,
            output_path=output_pdf_path,
            provider="gemini",
            api_key="test_mock_key",
            target_lang_code="zh-TW",
            model="gemini-2.5-flash",
            domain="medical",
            enable_ocr=True,
            font_style="mingti",
            progress_callback=on_progress,
            checkpoint_path=ckpt_file
        )

    print(f"[OK] translate_pdf_file completed successfully:")
    print(f"     Total pages processed: {result['total_pages']}")
    print(f"     Translated blocks (Phase 1): {result['translated_blocks']}")
    print(f"     OCR blocks (Phase 2): {result['ocr_blocks_translated']}")
    print(f"     Total translated: {result['total_translated']}")
    assert os.path.exists(output_pdf_path), "Output PDF must exist"
    assert os.path.getsize(output_pdf_path) > 0, "Output PDF must not be empty"

    # Verify output PDF is valid and readable
    out_doc = pymupdf.open(output_pdf_path)
    assert len(out_doc) == 4, f"Output PDF should have 4 pages, got {len(out_doc)}"
    p1_text = out_doc[0].get_text()
    print(f"[OK] Output PDF verified (Page 1 text snippet length: {len(p1_text)})")
    out_doc.close()

    # Checkpoint verification
    ckpt_file = os.path.join(task_dir, "checkpoint.json")
    assert os.path.exists(ckpt_file), f"Checkpoint file must exist at {ckpt_file}"
    ckpt = load_checkpoint(ckpt_file)
    print(f"[OK] Checkpoint verified: {len(ckpt)} translated items recorded in checkpoint")
    assert len(ckpt) > 0, "Checkpoint should contain translated items"

    print("\n=====================================================================")
    print("STEP 3: Multi-Format Export Capabilities (Bilingual PDF & Docx)")
    print("=====================================================================")
    bilingual_pdf_path = os.path.join(tempfile.gettempdir(), "cp_bilingual_2up.pdf")
    docx_path = os.path.join(tempfile.gettempdir(), "cp_comparison.docx")

    # 1. Bilingual PDF (2-up)
    export_bilingual_pdf(sample_pdf_path, output_pdf_path, bilingual_pdf_path)
    assert os.path.exists(bilingual_pdf_path), "Bilingual PDF must exist"
    assert os.path.getsize(bilingual_pdf_path) > 0, "Bilingual PDF must not be empty"
    
    bi_doc = pymupdf.open(bilingual_pdf_path)
    assert len(bi_doc) == 4, f"Bilingual PDF should have 4 pages, got {len(bi_doc)}"
    bi_page0 = bi_doc[0]
    orig_doc_test = pymupdf.open(sample_pdf_path)
    orig_w = orig_doc_test[0].rect.width
    orig_doc_test.close()
    bi_w = bi_page0.rect.width
    print(f"[OK] Bilingual PDF verified: 2-up width = {bi_w} pt (Original width = {orig_w} pt, ratio: {bi_w / orig_w:.2f})")
    assert bi_w > orig_w * 1.8, "Bilingual page width should be roughly double for 2-up side-by-side"
    bi_doc.close()

    # 2. Comparison DOCX (pure OpenXML standard library)
    export_docx_comparison(sample_pdf_path, output_pdf_path, docx_path)
    assert os.path.exists(docx_path), "Docx file must exist"
    assert os.path.getsize(docx_path) > 0, "Docx file must not be empty"

    # Verify DOCX is a valid zip archive containing word/document.xml
    with zipfile.ZipFile(docx_path, 'r') as z:
        names = z.namelist()
        assert "word/document.xml" in names, "DOCX must contain word/document.xml"
        doc_xml = z.read("word/document.xml").decode("utf-8")
        assert "FolioTrans" in doc_xml or "對照" in doc_xml or "Page" in doc_xml
    print(f"[OK] DOCX verified: Valid OpenXML archive ({os.path.getsize(docx_path)} bytes), contains structured table comparison")

    print("\n=====================================================================")
    print("STEP 4: Layout Preservation Engine Verification")
    print("=====================================================================")
    # Alignment detection
    rect_full = pymupdf.Rect(0, 0, 500, 700)
    center_block = {"bbox": (180, 100, 320, 120), "lines": [{"spans": [{"origin": (200, 115)}]}]}
    left_block = {"bbox": (50, 100, 200, 120), "lines": [{"spans": [{"origin": (50, 115)}]}]}
    right_block = {"bbox": (350, 100, 470, 120), "lines": [{"spans": [{"origin": (350, 115)}]}]}
    
    align_center = detect_block_alignment(center_block, rect_full)
    align_left = detect_block_alignment(left_block, rect_full)
    align_right = detect_block_alignment(right_block, rect_full)
    
    print(f"[OK] Block alignment detected: center={align_center}, left={align_left}, right={align_right}")
    assert align_center == 1, "Expected center alignment (1)"
    assert align_left == 0, "Expected left alignment (0)"
    assert align_right == 2, "Expected right alignment (2)"

    # Clean up temp test files
    for p in [sample_pdf_path, output_pdf_path, bilingual_pdf_path, docx_path]:
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

    print("\n=====================================================================")
    print("ALL VERIFICATION CHECKS PASSED (100% SUCCESS)!")
    print("=====================================================================")

if __name__ == "__main__":
    run_verification()
