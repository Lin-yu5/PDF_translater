import os
import sys
import time
import json
import re
from collections import Counter, defaultdict
import pymupdf
from rapidocr_onnxruntime import RapidOCR

PDF_PATH = r"c:\Users\james\python腳本\PDF_translater\CP.pdf"

def audit():
    doc = pymupdf.open(PDF_PATH)
    total_pages = len(doc)
    print(f"Total pages: {total_pages}")
    
    # 1. Document Structure
    dims = Counter()
    rotations = Counter()
    all_fonts = Counter()
    all_colorspaces = Counter()
    drawings_per_page = []
    
    # 2. Text Layer Analysis
    total_text_blocks = 0
    total_image_blocks = 0
    empty_pages = []
    pure_image_pages = []
    pure_text_pages = []
    mixed_pages = []
    
    single_line_blocks = 0
    multi_line_blocks = 0
    broken_hyphen_count = 0
    broken_hyphen_examples = []
    symbol_formula_pages = defaultdict(list)
    font_sizes = []
    
    # Track image counts per page
    images_per_page = []
    raw_images_per_page = []
    
    # Edge case candidates
    potential_glitch_pages = []
    tight_bbox_blocks = []
    
    for page_idx in range(total_pages):
        page = doc[page_idx]
        p_num = page_idx + 1
        rect = page.rect
        dims[(round(rect.width, 2), round(rect.height, 2))] += 1
        rotations[page.rotation] += 1
        
        drawings = page.get_drawings()
        drawings_per_page.append(len(drawings))
        
        # Colorspaces & Fonts
        page_fonts = page.get_fonts()
        for f in page_fonts:
            font_type = f[2]
            basefont = f[3]
            all_fonts[f"{font_type}: {basefont}"] += 1
            
        page_raw_imgs = page.get_images()
        raw_images_per_page.append(len(page_raw_imgs))
        for img in page_raw_imgs:
            xref = img[0]
            try:
                base_img = doc.extract_image(xref)
                cs = base_img.get("colorspace", "unknown")
                all_colorspaces[cs] += 1
            except Exception:
                pass
                
        # Text dict analysis
        page_dict = page.get_text("dict")
        blocks = page_dict.get("blocks", [])
        
        page_t_blocks = 0
        page_i_blocks = 0
        
        for b in blocks:
            b_type = b.get("type", 0)
            if b_type == 0: # text
                page_t_blocks += 1
                lines = b.get("lines", [])
                if len(lines) == 1:
                    single_line_blocks += 1
                elif len(lines) > 1:
                    multi_line_blocks += 1
                    
                # Font size & text inspection
                for l_idx, l in enumerate(lines):
                    for span in l.get("spans", []):
                        font_sizes.append(round(span.get("size", 0), 1))
                    line_text = "".join(span.get("text", "") for span in l.get("spans", []))
                    if line_text.endswith("-") and l_idx < len(lines) - 1:
                        broken_hyphen_count += 1
                        broken_hyphen_examples.append((p_num, line_text))
                    
                    # Math / Greek / super-sub scripts
                    syms = re.findall(r'[\u0370-\u03ff\u2100-\u214f\u2200-\u22ff\u00b1\u00b2\u00b3\u00bc-\u00be\u2070-\u209f±≤≥≠≈~•]', line_text)
                    if syms:
                        symbol_formula_pages[p_num].extend(syms)
                        
                # Check bounding box height vs font size
                bbox = b["bbox"]
                b_height = bbox[3] - bbox[1]
                b_width = bbox[2] - bbox[0]
                if b_height < 10 or b_width < 10:
                    tight_bbox_blocks.append((p_num, bbox, b.get("lines", [{}])[0].get("spans", [{}])[0].get("text", "")))
            elif b_type == 1: # image
                page_i_blocks += 1
                
        total_text_blocks += page_t_blocks
        total_image_blocks += page_i_blocks
        images_per_page.append(page_i_blocks)
        
        if page_t_blocks == 0 and (page_i_blocks > 0 or len(page_raw_imgs) > 0):
            pure_image_pages.append(p_num)
        elif page_t_blocks == 0 and page_i_blocks == 0 and len(page_raw_imgs) == 0:
            empty_pages.append(p_num)
        elif page_t_blocks > 0 and (page_i_blocks > 0 or len(page_raw_imgs) > 0):
            mixed_pages.append(p_num)
        else:
            pure_text_pages.append(p_num)
            
        # Check text-image overlap
        t_bboxes = [pymupdf.Rect(b["bbox"]) for b in blocks if b.get("type") == 0]
        i_bboxes = [pymupdf.Rect(b["bbox"]) for b in blocks if b.get("type") == 1]
        
        overlap_count = 0
        for tb in t_bboxes:
            for ib in i_bboxes:
                if tb.intersects(ib):
                    inter = tb & ib
                    if inter.width * inter.height > 20:
                        overlap_count += 1
                        
        if overlap_count > 0 or len(drawings) > 100:
            potential_glitch_pages.append({
                "page": p_num,
                "text_blocks": page_t_blocks,
                "img_blocks": page_i_blocks,
                "raw_images": len(page_raw_imgs),
                "overlaps": overlap_count,
                "drawings": len(drawings)
            })

    print("=== 1. DOCUMENT STRUCTURE ===")
    print("Dimensions:", dims)
    print("Rotations:", rotations)
    print("Color Spaces:", all_colorspaces)
    print("Embedded Font Types (Total unique font entries):", len(all_fonts))
    for f, count in all_fonts.most_common(12):
        print(f"  {f}: {count} occurrences")
        
    print("\n=== 2. TEXT LAYER ANALYSIS ===")
    print(f"Total text blocks: {total_text_blocks}")
    print(f"Single-line blocks: {single_line_blocks} ({round(single_line_blocks/total_text_blocks*100, 1)}%)")
    print(f"Multi-line blocks: {multi_line_blocks} ({round(multi_line_blocks/total_text_blocks*100, 1)}%)")
    print(f"Pure text pages ({len(pure_text_pages)}): {pure_text_pages}")
    print(f"Pure image/scan pages ({len(pure_image_pages)}): {pure_image_pages}")
    print(f"Mixed pages (text + image) ({len(mixed_pages)}): {mixed_pages}")
    print(f"Completely empty pages ({len(empty_pages)}): {empty_pages}")
    print(f"Broken hyphenations: {broken_hyphen_count}")
    for ex in broken_hyphen_examples:
        print(f"  Page {ex[0]}: {ex[1]}")
    print(f"Pages with symbols/bullet/math chars: {len(symbol_formula_pages)}")
    font_size_counts = Counter(font_sizes)
    print(f"Font size distribution (top 8): {font_size_counts.most_common(8)}")
    print(f"Min font size: {min(font_sizes)}, Max font size: {max(font_sizes)}")

    print("\n=== 3. IMAGE & OCR BENCHMARK ===")
    print(f"Total image blocks (dict type=1): {total_image_blocks}")
    print(f"Total raw images (get_images): {sum(raw_images_per_page)}")
    print(f"Total vector drawings across document: {sum(drawings_per_page)}")
    
    ocr = RapidOCR()
    sample_pages = [8, 13, 16, 21, 37, 45, 59, 75, 89, 94]
    ocr_results = {}
    
    for p_num in sample_pages:
        p = doc[p_num - 1]
        
        # Test full-page OCR at 150 DPI
        t0 = time.time()
        pix = p.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        t_render = time.time() - t0
        
        t1 = time.time()
        result, elapse = ocr(img_bytes)
        t_ocr = time.time() - t1
        
        num_detected = len(result) if result else 0
        avg_conf = (sum(float(r[2]) for r in result) / num_detected) if num_detected > 0 else 0
        
        # Also test OCR on individual image blocks if present
        p_dict = p.get_text("dict")
        img_blocks = [b for b in p_dict["blocks"] if b.get("type") == 1]
        text_blocks = [b for b in p_dict["blocks"] if b.get("type") == 0]
        
        img_block_ocr = []
        for idx, ib in enumerate(img_blocks):
            ib_rect = pymupdf.Rect(ib["bbox"])
            # render clip at 300 dpi as proposed in coordinator spec
            t_ib0 = time.time()
            pix_clip = p.get_pixmap(clip=ib_rect, dpi=300)
            res_clip, _ = ocr(pix_clip.tobytes("png"))
            t_clip_ocr = time.time() - t_ib0
            img_block_ocr.append({
                "block_idx": idx,
                "bbox": [round(x, 1) for x in ib["bbox"]],
                "detected_count": len(res_clip) if res_clip else 0,
                "time_ms": round(t_clip_ocr * 1000, 1),
                "texts": [r[1] for r in (res_clip or [])[:3]]
            })
            
        ocr_results[p_num] = {
            "page_width": p.rect.width,
            "page_height": p.rect.height,
            "render_ms": round(t_render * 1000, 1),
            "full_page_ocr_ms": round(t_ocr * 1000, 1),
            "full_page_detected": num_detected,
            "avg_confidence": round(avg_conf, 3),
            "sample_texts": [r[1] for r in (result or [])[:5]],
            "text_blocks_in_pdf": len(text_blocks),
            "img_blocks_in_pdf": len(img_blocks),
            "raw_imgs_in_pdf": len(p.get_images()),
            "clip_ocr": img_block_ocr
        }
        print(f"Page {p_num:02d}: FullPage OCR {round(t_ocr*1000, 1)}ms ({num_detected} items, conf={round(avg_conf, 2)}). Img blocks in PDF: {len(img_blocks)}, Text blocks: {len(text_blocks)}")
        if img_block_ocr:
            for c in img_block_ocr:
                print(f"   -> Block bbox {c['bbox']}: {c['detected_count']} texts in {c['time_ms']}ms: {c['texts']}")

    print("\n=== 4. EDGE CASES & POTENTIAL BUGS ===")
    print(f"Pages with text-image overlap: {len(potential_glitch_pages)}")
    for pg in potential_glitch_pages[:10]:
        print(f"  Page {pg['page']}: {pg['text_blocks']} text blocks, {pg['img_blocks']} img blocks, {pg['overlaps']} overlaps, {pg['drawings']} vector drawings")
    print(f"Tight bbox blocks (<10pt): {len(tight_bbox_blocks)}")

    report = {
        "dimensions": {str(k): v for k, v in dims.items()},
        "rotations": {str(k): v for k, v in rotations.items()},
        "colorspaces": dict(all_colorspaces),
        "fonts": dict(all_fonts),
        "total_text_blocks": total_text_blocks,
        "single_line_blocks": single_line_blocks,
        "multi_line_blocks": multi_line_blocks,
        "pure_text_pages": pure_text_pages,
        "pure_image_pages": pure_image_pages,
        "mixed_pages": mixed_pages,
        "empty_pages": empty_pages,
        "broken_hyphen_count": broken_hyphen_count,
        "broken_hyphen_examples": broken_hyphen_examples,
        "symbol_formula_page_count": len(symbol_formula_pages),
        "total_image_blocks": total_image_blocks,
        "total_raw_images": sum(raw_images_per_page),
        "total_drawings": sum(drawings_per_page),
        "ocr_sample_pages": ocr_results,
        "potential_glitch_pages": potential_glitch_pages,
        "tight_bbox_count": len(tight_bbox_blocks)
    }
    
    with open("audit_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    print("\nAudit completed and saved to audit_report.json")

if __name__ == "__main__":
    audit()
