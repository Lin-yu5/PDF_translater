import json
import re
import os
import time
import requests
import pymupdf

class FatalAPIError(Exception):
    """Fatal API errors (invalid key, bad model, unauthorized) that should abort the pipeline."""
    pass

LANGUAGE_CONFIG = {
    "zh-TW": {"name": "繁體中文", "font": "china-ts", "desc": "Traditional Chinese"},
    "zh-CN": {"name": "簡體中文", "font": "china-ss", "desc": "Simplified Chinese"},
    "en": {"name": "英文", "font": "helv", "desc": "English"},
    "ja": {"name": "日文", "font": "japan", "desc": "Japanese"},
    "ko": {"name": "韓文", "font": "korea", "desc": "Korean"},
    "es": {"name": "西班牙文", "font": "helv", "desc": "Spanish"},
    "fr": {"name": "法文", "font": "helv", "desc": "French"},
    "de": {"name": "德文", "font": "helv", "desc": "German"},
}

def is_translatable(text: str) -> bool:
    """Check whether a text segment requires translation."""
    clean = text.strip()
    if len(clean) <= 1:
        return False
    # skip pure numbers, dates, version numbers or punctuation
    if re.fullmatch(r"[\d\.\,\-\:\/\s\(\)\[\]\%\$\#\@\+\=\*]+", clean):
        return False
    return True

def safe_json_parse(raw: str) -> dict:
    """Parse JSON string, stripping markdown code fences or preambles if present."""
    trimmed = raw.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", trimmed, re.DOTALL)
    if match:
        trimmed = match.group(1).strip()
    else:
        start = trimmed.find("{")
        end = trimmed.rfind("}")
        if start != -1 and end != -1 and end > start:
            trimmed = trimmed[start:end + 1].strip()
    return json.loads(trimmed)

def extract_translation_map(content_text: str) -> dict:
    """Parse LLM output JSON and normalize IDs to integers."""
    parsed = safe_json_parse(content_text)
    raw_items = parsed.get("translations", [])
    if isinstance(raw_items, dict):
        raw_items = [{"id": k, "text": v} for k, v in raw_items.items()]
    result = {}
    for item in raw_items:
        try:
            tid = int(item["id"])
            text = str(item.get("text", ""))
            result[tid] = text
        except (ValueError, TypeError, KeyError):
            continue
    return result

def _post_with_retry(url: str, headers: dict = None, json_payload: dict = None, timeout: int = 60, max_retries: int = 3) -> requests.Response:
    """Execute HTTP POST with fast-fail on fatal client errors and backoff on transient errors."""
    delay = 1.5
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=json_payload, timeout=timeout)
            # Fast-fail fatal client errors (invalid key, bad model name, forbidden)
            if resp.status_code in (400, 401, 403, 404):
                err_msg = resp.text[:200]
                raise FatalAPIError(f"API 致命錯誤 (HTTP {resp.status_code}): {err_msg}")
            
            # Transient rate limits or server errors (429, 500, 502, 503, 504)
            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {resp.status_code}: {resp.text[:100]}"
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                resp.raise_for_status()

            resp.raise_for_status()
            return resp
        except FatalAPIError:
            raise
        except requests.exceptions.RequestException as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                raise e
    raise RuntimeError(f"API 連線重試 {max_retries} 次後失敗: {last_err}")

def load_translation_rules(target_lang_desc: str) -> str:
    """Load translation skill rules from skills/translation_rules.md, with fallback."""
    skill_file = os.path.join(os.path.dirname(__file__), "skills", "translation_rules.md")
    rules_content = ""
    if os.path.exists(skill_file):
        try:
            with open(skill_file, "r", encoding="utf-8") as f:
                rules_content = f.read().strip()
        except Exception:
            pass

    if not rules_content:
        rules_content = (
            "You are an expert document translation engine embedded in an automated PDF pipeline.\n"
            "Rules:\n"
            "1. STRICT ZERO EXPLANATION: Output ONLY the translated text in JSON format. Never add notes, preambles, or commentary.\n"
            "2. 1:1 ID MAPPING: Retain the exact same 'id' for each item in the translations array.\n"
            "3. PRESERVE ARTIFACTS: Keep citations (e.g. [1]), numbers, acronyms, formulas, trademarks, and code unaltered.\n"
            "4. NATURAL FLUENCY: Professional tone, appropriate punctuation (full-width for CJK)."
        )

    return (
        f"{rules_content}\n\n"
        f"Target Language: {target_lang_desc}.\n"
        "Required Output Format: Valid JSON with schema: {\"translations\": [{\"id\": <int>, \"text\": \"<translated text>\"}]}."
    )

def call_gemini(items: list, target_lang_desc: str, api_key: str, model: str = "gemini-2.5-flash") -> dict:
    """Call Google Gemini API using structured JSON output with embedded skill rules."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    system_instruction = load_translation_rules(target_lang_desc)
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": f"{system_instruction}\n\nInput JSON to translate:\n{json.dumps(items, ensure_ascii=False)}"}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json"
        }
    }
    resp = _post_with_retry(url, json_payload=payload, timeout=60)
    data = resp.json()
    try:
        content_text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        candidates = data.get("candidates", [])
        if candidates and "finishReason" in candidates[0]:
            raise RuntimeError(f"Gemini 回應受阻 (finishReason: {candidates[0]['finishReason']})")
        raise RuntimeError(f"無法解析 Gemini API 回應格式: {str(data)[:200]}")
    return extract_translation_map(content_text)

def call_openai(items: list, target_lang_desc: str, api_key: str, model: str = "gpt-4o-mini", base_url: str = None) -> dict:
    """Call OpenAI or OpenAI-compatible API with embedded skill rules."""
    endpoint = f"{base_url.rstrip('/')}/chat/completions" if base_url else "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    system_msg = load_translation_rules(target_lang_desc)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Translate the text fields of this JSON array. Output strictly JSON:\n{json.dumps(items, ensure_ascii=False)}"}
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"}
    }
    resp = _post_with_retry(endpoint, headers=headers, json_payload=payload, timeout=60)
    data = resp.json()
    try:
        content_text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise RuntimeError(f"無法解析 OpenAI API 回應格式: {str(data)[:200]}")
    return extract_translation_map(content_text)

def translate_blocks(items: list, provider: str, api_key: str, target_lang_desc: str, model: str = None, base_url: str = None) -> dict:
    """Dispatch translation batch to selected LLM provider."""
    if not items:
        return {}
    if provider == "gemini":
        m = model or "gemini-2.5-flash"
        return call_gemini(items, target_lang_desc, api_key, model=m)
    elif provider == "openai":
        m = model or "gpt-4o-mini"
        return call_openai(items, target_lang_desc, api_key, model=m, base_url=base_url)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def translate_pdf_file(
    input_path: str,
    output_path: str,
    provider: str,
    api_key: str,
    target_lang_code: str = "zh-TW",
    model: str = None,
    base_url: str = None,
    batch_size: int = 20,
    progress_callback = None
) -> dict:
    """
    Translates a PDF file while preserving layout, structure, and styles.
    progress_callback receives: (current_page, total_pages, message)
    """
    lang_info = LANGUAGE_CONFIG.get(target_lang_code, LANGUAGE_CONFIG["zh-TW"])
    fontname = lang_info["font"]
    target_lang_desc = lang_info["desc"]

    doc = pymupdf.open(input_path)
    if doc.is_encrypted:
        doc.close()
        raise ValueError("此 PDF 文件受密碼保護，無法進行翻譯處理。")

    total_pages = len(doc)
    total_blocks_translated = 0

    # Pre-scan document to verify selectable text content
    total_translatable_blocks = 0
    for page in doc:
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") == 0:
                lines = block.get("lines", [])
                raw_text = " ".join("".join(span.get("text", "") for span in line.get("spans", [])) for line in lines).strip()
                if raw_text and is_translatable(raw_text):
                    total_translatable_blocks += 1

    if total_translatable_blocks == 0:
        doc.close()
        raise ValueError("此 PDF 文件中未偵測到可翻譯的文字區塊，可能是純圖片或掃描檔案，需先透過 OCR 轉換為文字版 PDF。")

    if progress_callback:
        progress_callback(0, total_pages, "開始解析 PDF 文件...")

    for page_idx in range(total_pages):
        page = doc[page_idx]
        text_dict = page.get_text("dict")
        blocks_to_replace = []
        batch_items = []

        # Step 1: Collect text blocks
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:  # 0 is text, 1 is image
                continue
            
            lines = block.get("lines", [])
            if not lines:
                continue

            # Join lines to form paragraph text
            raw_text = " ".join("".join(span.get("text", "") for span in line.get("spans", [])) for line in lines).strip()
            if not raw_text or not is_translatable(raw_text):
                continue

            # Derive primary font size and color from first valid span
            first_span = lines[0]["spans"][0] if lines[0].get("spans") else {}
            font_size = float(first_span.get("size", 10.0))
            raw_color = first_span.get("color", 0)

            if isinstance(raw_color, int):
                r = (raw_color >> 16 & 0xFF) / 255.0
                g = (raw_color >> 8 & 0xFF) / 255.0
                b = (raw_color & 0xFF) / 255.0
                text_color = (r, g, b)
            elif isinstance(raw_color, (list, tuple)) and len(raw_color) == 3:
                text_color = tuple(raw_color)
            else:
                text_color = (0, 0, 0)

            item_id = len(batch_items)
            batch_items.append({"id": item_id, "text": raw_text})
            blocks_to_replace.append({
                "id": item_id,
                "bbox": block["bbox"],
                "is_single_line": len(lines) == 1,
                "font_size": font_size,
                "color": text_color,
                "original": raw_text
            })

        # Step 2: Translate in batches via LLM
        translations = {}
        if batch_items:
            for i in range(0, len(batch_items), batch_size):
                chunk = batch_items[i:i + batch_size]
                if progress_callback:
                    progress_callback(
                        page_idx + 1,
                        total_pages,
                        f"正在翻譯第 {page_idx + 1}/{total_pages} 頁（區塊 {i + 1}~{min(i + batch_size, len(batch_items))}）..."
                    )
                try:
                    res_map = translate_blocks(chunk, provider, api_key, target_lang_desc, model=model, base_url=base_url)
                    translations.update(res_map)
                except FatalAPIError:
                    doc.close()
                    raise
                except Exception as e:
                    if progress_callback:
                        progress_callback(page_idx + 1, total_pages, f"第 {page_idx + 1} 頁翻譯區塊遇到警告: {str(e)[:60]}")

        # Step 3: Redact original blocks
        for b in blocks_to_replace:
            if b["id"] in translations:
                # fill=False preserves underlying page background
                page.add_redact_annot(pymupdf.Rect(b["bbox"]), fill=False)
        page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE)

        # Step 4: Insert translated text preserving layout
        for b in blocks_to_replace:
            translated = translations.get(b["id"])
            if not translated:
                continue
            
            bbox = b["bbox"]
            rect = pymupdf.Rect(bbox)
            fs = b["font_size"]
            color = b["color"]

            if b["is_single_line"]:
                # Scale font size if translated text is wider than original box
                max_w = max(bbox[2] - bbox[0], 20)
                text_w = pymupdf.get_text_length(translated, fontname=fontname, fontsize=fs)
                if text_w > max_w:
                    fs = max(fs * (max_w / text_w) * 0.95, 6.0)
                # Position baseline 1.5pt above bbox bottom
                baseline_y = bbox[3] - 1.5
                page.insert_text((bbox[0], baseline_y), translated, fontsize=fs, fontname=fontname, color=color)
            else:
                # Fit multi-line block by stepping down font size until it fits
                min_fs = max(fs * 0.6, 6.0)
                cur_fs = fs * 0.95
                while cur_fs >= min_fs:
                    shape = page.new_shape()
                    deficit = shape.insert_textbox(rect, translated, fontsize=cur_fs, fontname=fontname, color=color)
                    if deficit >= 0:
                        shape.commit()
                        break
                    cur_fs -= 0.5
                else:
                    # Commit at minimum font size if still overflowing
                    shape = page.new_shape()
                    shape.insert_textbox(rect, translated, fontsize=min_fs, fontname=fontname, color=color)
                    shape.commit()

            total_blocks_translated += 1

    if total_translatable_blocks > 0 and total_blocks_translated == 0:
        doc.close()
        raise RuntimeError("翻譯失敗：所有文字區塊翻譯請求均未成功，請確認 API Key、網路連線或服務配額。")

    if progress_callback:
        progress_callback(total_pages, total_pages, "正在儲存已翻譯的 PDF 文件...")

    doc.save(output_path, garbage=4, deflate=True)
    doc.close()

    return {
        "total_pages": total_pages,
        "translated_blocks": total_blocks_translated,
        "output_path": output_path
    }

def run_self_check():
    """Verify PDF generation, redaction, and font insertion logic."""
    test_pdf = "self_check_in.pdf"
    test_out = "self_check_out.pdf"
    try:
        # Create a test PDF
        doc = pymupdf.open()
        p = doc.new_page(width=595, height=842)
        p.insert_text((50, 50), "Self Check Title", fontsize=16)
        p.insert_textbox(pymupdf.Rect(50, 80, 400, 200), "This is a paragraph test for the translator self check routine.", fontsize=12)
        doc.save(test_pdf)
        doc.close()

        # Mock translation mapping
        mock_translations = {0: "自我檢查標題", 1: "這是一段用於翻譯器自我檢查常式的測試段落。"}
        doc = pymupdf.open(test_pdf)
        p = doc[0]
        p.add_redact_annot(pymupdf.Rect(50, 40, 300, 60), fill=False)
        p.add_redact_annot(pymupdf.Rect(50, 80, 400, 200), fill=False)
        p.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE)

        p.insert_text((50, 50), mock_translations[0], fontsize=15, fontname="china-ts")
        shape = p.new_shape()
        shape.insert_textbox(pymupdf.Rect(50, 80, 400, 200), mock_translations[1], fontsize=11, fontname="china-ts")
        shape.commit()
        doc.save(test_out)
        doc.close()

        # Read back and verify text was replaced
        doc_out = pymupdf.open(test_out)
        full_text = doc_out[0].get_text()
        doc_out.close()

        assert "自我檢查" in full_text, "Failed to find inserted Chinese text in output PDF"
        assert "Self Check" not in full_text, "Original text was not redacted"
        print("translator.py self-check passed successfully!")
    finally:
        for f in (test_pdf, test_out):
            if os.path.exists(f):
                os.remove(f)

if __name__ == "__main__":
    run_self_check()
