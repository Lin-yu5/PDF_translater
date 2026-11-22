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

FONT_STYLE_MAP = {
    "mingti": "china-ts",       # 明體 / 宋體
    "heiti": "china-ss",        # 黑體
    "kaiu": "china-ts",         # 楷書 (fallback to china-ts if no ttf)
}

# Global lazy OCR engine
_OCR_ENGINE = None

def get_ocr_engine():
    """Lazy initialize RapidOCR engine singleton."""
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _OCR_ENGINE = RapidOCR()
        except ImportError:
            _OCR_ENGINE = False
    return _OCR_ENGINE if _OCR_ENGINE is not False else None

def is_translatable(text: str) -> bool:
    """Check whether a text segment requires translation."""
    clean = text.strip()
    if len(clean) <= 1:
        return False
    # skip pure numbers, dates, version numbers or punctuation
    if re.fullmatch(r"[\d\.\,\-\:\/\s\(\)\[\]\%\$\#\@\+\=\*]+", clean):
        return False
    return True

def clean_ocr_text(text: str) -> str:
    """Clean up OCR recognized text artifacts."""
    t = text.strip()
    # Remove obvious non-text line artifacts
    if re.fullmatch(r"[\-_=\.\|/\\~]{2,}", t):
        return ""
    return t

def safe_json_parse(raw: str):
    """Parse JSON string, stripping markdown code fences or preambles if present."""
    trimmed = raw.strip()
    # 1. Strip markdown code fences if present (```json ... ``` or ``` ... ```)
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", trimmed, re.DOTALL)
    if fence_match:
        trimmed = fence_match.group(1).strip()
    
    # 2. Try direct parse first
    try:
        return json.loads(trimmed)
    except Exception:
        pass
        
    # 3. Find outermost object { ... } or array [ ... ]
    first_brace = trimmed.find("{")
    first_bracket = trimmed.find("[")
    
    if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        last_brace = trimmed.rfind("}")
        if last_brace > first_brace:
            try:
                return json.loads(trimmed[first_brace:last_brace + 1].strip())
            except Exception:
                pass
    elif first_bracket != -1:
        last_bracket = trimmed.rfind("]")
        if last_bracket > first_bracket:
            try:
                return json.loads(trimmed[first_bracket:last_bracket + 1].strip())
            except Exception:
                pass
                
    return json.loads(trimmed)

def extract_translation_map(content_text: str) -> dict:
    """Parse LLM output JSON and normalize IDs to integers."""
    parsed = safe_json_parse(content_text)
    if isinstance(parsed, list):
        raw_items = parsed
    elif isinstance(parsed, dict):
        raw_items = parsed.get("translations", [])
        if isinstance(raw_items, dict):
            raw_items = [{"id": k, "text": v} for k, v in raw_items.items()]
    else:
        raw_items = []

    result = {}
    for item in raw_items:
        try:
            tid = int(item["id"])
            text = str(item.get("text", ""))
            result[tid] = text
        except (ValueError, TypeError, KeyError):
            continue
    return result

_LAST_GEMINI_CALL = 0.0

def rate_limit_pace(min_interval: float = 0.5):
    """Ensure minimum interval between API calls to prevent 429 quota exhaustion while maintaining high throughput."""
    global _LAST_GEMINI_CALL
    now = time.time()
    elapsed = now - _LAST_GEMINI_CALL
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _LAST_GEMINI_CALL = time.time()

def _post_with_retry(url: str, headers: dict = None, json_payload: dict = None, timeout: int = 60, max_retries: int = 4) -> requests.Response:
    """Execute HTTP POST with fast-fail on fatal client errors and extended backoff for 429 quota resets."""
    backoff_delays = [4.0, 15.0, 30.0, 60.0]
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=json_payload, timeout=timeout)
            if resp.status_code in (400, 401, 403, 404):
                err_msg = resp.text[:200]
                raise FatalAPIError(f"API 致命錯誤 (HTTP {resp.status_code}): {err_msg}")
            
            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {resp.status_code}: {resp.text[:120]}"
                if attempt < max_retries - 1:
                    wait_sec = backoff_delays[min(attempt, len(backoff_delays) - 1)]
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        try:
                            wait_sec = max(wait_sec, float(retry_after))
                        except (ValueError, TypeError):
                            pass
                    time.sleep(wait_sec)
                    continue
                if resp.status_code == 429:
                    raise RuntimeError(f"API 服務配額耗盡或請求頻率過高 (HTTP 429): {resp.text[:150]}")
                resp.raise_for_status()

            resp.raise_for_status()
            return resp
        except FatalAPIError:
            raise
        except requests.exceptions.RequestException as e:
            last_err = e
            if attempt < max_retries - 1:
                wait_sec = backoff_delays[min(attempt, len(backoff_delays) - 1)]
                time.sleep(wait_sec)
            else:
                raise e
    raise RuntimeError(f"API 連線重試 {max_retries} 次後失敗: {last_err}")

def load_translation_rules(target_lang_desc: str, domain: str = "general") -> str:
    """
    Load and concatenate translation skill rules based on domain.
    Dynamically stacks:
      1. translation_rules.md (Core Output Schema, Purity & Typography)
      2. academic_rules.md (Citations, Cross-References & Statistical Notations)
      3. skills/translate-pdf/SKILL.md (Layout Preservation & Style Transfer Guidelines)
      4. skills/pdf/SKILL.md (Structural & Formatting PDF Directives)
      5. Domain-specific rules: medical_terms.md (CP, GMFCS, MACS, ROM, CIMT terms)
    """
    skills_dir = os.path.join(os.path.dirname(__file__), "skills")
    rules_parts = []

    # 1. Base translation guidelines
    base_file = os.path.join(skills_dir, "translation_rules.md")
    if os.path.exists(base_file):
        try:
            with open(base_file, "r", encoding="utf-8") as f:
                rules_parts.append(f.read().strip())
        except Exception:
            pass

    # 2. Academic rules (applied to general academic, medical, and engineering)
    acad_file = os.path.join(skills_dir, "academic_rules.md")
    if os.path.exists(acad_file):
        try:
            with open(acad_file, "r", encoding="utf-8") as f:
                rules_parts.append(f.read().strip())
        except Exception:
            pass

    # 3. translate-pdf skill rule extraction
    trans_pdf_skill = os.path.join(skills_dir, "translate-pdf", "SKILL.md")
    if os.path.exists(trans_pdf_skill):
        try:
            with open(trans_pdf_skill, "r", encoding="utf-8") as f:
                c = f.read().strip()
                rules_parts.append(f"## translate-pdf Skill Directives:\n{c}")
        except Exception:
            pass

    # 4. pdf processing skill rule extraction
    pdf_skill = os.path.join(skills_dir, "pdf", "SKILL.md")
    if os.path.exists(pdf_skill):
        try:
            with open(pdf_skill, "r", encoding="utf-8") as f:
                c = f.read().strip()
                # Extract first 50 lines / core guidelines
                lines = c.splitlines()[:60]
                rules_parts.append(f"## pdf Processing Skill Directives:\n" + "\n".join(lines))
        except Exception:
            pass

    # 5. Domain-specific rules (medical, clinical)
    if domain in ("medical", "clinical", "med"):
        med_file = os.path.join(skills_dir, "medical_terms.md")
        if os.path.exists(med_file):
            try:
                with open(med_file, "r", encoding="utf-8") as f:
                    rules_parts.append(f.read().strip())
            except Exception:
                pass

    if not rules_parts:
        rules_parts.append(
            "You are an expert document translation engine embedded in an automated PDF pipeline.\n"
            "Rules:\n"
            "1. STRICT ZERO EXPLANATION: Output ONLY the translated text in JSON format. Never add notes, preambles, or commentary.\n"
            "2. 1:1 ID MAPPING: Retain the exact same 'id' for each item in the translations array.\n"
            "3. PRESERVE ARTIFACTS: Keep citations (e.g. [1]), numbers, acronyms, formulas, trademarks, and code unaltered.\n"
            "4. NATURAL FLUENCY: Professional tone, appropriate punctuation (full-width for CJK)."
        )

    combined_rules = "\n\n---\n\n".join(rules_parts)
    return (
        f"{combined_rules}\n\n"
        f"Target Language: {target_lang_desc}.\n"
        "Required Output Format: Valid JSON with schema: {\"translations\": [{\"id\": <int>, \"text\": \"<translated text>\"}]}."
    )

def call_gemini(items: list, target_lang_desc: str, api_key: str, model: str = "gemini-2.0-flash", domain: str = "general") -> dict:
    """Call Google Gemini API using structured JSON output with embedded skill rules and automatic fallback."""
    user_model = (model or "gemini-2.0-flash").strip()
    # Map non-existent or futuristic model aliases to official Gemini models
    model_aliases = {
        "gemini-3.8-flash": "gemini-2.0-flash",
        "gemini-3.5-flash": "gemini-2.0-flash",
        "gemini-2.5-flash": "gemini-2.0-flash",
        "gemini-2.5-pro": "gemini-1.5-pro",
        "gemini-1.5": "gemini-1.5-flash"
    }
    user_model = model_aliases.get(user_model, user_model)

    fallback_chain = [user_model]
    for alt in ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash", "gemini-1.5-pro"]:
        if alt not in fallback_chain:
            fallback_chain.append(alt)

    system_instruction = load_translation_rules(target_lang_desc, domain=domain)
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
    rate_limit_pace(4.0)

    last_404_err = None
    for target_m in fallback_chain:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_m}:generateContent?key={api_key}"
        try:
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
        except FatalAPIError as fe:
            # If 404 model not found, try the next model in fallback chain
            if "404" in str(fe) and target_m != fallback_chain[-1]:
                last_404_err = str(fe)
                rate_limit_pace(1.5)
                continue
            raise

    if last_404_err:
        raise FatalAPIError(last_404_err)

def call_openai(items: list, target_lang_desc: str, api_key: str, model: str = "gpt-4o-mini", base_url: str = None, domain: str = "general") -> dict:
    """Call OpenAI or OpenAI-compatible API with embedded skill rules."""
    endpoint = f"{base_url.rstrip('/')}/chat/completions" if base_url else "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    system_msg = load_translation_rules(target_lang_desc, domain=domain)
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

def call_claude(items: list, target_lang_desc: str, api_key: str, model: str = "claude-3-5-sonnet-20241022", base_url: str = None, domain: str = "general") -> dict:
    """Call Anthropic Claude native API with embedded skill rules."""
    endpoint = f"{base_url.rstrip('/')}/messages" if base_url else "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }
    system_msg = load_translation_rules(target_lang_desc, domain=domain)
    prompt = (
        "Translate the text fields of this JSON array. "
        "Output strictly valid JSON with format: {\"translations\": [{\"id\": <int>, \"text\": \"<translated text>\"}]}:\n"
        f"{json.dumps(items, ensure_ascii=False)}"
    )
    payload = {
        "model": model,
        "max_tokens": 4096,
        "system": system_msg,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }
    resp = _post_with_retry(endpoint, headers=headers, json_payload=payload, timeout=60)
    data = resp.json()
    try:
        content_text = data["content"][0]["text"]
    except (KeyError, IndexError):
        raise RuntimeError(f"無法解析 Claude API 回應格式: {str(data)[:200]}")
    return extract_translation_map(content_text)

def translate_blocks(items: list, provider: str, api_key: str, target_lang_desc: str, model: str = None, base_url: str = None, domain: str = "general") -> dict:
    """Dispatch translation batch to selected LLM provider."""
    if not items:
        return {}
    p = (provider or "gemini").lower()
    if p == "gemini":
        m = model or "gemini-1.5-pro"
        return call_gemini(items, target_lang_desc, api_key, model=m, domain=domain)
    elif p == "claude":
        m = model or "claude-3-5-sonnet-20241022"
        # Support OpenAI-compatible proxies (e.g. OpenRouter/OneAPI) if user sets a custom base_url without anthropic
        if base_url and ("openai" in base_url or "openrouter" in base_url or ("/v1" in base_url and "anthropic" not in base_url)):
            return call_openai(items, target_lang_desc, api_key, model=m, base_url=base_url, domain=domain)
        return call_claude(items, target_lang_desc, api_key, model=m, base_url=base_url, domain=domain)
    elif p in ("openai", "deepseek", "ollama", "custom"):
        # OpenAI-compatible providers
        default_models = {
            "openai": "gpt-4o-mini",
            "deepseek": "deepseek-chat",
            "ollama": "qwen2.5:7b",
            "custom": "gpt-4o-mini"
        }
        m = model or default_models.get(p, "gpt-4o-mini")
        # Default Base URLs
        default_urls = {
            "openai": "https://api.openai.com/v1",
            "deepseek": "https://api.deepseek.com/v1",
            "ollama": "http://localhost:11434/v1"
        }
        b_url = base_url or default_urls.get(p)
        return call_openai(items, target_lang_desc, api_key, model=m, base_url=b_url, domain=domain)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def analyze_pdf_pages(doc: pymupdf.Document) -> list:
    """
    Analyze all pages in a document and return metadata.
    Returns: [{page_num, width, height, text_blocks, img_blocks, page_type: 'text'|'scan'|'mixed'|'empty'}]
    """
    pages_meta = []
    for idx, page in enumerate(doc):
        td = page.get_text("dict")
        text_count = 0
        img_count = 0
        for block in td.get("blocks", []):
            if block.get("type") == 0:
                lines = block.get("lines", [])
                raw = " ".join("".join(s.get("text", "") for s in l.get("spans", [])) for l in lines).strip()
                if raw and is_translatable(raw):
                    text_count += 1
            elif block.get("type") == 1:
                img_count += 1
        
        if text_count > 0 and img_count > 0:
            ptype = "mixed"
        elif text_count > 0:
            ptype = "text"
        elif img_count > 0:
            ptype = "scan"
        else:
            ptype = "empty"

        pages_meta.append({
            "page_num": idx + 1,
            "width": round(page.rect.width),
            "height": round(page.rect.height),
            "text_blocks": text_count,
            "img_blocks": img_count,
            "page_type": ptype,
            "status": "pending"
        })
    return pages_meta

def ocr_page_image_regions(page: pymupdf.Page, min_confidence: float = 0.5) -> list:
    """
    Run RapidOCR on image regions (or the whole page if pure scan) of a PDF page.
    Returns list of dicts: [{bbox: (x0, y0, x1, y1), text: str, confidence: float}]
    Coordinates are normalized to PDF page coordinate system (points).
    """
    ocr_engine = get_ocr_engine()
    if ocr_engine is None:
        return []

    try:
        dpi = 150
        scale = 72.0 / dpi
        pix = page.get_pixmap(dpi=dpi)
        img_bytes = pix.tobytes("png")
        
        ocr_res = ocr_engine(img_bytes)
        ocr_result = ocr_res[0] if ocr_res and isinstance(ocr_res, (list, tuple)) else None
        if not ocr_result:
            return []
    except Exception as e:
        print(f"RapidOCR warning on page {page.number + 1}: {e}")
        return []

    # Get existing text block bboxes on page to avoid OCR re-recognizing native text
    td = page.get_text("dict")
    existing_text_rects = [pymupdf.Rect(b["bbox"]) for b in td.get("blocks", []) if b.get("type") == 0]

    detected_regions = []
    for item in ocr_result:
        box, text, score_str = item
        try:
            score = float(score_str)
        except (ValueError, TypeError):
            score = 0.0

        if score < min_confidence:
            continue

        clean_t = clean_ocr_text(text)
        if not clean_t or not is_translatable(clean_t):
            continue

        # box is 4 points: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]] in pixmap pixels
        xs = [pt[0] for pt in box]
        ys = [pt[1] for pt in box]
        pdf_x0 = min(xs) * scale
        pdf_y0 = min(ys) * scale
        pdf_x1 = max(xs) * scale
        pdf_y1 = max(ys) * scale
        ocr_rect = pymupdf.Rect(pdf_x0, pdf_y0, pdf_x1, pdf_y1)

        # Skip if this OCR box significantly overlaps with an existing native text block
        overlaps_native = False
        for erec in existing_text_rects:
            intersect = ocr_rect.intersect(erec)
            if intersect.is_valid and intersect.get_area() > 0.5 * ocr_rect.get_area():
                overlaps_native = True
                break

        if overlaps_native:
            continue

        detected_regions.append({
            "bbox": (pdf_x0, pdf_y0, pdf_x1, pdf_y1),
            "text": clean_t,
            "confidence": score
        })

    return detected_regions

def detect_block_alignment(block: dict, page_width) -> int:
    """
    Detect alignment (LEFT=0, CENTER=1, RIGHT=2) from block geometry and internal line spans.
    page_width can be a float/int or a pymupdf.Rect.
    Returns pymupdf.TEXT_ALIGN_LEFT, TEXT_ALIGN_CENTER, or TEXT_ALIGN_RIGHT.
    """
    if hasattr(page_width, "width"):
        page_width = page_width.width
    page_width = float(page_width)
    bbox = block.get("bbox", (0, 0, 0, 0))
    bw = bbox[2] - bbox[0]
    lines = block.get("lines", [])

    # 1. Multi-line internal line alignment
    if len(lines) >= 2:
        l_x0s = [l["bbox"][0] for l in lines]
        l_x1s = [l["bbox"][2] for l in lines]
        l_centers = [(l["bbox"][0] + l["bbox"][2]) / 2.0 for l in lines]

        x0_var = max(l_x0s) - min(l_x0s)
        x1_var = max(l_x1s) - min(l_x1s)
        center_var = max(l_centers) - min(l_centers)

        # If line centers are aligned but lefts vary: CENTER
        if center_var <= 6.0 and x0_var > 8.0:
            return pymupdf.TEXT_ALIGN_CENTER
        # If right edges are aligned but lefts vary: RIGHT
        if x1_var <= 6.0 and x0_var > 8.0:
            return pymupdf.TEXT_ALIGN_RIGHT
        # If left edges are strictly aligned: LEFT
        if x0_var <= 6.0:
            return pymupdf.TEXT_ALIGN_LEFT

    # 2. Block-level alignment relative to page width (e.g. titles or right-aligned metadata)
    if bw < page_width * 0.85:
        left_margin = bbox[0]
        right_margin = page_width - bbox[2]
        block_center = (bbox[0] + bbox[2]) / 2.0
        page_center = page_width / 2.0

        if abs(block_center - page_center) < 20.0 and abs(left_margin - right_margin) < 25.0:
            return pymupdf.TEXT_ALIGN_CENTER

        if right_margin < 60.0 and left_margin > right_margin * 2.0:
            return pymupdf.TEXT_ALIGN_RIGHT

    return pymupdf.TEXT_ALIGN_LEFT

def sample_background_and_text_color(pix: pymupdf.Pixmap, rect: pymupdf.Rect, page_rect: pymupdf.Rect):
    """
    Sample subtle matching background color around an OCR bounding box from the page pixmap.
    Returns: (bg_color_rgb_tuple, text_color_rgb_tuple)
    """
    if not pix or pix.width == 0 or pix.height == 0:
        return (1.0, 1.0, 1.0), (0.1, 0.1, 0.1)
    try:
        scale_x = pix.width / page_rect.width
        scale_y = pix.height / page_rect.height

        px0 = max(0, min(pix.width - 1, int(rect.x0 * scale_x)))
        py0 = max(0, min(pix.height - 1, int(rect.y0 * scale_y)))
        px1 = max(0, min(pix.width - 1, int(rect.x1 * scale_x)))
        py1 = max(0, min(pix.height - 1, int(rect.y1 * scale_y)))

        samples = []
        step_x = max(1, (px1 - px0) // 8)
        step_y = max(1, (py1 - py0) // 8)

        # Sample perimeter pixels
        for x in range(px0, px1 + 1, step_x):
            samples.append(pix.pixel(x, max(0, py0 - 1)))
            samples.append(pix.pixel(x, min(pix.height - 1, py1 + 1)))

        for y in range(py0, py1 + 1, step_y):
            samples.append(pix.pixel(max(0, px0 - 1), y))
            samples.append(pix.pixel(min(pix.width - 1, px1 + 1), y))

        if samples:
            r_vals = [s[0] if isinstance(s, (list, tuple)) else s for s in samples]
            g_vals = [s[1] if isinstance(s, (list, tuple)) and len(s) > 1 else s[0] if isinstance(s, (list, tuple)) else s for s in samples]
            b_vals = [s[2] if isinstance(s, (list, tuple)) and len(s) > 2 else s[0] if isinstance(s, (list, tuple)) else s for s in samples]

            r_avg = sum(r_vals) / (len(samples) * 255.0)
            g_avg = sum(g_vals) / (len(samples) * 255.0)
            b_avg = sum(b_vals) / (len(samples) * 255.0)

            # Clamp RGB
            r_avg = max(0.0, min(1.0, r_avg))
            g_avg = max(0.0, min(1.0, g_avg))
            b_avg = max(0.0, min(1.0, b_avg))

            luminance = 0.299 * r_avg + 0.587 * g_avg + 0.114 * b_avg
            text_col = (0.95, 0.95, 0.95) if luminance < 0.45 else (0.1, 0.1, 0.1)

            return (r_avg, g_avg, b_avg), text_col
    except Exception:
        pass
    return (1.0, 1.0, 1.0), (0.1, 0.1, 0.1)

def load_checkpoint(checkpoint_path: str) -> dict:
    """Load translation checkpoint from disk."""
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return {"native_translations": {}, "ocr_translations": {}, "pairs": []}
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw_native = data.get("native_translations", {})
        native_translations = {int(k): v for k, v in raw_native.items()}
        raw_ocr = data.get("ocr_translations", {})
        ocr_translations = {int(k): v for k, v in raw_ocr.items()}
        data["native_translations"] = native_translations
        data["ocr_translations"] = ocr_translations
        return data
    except Exception as e:
        print(f"Warning: Failed to load checkpoint {checkpoint_path}: {e}")
        return {"native_translations": {}, "ocr_translations": {}, "pairs": []}

def save_checkpoint(checkpoint_path: str, data: dict):
    """Atomically save checkpoint to disk."""
    if not checkpoint_path:
        return
    try:
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
        to_save = dict(data)
        to_save["updated_at"] = time.time()
        if "native_translations" in to_save:
            to_save["native_translations"] = {str(k): v for k, v in to_save["native_translations"].items()}
        if "ocr_translations" in to_save:
            to_save["ocr_translations"] = {str(k): v for k, v in to_save["ocr_translations"].items()}

        tmp_path = f"{checkpoint_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(to_save, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, checkpoint_path)
    except Exception as e:
        print(f"Warning: Failed to save checkpoint to {checkpoint_path}: {e}")

def export_bilingual_pdf(input_pdf_path: str, translated_pdf_path: str, output_path: str) -> str:
    """
    Create a side-by-side 2-up bilingual comparison PDF.
    Left spread: original page, Right spread: translated page.
    """
    if not os.path.exists(input_pdf_path):
        raise FileNotFoundError(f"Original PDF not found: {input_pdf_path}")
    if not os.path.exists(translated_pdf_path):
        raise FileNotFoundError(f"Translated PDF not found: {translated_pdf_path}")

    orig_doc = pymupdf.open(input_pdf_path)
    trans_doc = pymupdf.open(translated_pdf_path)
    bilingual_doc = pymupdf.open()

    total = min(len(orig_doc), len(trans_doc))
    for i in range(total):
        p_orig = orig_doc[i]
        p_trans = trans_doc[i]
        w1, h1 = p_orig.rect.width, p_orig.rect.height
        w2, h2 = p_trans.rect.width, p_trans.rect.height

        # Side-by-side 2-up spread page
        new_page = bilingual_doc.new_page(width=w1 + w2, height=max(h1, h2))
        new_page.show_pdf_page(pymupdf.Rect(0, 0, w1, h1), orig_doc, i)
        new_page.show_pdf_page(pymupdf.Rect(w1, 0, w1 + w2, h2), trans_doc, i)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    bilingual_doc.save(output_path, garbage=4, deflate=True)
    bilingual_doc.close()
    orig_doc.close()
    trans_doc.close()
    return output_path

def export_docx_comparison(title: str, pairs, output_path: str) -> str:
    """
    Generate a bilingual comparison Word document (.docx) using pure standard library zipfile.
    Supports either:
    1. (title: str, pairs: list, output_path: str) where pairs = [{"page": int, "source": str, "target": str}]
    2. (orig_pdf_path: str, trans_pdf_path: str, output_path: str)
    """
    import zipfile
    import xml.sax.saxutils as saxutils

    # If first two arguments are file paths, dynamically extract page text pairs
    if isinstance(title, str) and isinstance(pairs, str) and os.path.isfile(title) and os.path.isfile(pairs):
        orig_path = title
        trans_path = pairs
        title = os.path.splitext(os.path.basename(orig_path))[0] + " 雙語對照文檔"
        try:
            o_doc = pymupdf.open(orig_path)
            t_doc = pymupdf.open(trans_path)
            extracted_pairs = []
            for page_idx in range(min(len(o_doc), len(t_doc))):
                s_txt = o_doc[page_idx].get_text().strip()
                t_txt = t_doc[page_idx].get_text().strip()
                if s_txt or t_txt:
                    extracted_pairs.append({
                        "page": page_idx + 1,
                        "source": s_txt,
                        "target": t_txt
                    })
            o_doc.close()
            t_doc.close()
            pairs = extracted_pairs
        except Exception as e:
            print(f"Warning: could not extract pairs from PDFs for docx: {e}")
            pairs = []
    elif not isinstance(pairs, list):
        pairs = []

    def escape(s):
        clean = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", str(s or ""))
        return saxutils.escape(clean)

    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""

    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

    table_rows = []
    # Header row
    table_rows.append("""
      <w:tr>
        <w:tc>
          <w:tcPr><w:tcW w:w="1200" w:type="dxa"/><w:shd w:val="clear" w:color="auto" w:fill="2563EB"/></w:tcPr>
          <w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:rPr><w:b/><w:color w:val="FFFFFF"/></w:rPr><w:t>頁碼</w:t></w:r></w:p>
        </w:tc>
        <w:tc>
          <w:tcPr><w:tcW w:w="4400" w:type="dxa"/><w:shd w:val="clear" w:color="auto" w:fill="2563EB"/></w:tcPr>
          <w:p><w:r><w:rPr><w:b/><w:color w:val="FFFFFF"/></w:rPr><w:t>原文 (Source Text)</w:t></w:r></w:p>
        </w:tc>
        <w:tc>
          <w:tcPr><w:tcW w:w="4400" w:type="dxa"/><w:shd w:val="clear" w:color="auto" w:fill="2563EB"/></w:tcPr>
          <w:p><w:r><w:rPr><w:b/><w:color w:val="FFFFFF"/></w:rPr><w:t>譯文 (Translated Text)</w:t></w:r></w:p>
        </w:tc>
      </w:tr>""")

    for p in pairs:
        page_val = escape(p.get("page", ""))
        src_val = escape(p.get("source", "")).replace("\n", "<w:br/>")
        tgt_val = escape(p.get("target", "")).replace("\n", "<w:br/>")
        table_rows.append(f"""
      <w:tr>
        <w:tc>
          <w:tcPr><w:tcW w:w="1200" w:type="dxa"/><w:vAlign w:val="center"/></w:tcPr>
          <w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:rPr><w:color w:val="666666"/></w:rPr><w:t>第 {page_val} 頁</w:t></w:r></w:p>
        </w:tc>
        <w:tc>
          <w:tcPr><w:tcW w:w="4400" w:type="dxa"/></w:tcPr>
          <w:p><w:r><w:t>{src_val}</w:t></w:r></w:p>
        </w:tc>
        <w:tc>
          <w:tcPr><w:tcW w:w="4400" w:type="dxa"/></w:tcPr>
          <w:p><w:r><w:rPr><w:b/></w:rPr><w:t>{tgt_val}</w:t></w:r></w:p>
        </w:tc>
      </w:tr>""")

    body_xml = f"""
    <w:p>
      <w:pPr><w:jc w:val="center"/></w:pPr>
      <w:r><w:rPr><w:b/><w:sz w:val="36"/><w:color w:val="1E3A8A"/></w:rPr><w:t>{escape(title)}</w:t></w:r>
    </w:p>
    <w:p>
      <w:pPr><w:jc w:val="center"/></w:pPr>
      <w:r><w:rPr><w:color w:val="64748B"/><w:sz w:val="20"/></w:rPr><w:t>文脈 FolioTrans - 雙語對照匯出文檔 (Bilingual Alignment Export)</w:t></w:r>
    </w:p>
    <w:p><w:r><w:t></w:t></w:r></w:p>
    <w:tbl>
      <w:tblPr>
        <w:tblW w:w="10000" w:type="dxa"/>
        <w:tblBorders>
          <w:top w:val="single" w:sz="6" w:space="0" w:color="CBD5E1"/>
          <w:left w:val="single" w:sz="6" w:space="0" w:color="CBD5E1"/>
          <w:bottom w:val="single" w:sz="6" w:space="0" w:color="CBD5E1"/>
          <w:right w:val="single" w:sz="6" w:space="0" w:color="CBD5E1"/>
          <w:insideH w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>
          <w:insideV w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>
        </w:tblBorders>
      </w:tblPr>
      {"".join(table_rows)}
    </w:tbl>
"""

    doc_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {body_xml}
  </w:body>
</w:document>"""

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", doc_xml)
    return output_path

def translate_pdf_file(
    input_path: str,
    output_path: str,
    provider: str,
    api_key: str,
    target_lang_code: str = "zh-TW",
    model: str = None,
    base_url: str = None,
    domain: str = "general",
    font_style: str = "mingti",
    enable_ocr: bool = True,
    batch_size: int = 40,
    progress_callback = None,
    checkpoint_path: str = None
) -> dict:
    """
    Translates a PDF file with two-phase workflow:
      Phase 1: Native selectable text translation & redaction
      Phase 2: OCR scanning on image regions & overlay translation
    """
    lang_info = LANGUAGE_CONFIG.get(target_lang_code, LANGUAGE_CONFIG["zh-TW"])
    # Determine font
    if font_style in FONT_STYLE_MAP:
        fontname = FONT_STYLE_MAP[font_style]
    else:
        fontname = lang_info["font"]
    target_lang_desc = lang_info["desc"]

    doc = pymupdf.open(input_path)
    if doc.is_encrypted:
        doc.close()
        raise ValueError("此 PDF 文件受密碼保護，無法進行翻譯處理。")

    total_pages = len(doc)
    pages_meta = analyze_pdf_pages(doc)
    total_native_blocks = sum(p["text_blocks"] for p in pages_meta)
    total_img_pages = sum(1 for p in pages_meta if p["img_blocks"] > 0)

    # Check if completely empty
    if total_native_blocks == 0 and not enable_ocr:
        doc.close()
        raise ValueError("此 PDF 未偵測到文字層。請開啟「混合型兩階段翻譯」以啟用 OCR 掃描辨識。")

    # Load checkpoint if available to support resume
    checkpoint_data = load_checkpoint(checkpoint_path)
    native_translations = checkpoint_data.get("native_translations", {})
    ocr_translations = checkpoint_data.get("ocr_translations", {})

    # Ensure checkpoint file exists on disk immediately
    if checkpoint_path and not os.path.exists(checkpoint_path):
        save_checkpoint(checkpoint_path, {
            "status": "in_progress",
            "phase": 1,
            "native_translations": native_translations,
            "ocr_translations": ocr_translations,
            "total_pages": total_pages
        })

    total_blocks_translated = 0
    total_ocr_translated = 0

    # -------------------------------------------------------------
    # PHASE 1: NATIVE TEXT LAYER TRANSLATION (CROSS-PAGE BATCHING)
    # -------------------------------------------------------------
    if progress_callback:
        progress_callback(0, total_pages, 1, "第一階段：正在解析全文件文字層與段落排版...")

    all_native_blocks = []
    # Collect all blocks across all pages with geometry & alignment
    for page_idx in range(total_pages):
        page = doc[page_idx]
        text_dict = page.get_text("dict")
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            lines = block.get("lines", [])
            if not lines:
                continue

            raw_text = " ".join("".join(s.get("text", "") for s in l.get("spans", [])) for l in lines).strip()
            if not raw_text or not is_translatable(raw_text):
                continue

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

            align = detect_block_alignment(block, page.rect.width)

            all_native_blocks.append({
                "global_id": len(all_native_blocks),
                "page_idx": page_idx,
                "bbox": block["bbox"],
                "is_single_line": len(lines) == 1,
                "font_size": font_size,
                "color": text_color,
                "align": align,
                "text": raw_text
            })

    total_native_count = len(all_native_blocks)
    if total_native_count > 0:
        total_batches = (total_native_count + batch_size - 1) // batch_size
        consecutive_failures = 0
        last_batch_err = None
        for b_idx in range(0, total_native_count, batch_size):
            chunk = all_native_blocks[b_idx:b_idx + batch_size]
            batch_num = (b_idx // batch_size) + 1
            chunk_items = [{"id": b["global_id"], "text": b["text"]} for b in chunk]
            start_p = chunk[0]["page_idx"] + 1
            end_p = chunk[-1]["page_idx"] + 1

            # Checkpoint: filter out items already translated
            needed_items = [it for it in chunk_items if it["id"] not in native_translations]

            if needed_items:
                if progress_callback:
                    progress_callback(
                        end_p,
                        total_pages,
                        1,
                        f"第一階段 (文字層)：正在翻譯批次 {batch_num}/{total_batches}（第 {start_p}~{end_p} 頁，共 {len(needed_items)} 個新區塊）..."
                    )
                try:
                    res_map = translate_blocks(needed_items, provider, api_key, target_lang_desc, model=model, base_url=base_url, domain=domain)
                    native_translations.update(res_map)
                    consecutive_failures = 0
                    # Persist intermediate checkpoint immediately
                    save_checkpoint(checkpoint_path, {
                        "status": "in_progress",
                        "phase": 1,
                        "native_translations": native_translations,
                        "ocr_translations": ocr_translations,
                        "total_pages": total_pages
                    })
                except FatalAPIError:
                    doc.close()
                    raise
                except Exception as e:
                    consecutive_failures += 1
                    last_batch_err = str(e)
                    if progress_callback:
                        progress_callback(end_p, total_pages, 1, f"批次 {batch_num} 翻譯警告: {str(e)[:60]}")
                    if consecutive_failures >= 2 and len(native_translations) == 0:
                        doc.close()
                        raise RuntimeError(f"翻譯失敗：連續批次請求失敗（{last_batch_err[:120]}）。請確認 API Key、網路連線或服務配額。")
            else:
                if progress_callback:
                    progress_callback(
                        end_p,
                        total_pages,
                        1,
                        f"第一階段 (文字層)：批次 {batch_num}/{total_batches}（已載入快取，跳過 API 請求）..."
                    )

        # Group blocks by page and apply redaction & insertion
        page_to_blocks = {}
        for b in all_native_blocks:
            page_to_blocks.setdefault(b["page_idx"], []).append(b)

        pad = 1.5  # 1.5pt padding to completely eliminate descending letter artifacts (g, p, y, q, j)
        for p_idx in range(total_pages):
            page_blocks = page_to_blocks.get(p_idx, [])
            if not page_blocks:
                continue
            page = doc[p_idx]
            for b in page_blocks:
                if b["global_id"] in native_translations:
                    bx0, by0, bx1, by1 = b["bbox"]
                    padded_rect = pymupdf.Rect(
                        max(0.0, bx0 - pad),
                        max(0.0, by0 - pad),
                        min(page.rect.width, bx1 + pad),
                        min(page.rect.height, by1 + pad)
                    )
                    page.add_redact_annot(padded_rect, fill=False)
            page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE)

            for b in page_blocks:
                translated = native_translations.get(b["global_id"])
                if not translated:
                    continue
                bbox = b["bbox"]
                rect = pymupdf.Rect(bbox)
                fs = b["font_size"]
                color = b["color"]
                align = b.get("align", pymupdf.TEXT_ALIGN_LEFT)
                max_w = max(bbox[2] - bbox[0], 20.0)

                if b["is_single_line"]:
                    text_w = pymupdf.get_text_length(translated, fontname=fontname, fontsize=fs)
                    if text_w <= max_w:
                        if align == pymupdf.TEXT_ALIGN_CENTER:
                            insert_x = max(bbox[0], (bbox[0] + bbox[2] - text_w) / 2.0)
                        elif align == pymupdf.TEXT_ALIGN_RIGHT:
                            insert_x = max(bbox[0], bbox[2] - text_w)
                        else:
                            insert_x = bbox[0]
                        baseline_y = bbox[3] - 1.5
                        page.insert_text((insert_x, baseline_y), translated, fontsize=fs, fontname=fontname, color=color)
                    else:
                        scaled_fs = fs * (max_w / text_w) * 0.96
                        if scaled_fs >= max(fs * 0.65, 6.0):
                            scaled_w = pymupdf.get_text_length(translated, fontname=fontname, fontsize=scaled_fs)
                            if align == pymupdf.TEXT_ALIGN_CENTER:
                                insert_x = max(bbox[0], (bbox[0] + bbox[2] - scaled_w) / 2.0)
                            elif align == pymupdf.TEXT_ALIGN_RIGHT:
                                insert_x = max(bbox[0], bbox[2] - scaled_w)
                            else:
                                insert_x = bbox[0]
                            baseline_y = bbox[3] - 1.5
                            page.insert_text((insert_x, baseline_y), translated, fontsize=scaled_fs, fontname=fontname, color=color)
                        else:
                            # Auto-fit multi-line textbox with zero truncation
                            fitted = False
                            cur_fs = fs * 0.95
                            min_fs = max(fs * 0.55, 4.5)
                            while cur_fs >= min_fs:
                                shape = page.new_shape()
                                deficit = shape.insert_textbox(rect, translated, fontsize=cur_fs, fontname=fontname, color=color, align=align)
                                if deficit >= 0:
                                    shape.commit()
                                    fitted = True
                                    break
                                cur_fs -= 0.5
                            if not fitted:
                                exp_bottom = min(page.rect.height - 10.0, rect.y1 + 40.0)
                                exp_rect = pymupdf.Rect(rect.x0, rect.y0, rect.x1, exp_bottom)
                                shape = page.new_shape()
                                shape.insert_textbox(exp_rect, translated, fontsize=min_fs, fontname=fontname, color=color, align=align)
                                shape.commit()
                else:
                    # Multi-line paragraphs: dynamic font scale & vertical headroom expansion
                    fitted = False
                    cur_fs = fs * 0.95
                    min_fs = max(fs * 0.55, 4.5)
                    while cur_fs >= min_fs:
                        shape = page.new_shape()
                        deficit = shape.insert_textbox(rect, translated, fontsize=cur_fs, fontname=fontname, color=color, align=align)
                        if deficit >= 0:
                            shape.commit()
                            fitted = True
                            break
                        cur_fs -= 0.5
                    if not fitted:
                        # Expand vertically downwards towards margin to guarantee zero silent truncation
                        exp_bottom = min(page.rect.height - 12.0, rect.y1 + 60.0)
                        exp_rect = pymupdf.Rect(rect.x0, rect.y0, rect.x1, exp_bottom)
                        shape = page.new_shape()
                        deficit = shape.insert_textbox(exp_rect, translated, fontsize=min_fs, fontname=fontname, color=color, align=align)
                        if deficit < 0:
                            shape = page.new_shape()
                            shape.insert_textbox(exp_rect, translated, fontsize=4.0, fontname=fontname, color=color, align=align)
                        shape.commit()

                total_blocks_translated += 1

        # Intermediate save: write Phase 1 translated PDF to output_path
        # Ensures UI canvas preview shows translated pages immediately even while Phase 2 runs
        if total_blocks_translated > 0:
            try:
                doc.save(output_path, garbage=3, deflate=True)
            except Exception:
                pass

    # -------------------------------------------------------------
    # PHASE 2: OCR SCANNING & OVERLAY ON IMAGE REGIONS (CROSS-PAGE BATCHING)
    # -------------------------------------------------------------
    ocr_available = get_ocr_engine() is not None
    if enable_ocr and ocr_available and total_img_pages > 0:
        if progress_callback:
            progress_callback(0, total_pages, 2, "第二階段：啟動 RapidOCR 掃描全文件圖表與圖片...")

        all_ocr_regions = []
        for page_idx in range(total_pages):
            meta = pages_meta[page_idx]
            if meta["img_blocks"] == 0:
                continue

            page = doc[page_idx]
            if progress_callback:
                progress_callback(
                    page_idx + 1,
                    total_pages,
                    2,
                    f"第二階段 (OCR 掃描)：辨識第 {page_idx + 1}/{total_pages} 頁之圖表文字..."
                )

            ocr_regions = ocr_page_image_regions(page, min_confidence=0.55)
            for reg in ocr_regions:
                all_ocr_regions.append({
                    "global_id": len(all_ocr_regions),
                    "page_idx": page_idx,
                    "bbox": reg["bbox"],
                    "text": reg["text"]
                })

        total_ocr_count = len(all_ocr_regions)
        if total_ocr_count > 0:
            total_ocr_batches = (total_ocr_count + batch_size - 1) // batch_size
            for b_idx in range(0, total_ocr_count, batch_size):
                chunk = all_ocr_regions[b_idx:b_idx + batch_size]
                batch_num = (b_idx // batch_size) + 1
                chunk_items = [{"id": r["global_id"], "text": r["text"]} for r in chunk]
                start_p = chunk[0]["page_idx"] + 1
                end_p = chunk[-1]["page_idx"] + 1

                needed_ocr = [it for it in chunk_items if it["id"] not in ocr_translations]

                if needed_ocr:
                    if progress_callback:
                        progress_callback(
                            end_p,
                            total_pages,
                            2,
                            f"第二階段 (OCR 翻譯)：正在翻譯批次 {batch_num}/{total_ocr_batches}（第 {start_p}~{end_p} 頁，共 {len(needed_ocr)} 處新標籤）..."
                        )
                    try:
                        res_map = translate_blocks(needed_ocr, provider, api_key, target_lang_desc, model=model, base_url=base_url, domain=domain)
                        ocr_translations.update(res_map)
                        # Persist intermediate checkpoint
                        save_checkpoint(checkpoint_path, {
                            "status": "in_progress",
                            "phase": 2,
                            "native_translations": native_translations,
                            "ocr_translations": ocr_translations,
                            "total_pages": total_pages
                        })
                    except FatalAPIError:
                        doc.close()
                        raise
                    except Exception as e:
                        if progress_callback:
                            progress_callback(end_p, total_pages, 2, f"OCR 批次 {batch_num} 翻譯警告: {str(e)[:60]}")
                else:
                    if progress_callback:
                        progress_callback(
                            end_p,
                            total_pages,
                            2,
                            f"第二階段 (OCR 翻譯)：批次 {batch_num}/{total_ocr_batches}（已載入快取，跳過 API 請求）..."
                        )

            # Group OCR regions by page and overlay with subtle matching background
            page_to_ocr = {}
            for r in all_ocr_regions:
                page_to_ocr.setdefault(r["page_idx"], []).append(r)

            for p_idx, regions in page_to_ocr.items():
                page = doc[p_idx]
                page_pix = page.get_pixmap(dpi=150)
                for reg in regions:
                    trans_text = ocr_translations.get(reg["global_id"])
                    if not trans_text:
                        continue
                    bbox = reg["bbox"]
                    rect = pymupdf.Rect(bbox)

                    # Sample subtle matching background color from surrounding pixels
                    bg_col, text_col = sample_background_and_text_color(page_pix, rect, page.rect)

                    pad_ocr = 1.5
                    padded_rect = pymupdf.Rect(
                        max(0.0, rect.x0 - pad_ocr),
                        max(0.0, rect.y0 - pad_ocr),
                        min(page.rect.width, rect.x1 + pad_ocr),
                        min(page.rect.height, rect.y1 + pad_ocr)
                    )
                    shape = page.new_shape()
                    shape.draw_rect(padded_rect)
                    shape.finish(color=bg_col, fill=bg_col, fill_opacity=0.94)
                    shape.commit()

                    fs = max(min((rect.y1 - rect.y0) * 0.75, 14.0), 6.0)
                    shape_text = page.new_shape()
                    deficit = shape_text.insert_textbox(rect, trans_text, fontsize=fs, fontname=fontname, color=text_col)
                    if deficit < 0:
                        fs = max(fs * 0.8, 5.0)
                        shape_text = page.new_shape()
                        deficit = shape_text.insert_textbox(rect, trans_text, fontsize=fs, fontname=fontname, color=text_col)
                    shape_text.commit()
                    total_ocr_translated += 1

    if (total_native_blocks > 0 or total_img_pages > 0) and (total_blocks_translated + total_ocr_translated == 0):
        doc.close()
        raise RuntimeError("翻譯失敗：所有文字區塊翻譯請求均未成功，請確認 API Key、網路連線或服務配額。")

    # Build translation pairs for docx/bilingual export
    translation_pairs = []
    for b in all_native_blocks:
        t = native_translations.get(b["global_id"])
        if t:
            translation_pairs.append({
                "page": b["page_idx"] + 1,
                "source": b["text"],
                "target": t
            })
    if enable_ocr and ocr_available:
        for r in all_ocr_regions if 'all_ocr_regions' in locals() else []:
            t = ocr_translations.get(r["global_id"])
            if t:
                translation_pairs.append({
                    "page": r["page_idx"] + 1,
                    "source": r["text"],
                    "target": t
                })
    translation_pairs.sort(key=lambda x: x["page"])

    # Persist final completed checkpoint
    save_checkpoint(checkpoint_path, {
        "status": "completed",
        "total_pages": total_pages,
        "translated_blocks": total_blocks_translated,
        "ocr_blocks_translated": total_ocr_translated,
        "total_translated": total_blocks_translated + total_ocr_translated,
        "native_translations": native_translations,
        "ocr_translations": ocr_translations,
        "pairs": translation_pairs,
        "output_path": output_path
    })

    if progress_callback:
        progress_callback(total_pages, total_pages, 2, "正在儲存已翻譯的完整 PDF 文件...")

    doc.save(output_path, garbage=4, deflate=True)
    doc.close()

    return {
        "total_pages": total_pages,
        "translated_blocks": total_blocks_translated,
        "ocr_blocks_translated": total_ocr_translated,
        "total_translated": total_blocks_translated + total_ocr_translated,
        "pairs": translation_pairs,
        "output_path": output_path
    }

def run_self_check():
    """Verify PDF generation, redaction, and font insertion logic."""
    test_pdf = "self_check_in.pdf"
    test_out = "self_check_out.pdf"
    try:
        doc = pymupdf.open()
        p = doc.new_page(width=595, height=842)
        p.insert_text((50, 50), "Self Check Title", fontsize=16)
        p.insert_textbox(pymupdf.Rect(50, 80, 400, 200), "This is a paragraph test for the translator self check routine.", fontsize=12)
        doc.save(test_pdf)
        doc.close()

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

        doc_out = pymupdf.open(test_out)
        full_text = doc_out[0].get_text()
        doc_out.close()

        assert "自我檢查" in full_text, "Failed to find inserted Chinese text in output PDF"
        assert "Self Check" not in full_text, "Original text was not redacted"
        
        # Test OCR engine presence
        ocr = get_ocr_engine()
        ocr_status = "Available (RapidOCR)" if ocr else "Not installed"
        print(f"translator.py self-check passed successfully! [OCR Engine: {ocr_status}]")
    finally:
        for f in (test_pdf, test_out):
            if os.path.exists(f):
                os.remove(f)

if __name__ == "__main__":
    run_self_check()
