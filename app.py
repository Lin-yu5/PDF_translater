import os
import time
import uuid
import threading
import webbrowser
import argparse
import importlib
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Response
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import pymupdf

import translator
from translator import (
    translate_pdf_file,
    analyze_pdf_pages,
    run_self_check,
    LANGUAGE_CONFIG,
    get_ocr_engine,
    export_bilingual_pdf,
    export_docx_comparison,
    load_checkpoint,
    save_checkpoint
)

app = FastAPI(title="文脈 FolioTrans - AI PDF 雙階段排版翻譯器")

TASKS_DIR = os.path.join(os.path.dirname(__file__), "temp_tasks")
os.makedirs(TASKS_DIR, exist_ok=True)

def cleanup_stale_tasks(max_age_hours: int = 24):
    """Remove temporary task files older than max_age_hours."""
    try:
        now = time.time()
        cutoff = now - (max_age_hours * 3600)
        if os.path.exists(TASKS_DIR):
            for fname in os.listdir(TASKS_DIR):
                fpath = os.path.join(TASKS_DIR, fname)
                if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
                    try:
                        os.remove(fpath)
                    except Exception:
                        pass
                elif os.path.isdir(fpath) and os.path.getmtime(fpath) < cutoff:
                    import shutil
                    try:
                        shutil.rmtree(fpath)
                    except Exception:
                        pass
    except Exception:
        pass

# In-memory task state tracking
tasks = {}

def background_translate(
    task_id: str,
    input_path: str,
    output_path: str,
    provider: str,
    api_key: str,
    target_lang: str,
    model: Optional[str],
    base_url: Optional[str],
    domain: str,
    font_style: str,
    enable_ocr: bool,
    checkpoint_path: Optional[str] = None
):
    """Worker thread running the translation pipeline with disk checkpointing."""
    def on_progress(cur_page: int, total_pages: int, phase: int, message: str):
        if task_id in tasks and tasks[task_id].get("status") == "stopped":
            raise InterruptedError("使用者已中止此翻譯任務")
        pct = 5 if total_pages == 0 else int((cur_page / total_pages) * 90) + 5
        if task_id in tasks:
            tasks[task_id].update({
                "progress": min(pct, 98),
                "current_phase": phase,
                "message": message
            })

    try:
        if task_id in tasks:
            tasks[task_id]["message"] = "開始解析 PDF 文件結構與各頁型態..."
        res = translate_pdf_file(
            input_path=input_path,
            output_path=output_path,
            provider=provider,
            api_key=api_key,
            target_lang_code=target_lang,
            model=model if model else None,
            base_url=base_url if base_url else None,
            domain=domain,
            font_style=font_style,
            enable_ocr=enable_ocr,
            progress_callback=on_progress,
            checkpoint_path=checkpoint_path
        )

        # Mirror to legacy_out if defined
        legacy_out = tasks.get(task_id, {}).get("legacy_out")
        if legacy_out and os.path.exists(output_path):
            try:
                import shutil
                shutil.copyfile(output_path, legacy_out)
            except Exception:
                pass

        if task_id in tasks:
            tasks[task_id].update({
                "status": "completed",
                "progress": 100,
                "current_phase": 2 if enable_ocr else 1,
                "message": "翻譯全部完成！",
                "total_pages": res["total_pages"],
                "translated_blocks": res["translated_blocks"],
                "ocr_blocks_translated": res.get("ocr_blocks_translated", 0),
                "total_translated": res.get("total_translated", res["translated_blocks"]),
                "pairs": res.get("pairs", []),
                "output_path": output_path
            })
    except Exception as e:
        if task_id in tasks:
            tasks[task_id].update({
                "status": "failed",
                "error": str(e),
                "message": f"失敗: {str(e)}"
            })

@app.get("/")
def serve_ui():
    """Serve the single-page HTML UI."""
    html_file = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(html_file):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(html_file)

@app.post("/api/translate")
async def start_translation_task(
    file: UploadFile = File(...),
    provider: str = Form(...),
    api_key: str = Form(""),
    target_lang: str = Form("zh-TW"),
    model: Optional[str] = Form(None),
    base_url: Optional[str] = Form(None),
    domain: str = Form("general"),
    font_style: str = Form("mingti"),
    enable_ocr: bool = Form(True)
):
    """Upload PDF and trigger background translation with disk checkpointing."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="請上傳 .pdf 格式的檔案")

    cleanup_stale_tasks()

    task_id = uuid.uuid4().hex
    task_dir = os.path.join(TASKS_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)

    input_path = os.path.join(task_dir, "input.pdf")
    output_path = os.path.join(task_dir, "output.pdf")
    checkpoint_path = os.path.join(task_dir, "checkpoint.json")

    # Legacy paths for backward compatibility
    legacy_in = os.path.join(TASKS_DIR, f"{task_id}_in.pdf")
    legacy_out = os.path.join(TASKS_DIR, f"{task_id}_out.pdf")

    contents = await file.read()
    with open(input_path, "wb") as f:
        f.write(contents)
    with open(legacy_in, "wb") as f:
        f.write(contents)

    # Pre-analyze pages metadata
    doc = pymupdf.open(input_path)
    pages_meta = analyze_pdf_pages(doc)
    doc.close()

    tasks[task_id] = {
        "status": "processing",
        "progress": 5,
        "current_phase": 1,
        "message": f"文件上傳成功（共 {len(pages_meta)} 頁），排入處理佇列...",
        "input_filename": file.filename,
        "task_dir": task_dir,
        "input_path": input_path,
        "output_path": output_path,
        "checkpoint_path": checkpoint_path,
        "legacy_out": legacy_out,
        "total_pages": len(pages_meta),
        "pages_meta": pages_meta,
        "domain": domain,
        "enable_ocr": enable_ocr,
        "provider": provider,
        "api_key": api_key,
        "target_lang": target_lang,
        "model": model,
        "base_url": base_url,
        "font_style": font_style,
        "pairs": [],
        "error": None
    }

    thread = threading.Thread(
        target=background_translate,
        args=(task_id, input_path, output_path, provider, api_key, target_lang, model, base_url, domain, font_style, enable_ocr, checkpoint_path),
        daemon=True
    )
    thread.start()

    return {"task_id": task_id, "total_pages": len(pages_meta)}

@app.post("/api/resume/{task_id}")
def resume_translation_task(
    task_id: str,
    api_key: Optional[str] = Form(None),
    provider: Optional[str] = Form(None)
):
    """Resume an interrupted or failed translation task from its disk checkpoint."""
    task = tasks.get(task_id)
    task_dir = os.path.join(TASKS_DIR, task_id)
    checkpoint_path = os.path.join(task_dir, "checkpoint.json")
    input_path = os.path.join(task_dir, "input.pdf")
    output_path = os.path.join(task_dir, "output.pdf")

    if not os.path.exists(input_path):
        legacy_in = os.path.join(TASKS_DIR, f"{task_id}_in.pdf")
        if os.path.exists(legacy_in):
            input_path = legacy_in
            output_path = os.path.join(TASKS_DIR, f"{task_id}_out.pdf")
            checkpoint_path = os.path.join(TASKS_DIR, f"{task_id}_checkpoint.json")
        else:
            raise HTTPException(status_code=404, detail="找不到原任務輸入檔案，無法續傳")

    if not task:
        doc = pymupdf.open(input_path)
        pages_meta = analyze_pdf_pages(doc)
        doc.close()
        tasks[task_id] = {
            "status": "processing",
            "progress": 10,
            "current_phase": 1,
            "message": "正在從磁碟檢查點恢復任務...",
            "input_filename": f"{task_id}.pdf",
            "task_dir": task_dir,
            "input_path": input_path,
            "output_path": output_path,
            "checkpoint_path": checkpoint_path,
            "total_pages": len(pages_meta),
            "pages_meta": pages_meta,
            "domain": "general",
            "enable_ocr": True,
            "provider": provider or "gemini",
            "api_key": api_key or "",
            "target_lang": "zh-TW",
            "model": None,
            "base_url": None,
            "font_style": "mingti",
            "pairs": [],
            "error": None
        }
        task = tasks[task_id]
    else:
        task["status"] = "processing"
        task["error"] = None
        task["message"] = "正在從磁碟檢查點恢復任務..."
        if api_key:
            task["api_key"] = api_key
        if provider:
            task["provider"] = provider

    thread = threading.Thread(
        target=background_translate,
        args=(
            task_id,
            input_path,
            output_path,
            task.get("provider", "gemini"),
            task.get("api_key", ""),
            task.get("target_lang", "zh-TW"),
            task.get("model"),
            task.get("base_url"),
            task.get("domain", "general"),
            task.get("font_style", "mingti"),
            task.get("enable_ocr", True),
            checkpoint_path
        ),
        daemon=True
    )
    thread.start()

    return {"status": "resumed", "task_id": task_id, "message": "任務已從檢查點成功重啟"}

@app.post("/api/stop/{task_id}")
def stop_task(task_id: str):
    """Mark a running task as stopped/cancelled."""
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="找不到此任務 ID")
    task["status"] = "stopped"
    task["message"] = "任務已由使用者手動中止"
    return {"status": "stopped", "task_id": task_id, "message": "已中止任務"}

@app.get("/api/status/{task_id}")
def get_task_status(task_id: str):
    """Query translation status for a task."""
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="找不到此任務 ID")
    return {
        "status": task["status"],
        "progress": task.get("progress", 0),
        "current_phase": task.get("current_phase", 1),
        "message": task.get("message", ""),
        "total_pages": task.get("total_pages", 0),
        "translated_blocks": task.get("translated_blocks", 0),
        "ocr_blocks_translated": task.get("ocr_blocks_translated", 0),
        "total_translated": task.get("total_translated", 0),
        "error": task.get("error")
    }

@app.get("/api/pages/{task_id}")
def get_pages_metadata(task_id: str):
    """Get metadata for all pages of a task."""
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="找不到此任務")
    return {
        "task_id": task_id,
        "total_pages": task.get("total_pages", 0),
        "filename": task.get("input_filename", ""),
        "pages": task.get("pages_meta", [])
    }

@app.get("/api/page-image/{task_id}/{page_num}")
def get_page_image(task_id: str, page_num: int, type: str = Query("original")):
    """
    Render and return a PDF page as PNG image.
    type: 'original' or 'translated'
    """
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="找不到此任務")

    if type == "translated":
        pdf_path = task.get("output_path")
        if not pdf_path or not os.path.exists(pdf_path):
            # Fallback to original if translated not ready yet
            pdf_path = task.get("input_path")
    else:
        pdf_path = task.get("input_path")

    if not pdf_path or not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF 檔案不存在")

    doc = pymupdf.open(pdf_path)
    if page_num < 1 or page_num > len(doc):
        doc.close()
        raise HTTPException(status_code=400, detail="頁碼超出範圍")

    page = doc[page_num - 1]
    pix = page.get_pixmap(dpi=150)
    png_bytes = pix.tobytes("png")
    doc.close()

    return Response(content=png_bytes, media_type="image/png")

@app.get("/api/download/{task_id}")
def download_translated_pdf(task_id: str):
    """Download the resulting translated PDF."""
    return export_translated_document(task_id=task_id, format="pdf")

@app.get("/api/export/{task_id}")
def export_translated_document(task_id: str, format: str = Query("pdf")):
    """
    Export translated document in requested format:
    format: 'pdf', 'bilingual' (or 'bilingual_pdf'), or 'docx'
    """
    task = tasks.get(task_id)
    task_dir = os.path.join(TASKS_DIR, task_id)
    checkpoint_path = os.path.join(task_dir, "checkpoint.json")

    # If task is not in memory, try to load from disk
    if not task:
        if os.path.exists(checkpoint_path):
            cp_data = load_checkpoint(checkpoint_path)
            if cp_data.get("status") == "completed":
                task = {
                    "status": "completed",
                    "input_filename": f"{task_id}.pdf",
                    "output_path": cp_data.get("output_path", os.path.join(task_dir, "output.pdf")),
                    "input_path": os.path.join(task_dir, "input.pdf"),
                    "pairs": cp_data.get("pairs", [])
                }

    if not task or task.get("status") != "completed":
        raise HTTPException(status_code=400, detail="任務尚未完成或不存在")

    orig_name = task.get("input_filename", "translated.pdf")
    base_stem = os.path.splitext(orig_name)[0]
    output_path = task.get("output_path")
    input_path = task.get("input_path")

    fmt = format.lower().strip()

    if fmt == "pdf":
        if not output_path or not os.path.exists(output_path):
            # Check legacy output path
            legacy_out = os.path.join(TASKS_DIR, f"{task_id}_out.pdf")
            if os.path.exists(legacy_out):
                output_path = legacy_out
            else:
                raise HTTPException(status_code=404, detail="找不到翻譯後 PDF 檔案")
        return FileResponse(
            output_path,
            media_type="application/pdf",
            filename=f"{base_stem}_translated.pdf"
        )

    elif fmt in ("bilingual", "bilingual_pdf"):
        if not input_path or not os.path.exists(input_path):
            legacy_in = os.path.join(TASKS_DIR, f"{task_id}_in.pdf")
            if os.path.exists(legacy_in):
                input_path = legacy_in
            else:
                raise HTTPException(status_code=404, detail="找不到原始 PDF 檔案")

        if not output_path or not os.path.exists(output_path):
            legacy_out = os.path.join(TASKS_DIR, f"{task_id}_out.pdf")
            if os.path.exists(legacy_out):
                output_path = legacy_out
            else:
                raise HTTPException(status_code=404, detail="找不到翻譯後 PDF 檔案")

        bilingual_path = os.path.join(task_dir, f"{task_id}_bilingual.pdf") if os.path.isdir(task_dir) else os.path.join(TASKS_DIR, f"{task_id}_bilingual.pdf")
        if not os.path.exists(bilingual_path):
            export_bilingual_pdf(input_path, output_path, bilingual_path)

        return FileResponse(
            bilingual_path,
            media_type="application/pdf",
            filename=f"{base_stem}_bilingual.pdf"
        )

    elif fmt == "docx":
        pairs = task.get("pairs", [])
        if not pairs and os.path.exists(checkpoint_path):
            cp_data = load_checkpoint(checkpoint_path)
            pairs = cp_data.get("pairs", [])

        docx_path = os.path.join(task_dir, f"{task_id}_bilingual.docx") if os.path.isdir(task_dir) else os.path.join(TASKS_DIR, f"{task_id}_bilingual.docx")
        if not os.path.exists(docx_path):
            title = f"{base_stem} 雙語對照文檔"
            if pairs:
                export_docx_comparison(title, pairs, docx_path)
            else:
                export_docx_comparison(input_path, output_path, docx_path)

        return FileResponse(
            docx_path,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            filename=f"{base_stem}_bilingual.docx"
        )

    else:
        raise HTTPException(
            status_code=400,
            detail=f"不支援的匯出格式: '{format}'。支援的格式包含: pdf, bilingual, docx"
        )

def main():
    parser = argparse.ArgumentParser(description="文脈 FolioTrans - AI PDF 雙階段排版翻譯器")
    parser.add_argument("--cli", action="store_true", help="以命令列模式執行")
    parser.add_argument("input_pdf", nargs="?", help="輸入 PDF 檔案")
    parser.add_argument("output_pdf", nargs="?", help="輸出 PDF 檔案")
    parser.add_argument("--extract", action="store_true", help="擷取 PDF 唯一文字為 JSON (免 API 模式)")
    parser.add_argument("--apply", help="套用 translations.json 翻譯映射檔案")
    parser.add_argument("--font", default="china-ts", help="字體名稱 (china-ts, china-ss, helv, japan, korea)")
    parser.add_argument("--font-style", default="mingti", choices=["mingti", "heiti", "kaiu"], help="字體風格 (明體, 黑體)")
    parser.add_argument("--provider", default="gemini", choices=["gemini", "openai", "deepseek", "claude", "ollama", "custom"], help="AI 服務商")
    parser.add_argument("--key", help="API Key")
    parser.add_argument("--lang", default="zh-TW", help="目標語言代碼 (zh-TW, zh-CN, en, ja, ko)")
    parser.add_argument("--domain", default="general", choices=["general", "medical", "academic", "engineering", "legal"], help="學術翻譯領域")
    parser.add_argument("--model", help="自訂模型名稱")
    parser.add_argument("--base-url", help="自訂 API Base URL (適用於 Ollama, DeepSeek, vLLM 等端點)")
    parser.add_argument("--no-ocr", action="store_true", help="關閉第二階段 OCR 掃描辨識")
    parser.add_argument("--port", type=int, default=8000, help="Web 服務連接埠")
    args = parser.parse_args()

    # Mode 1: Extract texts for manual / chatbot translation
    if args.extract:
        if not args.input_pdf:
            print("錯誤: 請指定欲擷取的 PDF 檔案: python app.py input.pdf --extract [output.json]")
            return
        if not os.path.exists(args.input_pdf):
            print(f"錯誤: 找不到輸入檔案: {args.input_pdf}")
            return
        out_json = args.output_pdf or "extracted_texts.json"
        import json
        doc = pymupdf.open(args.input_pdf)
        all_texts = sorted({
            span.get("text", "").strip()
            for page in doc
            for block in page.get_text("dict").get("blocks", [])
            if block.get("type") == 0
            for line in block.get("lines", [])
            for span in line.get("spans", [])
            if span.get("text", "").strip()
        })
        doc.close()
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(all_texts, f, ensure_ascii=False, indent=2)
        print(f"已擷取 {len(all_texts)} 個唯一文字區塊至: {out_json}")
        return

    # Mode 2: Apply JSON translation mapping
    if args.apply:
        if not args.input_pdf or not args.output_pdf:
            print("錯誤: 套用翻譯映射需提供輸入與輸出 PDF: python app.py input.pdf output.pdf --apply translations.json")
            return
        if not os.path.exists(args.input_pdf):
            print(f"錯誤: 找不到輸入的 PDF 檔案: {args.input_pdf}")
            return
        if not os.path.exists(args.apply):
            print(f"錯誤: 找不到翻譯映射檔案: {args.apply}")
            return
        import json
        with open(args.apply, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        sys_path = os.path.join(os.path.dirname(__file__), "skills", "translate-pdf", "scripts")
        import sys
        if sys_path not in sys.path:
            sys.path.insert(0, sys_path)
        from translate_pdf import translate_pdf
        res = translate_pdf(args.input_pdf, mapping, args.output_pdf, fontname=args.font)
        print(f"套用完成！共置換 {res['translated']} 個文字區塊 -> {args.output_pdf}")
        return

    # Mode 3: CLI automated LLM translation
    if args.cli or (args.input_pdf and args.output_pdf):
        if not args.input_pdf or not os.path.exists(args.input_pdf):
            print(f"錯誤: 找不到輸入的 PDF 檔案: {args.input_pdf}")
            return
        if not args.key:
            print("錯誤: CLI 模式請使用 --key 提供 API Key")
            return
        print(f"正在以 CLI 模式執行兩階段翻譯: {args.input_pdf} -> {args.output_pdf}")
        print(f"領域: {args.domain} | OCR: {'關閉' if args.no_ocr else '開啟'}")
        res = translate_pdf_file(
            input_path=args.input_pdf,
            output_path=args.output_pdf,
            provider=args.provider,
            api_key=args.key,
            target_lang_code=args.lang,
            model=args.model,
            base_url=args.base_url,
            domain=args.domain,
            font_style=args.font_style,
            enable_ocr=not args.no_ocr,
            progress_callback=lambda cur, tot, phase, msg: print(f"[Phase {phase}][{cur}/{tot}] {msg}")
        )
        print(f"完成！共處理 {res['total_pages']} 頁，文字層替換 {res['translated_blocks']} 個區塊，OCR 掃描替換 {res['ocr_blocks_translated']} 個區塊。")
        return

    # Mode 4: Launch Web UI
    url = f"http://127.0.0.1:{args.port}"
    print(f"===================================================")
    print(f"🚀 文脈 FolioTrans - AI PDF 雙階段排版翻譯工作台")
    print(f"👉 請開啟瀏覽器造訪: {url}")
    print(f"===================================================")
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run("app:app", host="127.0.0.1", port=args.port, reload=True)

if __name__ == "__main__":
    main()
