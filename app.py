import os
import time
import uuid
import threading
import webbrowser
import argparse
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

from translator import translate_pdf_file, run_self_check, LANGUAGE_CONFIG

app = FastAPI(title="AI PDF 排版翻譯器")

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
    base_url: Optional[str]
):
    """Worker thread running the translation pipeline."""
    def on_progress(cur_page: int, total_pages: int, message: str):
        pct = 5 if total_pages == 0 else int((cur_page / total_pages) * 90) + 5
        tasks[task_id].update({
            "progress": min(pct, 95),
            "message": message
        })

    try:
        tasks[task_id]["message"] = "開始解析 PDF 文件結構..."
        res = translate_pdf_file(
            input_path=input_path,
            output_path=output_path,
            provider=provider,
            api_key=api_key,
            target_lang_code=target_lang,
            model=model if model else None,
            base_url=base_url if base_url else None,
            progress_callback=on_progress
        )
        tasks[task_id].update({
            "status": "completed",
            "progress": 100,
            "message": "翻譯完成！",
            "total_pages": res["total_pages"],
            "translated_blocks": res["translated_blocks"],
            "output_path": output_path
        })
    except Exception as e:
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
    api_key: str = Form(...),
    target_lang: str = Form("zh-TW"),
    model: Optional[str] = Form(None),
    base_url: Optional[str] = Form(None)
):
    """Upload PDF and trigger background translation."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="請上傳 .pdf 格式的檔案")

    # Clean up any stale task files older than 24 hours
    cleanup_stale_tasks()

    task_id = uuid.uuid4().hex
    input_path = os.path.join(TASKS_DIR, f"{task_id}_in.pdf")
    output_path = os.path.join(TASKS_DIR, f"{task_id}_out.pdf")

    # Save uploaded file
    contents = await file.read()
    with open(input_path, "wb") as f:
        f.write(contents)

    tasks[task_id] = {
        "status": "processing",
        "progress": 5,
        "message": "文件上傳成功，排入處理佇列...",
        "input_filename": file.filename,
        "output_path": output_path,
        "error": None
    }

    # Start thread
    thread = threading.Thread(
        target=background_translate,
        args=(task_id, input_path, output_path, provider, api_key, target_lang, model, base_url),
        daemon=True
    )
    thread.start()

    return {"task_id": task_id}

@app.get("/api/status/{task_id}")
def get_task_status(task_id: str):
    """Query translation status for a task."""
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="找不到此任務 ID")
    return {
        "status": task["status"],
        "progress": task.get("progress", 0),
        "message": task.get("message", ""),
        "total_pages": task.get("total_pages", 0),
        "translated_blocks": task.get("translated_blocks", 0),
        "error": task.get("error")
    }

@app.get("/api/download/{task_id}")
def download_translated_pdf(task_id: str):
    """Download the resulting translated PDF."""
    task = tasks.get(task_id)
    if not task or task.get("status") != "completed":
        raise HTTPException(status_code=400, detail="任務尚未完成或不存在")

    output_path = task.get("output_path")
    if not output_path or not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="找不到翻譯後檔案")

    orig_name = task.get("input_filename", "translated.pdf")
    base_stem = os.path.splitext(orig_name)[0]
    download_filename = f"{base_stem}_translated.pdf"

    return FileResponse(
        output_path,
        media_type="application/pdf",
        filename=download_filename
    )

def main():
    parser = argparse.ArgumentParser(description="AI PDF 排版翻譯器")
    parser.add_argument("--cli", action="store_true", help="以命令列模式執行")
    parser.add_argument("input_pdf", nargs="?", help="輸入 PDF 檔案")
    parser.add_argument("output_pdf", nargs="?", help="輸出 PDF 檔案")
    parser.add_argument("--extract", action="store_true", help="擷取 PDF 唯一文字為 JSON (免 API 模式)")
    parser.add_argument("--apply", help="套用 translations.json 翻譯映射檔案")
    parser.add_argument("--font", default="china-ts", help="字體名稱 (china-ts, china-ss, helv, japan, korea)")
    parser.add_argument("--provider", default="gemini", choices=["gemini", "openai"], help="AI 服務商")
    parser.add_argument("--key", help="API Key")
    parser.add_argument("--lang", default="zh-TW", help="目標語言代碼 (zh-TW, zh-CN, en, ja, ko)")
    parser.add_argument("--model", help="自訂模型名稱")
    parser.add_argument("--base-url", help="自訂 API Base URL (適用於 Ollama, DeepSeek, vLLM 等端點)")
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
        import pymupdf, json
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
        # Import script function
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
        print(f"正在以 CLI 模式翻譯: {args.input_pdf} -> {args.output_pdf}")
        res = translate_pdf_file(
            input_path=args.input_pdf,
            output_path=args.output_pdf,
            provider=args.provider,
            api_key=args.key,
            target_lang_code=args.lang,
            model=args.model,
            base_url=args.base_url,
            progress_callback=lambda cur, tot, msg: print(f"[{cur}/{tot}] {msg}")
        )
        print(f"完成！共處理 {res['total_pages']} 頁，排版替換 {res['translated_blocks']} 個文字區塊。")
        return

    # Mode 4: Launch Web UI
    url = f"http://127.0.0.1:{args.port}"
    print(f"🚀 AI PDF 排版翻譯器網頁介面已啟動！")
    print(f"👉 請開啟瀏覽器造訪: {url}")
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=args.port)

if __name__ == "__main__":
    main()
