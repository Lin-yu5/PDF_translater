import os
import sys
import time
import json
import zipfile
import threading
from unittest.mock import patch
import requests
import pymupdf
import uvicorn

# Ensure root dir is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import app
import translator

def mock_translate_impl(items, *args, **kwargs):
    # Simulate high quality translation
    res = {}
    for item in items:
        text = item.get("text", "")
        if "Cerebral Palsy" in text:
            res[item["id"]] = "腦性麻痺的臨床診斷與介入治療"
        elif "Intervention" in text or "CIMT" in text:
            res[item["id"]] = "侷限誘發動作治療 (CIMT) 具高度療效"
        else:
            res[item["id"]] = f"繁中譯文: {text[:20]}"
    return res

def test_full_api_lifecycle():
    port = 8771
    base_url = f"http://127.0.0.1:{port}"
    config = uvicorn.Config(app.app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()

    session = requests.Session()
    session.trust_env = False

    # Wait for server ready
    ready = False
    for _ in range(30):
        try:
            r = session.get(base_url)
            if r.status_code == 200:
                ready = True
                break
        except Exception:
            time.sleep(0.1)

    assert ready, "Failed to start uvicorn test server"

    try:
        with patch("translator.translate_blocks", side_effect=mock_translate_impl):
            print("=====================================================================")
            print("1. Testing /api/translate with 2-Page Clinical Sample PDF")
            print("=====================================================================")

            # Create a 2-page test PDF
            doc = pymupdf.open()
            p1 = doc.new_page(width=600, height=800)
            p1.insert_text((50, 50), "Clinical Diagnosis of Cerebral Palsy", fontsize=14)
            p1.insert_textbox(pymupdf.Rect(50, 80, 500, 200), "Motor impairment in CP is often accompanied by disturbances of sensation and perception.", fontsize=11)
            
            p2 = doc.new_page(width=600, height=800)
            p2.insert_text((50, 50), "Intervention Strategies", fontsize=14)
            p2.insert_textbox(pymupdf.Rect(50, 80, 500, 200), "Constraint-induced movement therapy (CIMT) shows high efficacy.", fontsize=11)
            
            pdf_bytes = doc.tobytes()
            doc.close()

            files = {"file": ("cp_clinical_eval.pdf", pdf_bytes, "application/pdf")}
            data = {
                "provider": "gemini",
                "api_key": "mock_api_key_valid",
                "target_lang": "zh-TW",
                "domain": "medical",
                "font_style": "mingti",
                "enable_ocr": "false"
            }

            resp = session.post(f"{base_url}/api/translate", files=files, data=data)
            assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
            resp_data = resp.json()
            task_id = resp_data["task_id"]
            assert resp_data["total_pages"] == 2
            print(f"[OK] /api/translate started task: {task_id} (total_pages=2)")

            # Test immediate canvas rendering while task is running or just started
            # Original page image
            resp_orig_img = session.get(f"{base_url}/api/page-image/{task_id}/1?type=original")
            assert resp_orig_img.status_code == 200
            assert resp_orig_img.headers["content-type"] == "image/png"
            assert resp_orig_img.content[:8] == b"\x89PNG\r\n\x1a\n"
            print(f"[OK] /api/page-image (type=original) returned valid PNG image ({len(resp_orig_img.content)} bytes)")

            # Immediate translated preview fallback before completion
            resp_trans_fallback = session.get(f"{base_url}/api/page-image/{task_id}/1?type=translated")
            assert resp_trans_fallback.status_code == 200
            assert resp_trans_fallback.headers["content-type"] == "image/png"
            assert resp_trans_fallback.content[:8] == b"\x89PNG\r\n\x1a\n"
            print(f"[OK] /api/page-image (type=translated immediate preview) rendered successfully ({len(resp_trans_fallback.content)} bytes)")

            # Metadata endpoint
            resp_pages = session.get(f"{base_url}/api/pages/{task_id}")
            assert resp_pages.status_code == 200
            assert resp_pages.json()["total_pages"] == 2
            print(f"[OK] /api/pages/{task_id} returned metadata for 2 pages")

            # Wait for background thread to complete
            for _ in range(50):
                st_resp = session.get(f"{base_url}/api/status/{task_id}")
                st = st_resp.json()
                if st.get("status") == "completed":
                    break
                time.sleep(0.1)

            assert st.get("status") == "completed", f"Task failed to complete: {st}"
            assert st.get("progress") == 100
            assert st.get("translated_blocks") > 0
            print(f"[OK] /api/status/{task_id} completed: 100% progress, {st.get('translated_blocks')} blocks translated")

            # Page image after completion (Real translated rendering)
            resp_trans_done = session.get(f"{base_url}/api/page-image/{task_id}/1?type=translated")
            assert resp_trans_done.status_code == 200
            assert resp_trans_done.content[:8] == b"\x89PNG\r\n\x1a\n"
            print(f"[OK] /api/page-image (type=translated post-completion) rendered translated canvas successfully ({len(resp_trans_done.content)} bytes)")

            # Multi-format exports
            print("\n=====================================================================")
            print("2. Testing /api/export/{task_id} (PDF, Bilingual 2-up, DOCX)")
            print("=====================================================================")

            # Export PDF
            resp_exp_pdf = session.get(f"{base_url}/api/export/{task_id}?format=pdf")
            assert resp_exp_pdf.status_code == 200
            assert resp_exp_pdf.headers["content-type"] == "application/pdf"
            assert resp_exp_pdf.content[:4] == b"%PDF"
            assert "cp_clinical_eval_translated.pdf" in resp_exp_pdf.headers.get("content-disposition", "")
            print(f"[OK] /api/export?format=pdf verified ({len(resp_exp_pdf.content)} bytes)")

            # Export Bilingual 2-up PDF
            resp_exp_bi = session.get(f"{base_url}/api/export/{task_id}?format=bilingual")
            assert resp_exp_bi.status_code == 200
            assert resp_exp_bi.headers["content-type"] == "application/pdf"
            assert "cp_clinical_eval_bilingual.pdf" in resp_exp_bi.headers.get("content-disposition", "")
            bi_doc = pymupdf.open(stream=resp_exp_bi.content, filetype="pdf")
            assert len(bi_doc) == 2
            assert bi_doc[0].rect.width == 1200.0, f"Expected 1200.0, got {bi_doc[0].rect.width}"
            bi_doc.close()
            print(f"[OK] /api/export?format=bilingual 2-up PDF verified (2-up width: 1200.0 pt)")

            # Export DOCX
            resp_exp_docx = session.get(f"{base_url}/api/export/{task_id}?format=docx")
            assert resp_exp_docx.status_code == 200
            assert "wordprocessingml" in resp_exp_docx.headers.get("content-type", "")
            assert "cp_clinical_eval_bilingual.docx" in resp_exp_docx.headers.get("content-disposition", "")
            import io
            docx_zip = zipfile.ZipFile(io.BytesIO(resp_exp_docx.content))
            assert "word/document.xml" in docx_zip.namelist()
            docx_zip.close()
            print(f"[OK] /api/export?format=docx OpenXML Word document verified ({len(resp_exp_docx.content)} bytes)")

            # Cold export recovery without in-memory state
            print("\n=====================================================================")
            print("3. Testing Cold Export from Disk Checkpoint (In-memory task cleared)")
            print("=====================================================================")
            del app.tasks[task_id]
            assert task_id not in app.tasks
            resp_cold = session.get(f"{base_url}/api/export/{task_id}?format=pdf")
            assert resp_cold.status_code == 200
            print("[OK] Cold export from disk checkpoint succeeded without in-memory state")

            # Test /api/stop and /api/resume
            print("\n=====================================================================")
            print("4. Testing /api/stop and /api/resume Checkpoint Resumption")
            print("=====================================================================")
            # Start another task with a slow mock
            slow_stop_flag = threading.Event()
            def slow_mock(items, *args, **kwargs):
                slow_stop_flag.wait(timeout=5.0)
                return {item["id"]: f"譯: {item['text']}" for item in items}

            with patch("translator.translate_blocks", side_effect=slow_mock):
                resp2 = session.post(f"{base_url}/api/translate", files=files, data=data)
                task_id2 = resp2.json()["task_id"]

                # Stop it
                resp_stop = session.post(f"{base_url}/api/stop/{task_id2}")
                assert resp_stop.status_code == 200
                assert resp_stop.json()["status"] == "stopped"
                print(f"[OK] Task {task_id2} stopped via /api/stop")
                slow_stop_flag.set()

            # Now resume task_id2
            resp_res = session.post(f"{base_url}/api/resume/{task_id2}", data={"provider": "gemini", "api_key": "resumed_key"})
            assert resp_res.status_code == 200
            assert resp_res.json()["status"] == "resumed"
            print(f"[OK] Task {task_id2} resumed via /api/resume/{task_id2}")

            # Wait for task_id2 to complete
            for _ in range(50):
                st2 = session.get(f"{base_url}/api/status/{task_id2}").json()
                if st2.get("status") == "completed":
                    break
                time.sleep(0.1)

            assert st2.get("status") == "completed", f"Resumed task failed to complete: {st2}"
            print(f"[OK] Resumed task {task_id2} completed successfully from checkpoint!")

            # Clean up task folders
            import shutil
            for tid in (task_id, task_id2):
                td = os.path.join(app.TASKS_DIR, tid)
                if os.path.exists(td):
                    shutil.rmtree(td, ignore_errors=True)

            print("\n=====================================================================")
            print("ALL APP.PY ENDPOINTS AND WORKFLOW CHECKS PASSED (100% SUCCESS)!")
            print("=====================================================================")

    finally:
        server.should_exit = True

if __name__ == "__main__":
    test_full_api_lifecycle()
