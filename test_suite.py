import sys
import os
import json
import re
import time
import unittest
from unittest.mock import patch, MagicMock
import pymupdf

# Import application modules
import translator
from translator import (
    safe_json_parse,
    extract_translation_map,
    is_translatable,
    clean_ocr_text,
    load_translation_rules,
    call_gemini,
    call_openai,
    translate_blocks,
    analyze_pdf_pages,
    ocr_page_image_regions,
    translate_pdf_file,
    get_ocr_engine,
    FatalAPIError,
    detect_block_alignment,
    sample_background_and_text_color,
    load_checkpoint,
    save_checkpoint,
    export_bilingual_pdf,
    export_docx_comparison
)
import app

class TestTranslatorCore(unittest.TestCase):
    """Test pure functions, parsing, rules, and sanitization in translator.py."""

    def test_is_translatable(self):
        # Translatable
        self.assertTrue(is_translatable("Hello World"))
        self.assertTrue(is_translatable("Cerebral Palsy"))
        self.assertTrue(is_translatable("第 1 頁"))
        
        # Not translatable
        self.assertFalse(is_translatable(""))
        self.assertFalse(is_translatable("A"))
        self.assertFalse(is_translatable("12345"))
        self.assertFalse(is_translatable("2024-11-20"))
        self.assertFalse(is_translatable("10.5%"))
        self.assertFalse(is_translatable("---"))
        self.assertFalse(is_translatable("[1, 2, 3]"))

    def test_clean_ocr_text(self):
        self.assertEqual(clean_ocr_text("  Valid Text  "), "Valid Text")
        self.assertEqual(clean_ocr_text("-----"), "")
        self.assertEqual(clean_ocr_text("======"), "")
        self.assertEqual(clean_ocr_text("..."), "")

    def test_safe_json_parse_and_extract(self):
        # 1. Standard json object
        raw1 = '{"translations": [{"id": 0, "text": "你好"}, {"id": 1, "text": "世界"}]}'
        m1 = extract_translation_map(raw1)
        self.assertEqual(m1, {0: "你好", 1: "世界"})

        # 2. Markdown code block with nested braces
        raw2 = '```json\n{\n  "translations": [\n    {"id": 0, "text": "測試"}\n  ]\n}\n```'
        m2 = extract_translation_map(raw2)
        self.assertEqual(m2, {0: "測試"})

        # 3. Dict-style translations
        raw3 = '{"translations": {"0": "鍵值一", "1": "鍵值二"}}'
        m3 = extract_translation_map(raw3)
        self.assertEqual(m3, {0: "鍵值一", 1: "鍵值二"})

        # 4. Preamble text before json
        raw4 = 'Here is the translation:\n{"translations": [{"id": 10, "text": "成功"}]}\nHope this helps!'
        m4 = extract_translation_map(raw4)
        self.assertEqual(m4, {10: "成功"})

        # 5. Non-integer ID or missing text handled gracefully
        raw5 = '{"translations": [{"id": "not-int", "text": "忽略"}, {"id": 2, "text": "有效"}]}'
        m5 = extract_translation_map(raw5)
        self.assertEqual(m5, {2: "有效"})

    def test_json_array_handling(self):
        """Test whether extract_translation_map tolerates top-level JSON array."""
        raw = '[{"id": 0, "text": "陣列項目一"}, {"id": 1, "text": "陣列項目二"}]'
        try:
            m = extract_translation_map(raw)
            self.assertEqual(m.get(0), "陣列項目一")
        except Exception as e:
            self.fail(f"extract_translation_map crashed on JSON array: {type(e).__name__}: {e}")

    def test_load_translation_rules(self):
        # Test general domain
        rules_general = load_translation_rules("Traditional Chinese", domain="general")
        self.assertIn("Target Language: Traditional Chinese", rules_general)
        self.assertIn("STRICT ZERO EXPLANATION", rules_general)

        # Test medical domain
        rules_med = load_translation_rules("Traditional Chinese", domain="medical")
        self.assertIn("Target Language: Traditional Chinese", rules_med)
        # Check if medical skills file loaded if present
        if os.path.exists("skills/medical_terms.md"):
            self.assertIn("CP", rules_med)


class TestProviderRouting(unittest.TestCase):
    """Test provider dispatch and parameter routing (Gemini, DeepSeek, Claude, OpenAI, Ollama)."""

    @patch("translator.rate_limit_pace")
    @patch("requests.post")
    def test_gemini_routing(self, mock_post, mock_pace):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{"text": '{"translations": [{"id": 0, "text": "雙子座譯文"}]}'}]
                }
            }]
        }
        mock_post.return_value = mock_resp

        items = [{"id": 0, "text": "Gemini text"}]
        res = translate_blocks(items, provider="gemini", api_key="gemini-key", target_lang_desc="Traditional Chinese", model="gemini-1.5-pro")
        
        self.assertEqual(res, {0: "雙子座譯文"})
        mock_post.assert_called_once()
        called_url = mock_post.call_args[0][0]
        self.assertIn("generativelanguage.googleapis.com", called_url)
        self.assertIn("gemini-1.5-pro", called_url)
        self.assertIn("key=gemini-key", called_url)

    @patch("requests.post")
    def test_openai_routing(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{
                "message": {"content": '{"translations": [{"id": 0, "text": "OpenAI 譯文"}]}'}
            }]
        }
        mock_post.return_value = mock_resp

        items = [{"id": 0, "text": "OpenAI text"}]
        res = translate_blocks(items, provider="openai", api_key="sk-openai", target_lang_desc="Traditional Chinese")
        
        self.assertEqual(res, {0: "OpenAI 譯文"})
        called_url = mock_post.call_args[0][0]
        self.assertEqual(called_url, "https://api.openai.com/v1/chat/completions")
        headers = mock_post.call_args[1]["headers"]
        self.assertEqual(headers["Authorization"], "Bearer sk-openai")
        payload = mock_post.call_args[1]["json"]
        self.assertEqual(payload["model"], "gpt-4o-mini")

    @patch("requests.post")
    def test_deepseek_routing(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{
                "message": {"content": '{"translations": [{"id": 1, "text": "DeepSeek 譯文"}]}'}
            }]
        }
        mock_post.return_value = mock_resp

        items = [{"id": 1, "text": "DeepSeek text"}]
        res = translate_blocks(items, provider="deepseek", api_key="sk-deepseek", target_lang_desc="Traditional Chinese")
        
        self.assertEqual(res, {1: "DeepSeek 譯文"})
        called_url = mock_post.call_args[0][0]
        self.assertEqual(called_url, "https://api.deepseek.com/v1/chat/completions")
        headers = mock_post.call_args[1]["headers"]
        self.assertEqual(headers["Authorization"], "Bearer sk-deepseek")
        payload = mock_post.call_args[1]["json"]
        self.assertEqual(payload["model"], "deepseek-chat")

    @patch("requests.post")
    def test_ollama_routing(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{
                "message": {"content": '{"translations": [{"id": 2, "text": "Ollama 譯文"}]}'}
            }]
        }
        mock_post.return_value = mock_resp

        items = [{"id": 2, "text": "Ollama text"}]
        res = translate_blocks(items, provider="ollama", api_key="", target_lang_desc="Traditional Chinese")
        
        self.assertEqual(res, {2: "Ollama 譯文"})
        called_url = mock_post.call_args[0][0]
        self.assertEqual(called_url, "http://localhost:11434/v1/chat/completions")
        payload = mock_post.call_args[1]["json"]
        self.assertEqual(payload["model"], "qwen2.5:7b")

    @patch("requests.post")
    def test_claude_native_routing(self, mock_post):
        """Test Claude provider native Anthropic endpoint routing."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "content": [{"type": "text", "text": '{"translations": [{"id": 3, "text": "Claude 原生譯文"}]}'}]
        }
        mock_post.return_value = mock_resp

        items = [{"id": 3, "text": "Claude text"}]
        res = translate_blocks(items, provider="claude", api_key="sk-ant-test", target_lang_desc="Traditional Chinese")
        
        self.assertEqual(res, {3: "Claude 原生譯文"})
        called_url = mock_post.call_args[0][0]
        self.assertEqual(called_url, "https://api.anthropic.com/v1/messages")
        headers = mock_post.call_args[1]["headers"]
        self.assertEqual(headers["x-api-key"], "sk-ant-test")
        self.assertEqual(headers["anthropic-version"], "2023-06-01")
        payload = mock_post.call_args[1]["json"]
        self.assertEqual(payload["model"], "claude-3-5-sonnet-20241022")

    @patch("requests.post")
    def test_claude_proxy_routing(self, mock_post):
        """Test Claude provider with OpenAI-compatible proxy (OpenRouter / OneAPI)."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": '{"translations": [{"id": 4, "text": "Claude Proxy 譯文"}]}'}}]
        }
        mock_post.return_value = mock_resp

        items = [{"id": 4, "text": "Claude proxy text"}]
        res = translate_blocks(
            items,
            provider="claude",
            api_key="sk-proxy-test",
            target_lang_desc="Traditional Chinese",
            base_url="https://openrouter.ai/api/v1"
        )
        self.assertEqual(res, {4: "Claude Proxy 譯文"})
        called_url = mock_post.call_args[0][0]
        self.assertEqual(called_url, "https://openrouter.ai/api/v1/chat/completions")
        headers = mock_post.call_args[1]["headers"]
        self.assertEqual(headers["Authorization"], "Bearer sk-proxy-test")

    def test_unsupported_provider(self):
        items = [{"id": 0, "text": "test"}]
        with self.assertRaises(ValueError) as ctx:
            translate_blocks(items, provider="unsupported_ai", api_key="key", target_lang_desc="Traditional Chinese")
        self.assertIn("Unsupported provider", str(ctx.exception))

    @patch("requests.post")
    def test_fatal_api_error_fast_fail(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized: Invalid API Key"
        mock_post.return_value = mock_resp

        items = [{"id": 0, "text": "test"}]
        with self.assertRaises(FatalAPIError):
            translate_blocks(items, provider="openai", api_key="bad-key", target_lang_desc="Traditional Chinese")


class TestPipelineAndEdgeCases(unittest.TestCase):
    """Test full translation pipeline with mock responses and edge cases."""

    def setUp(self):
        self.temp_in = "temp_test_in.pdf"
        self.temp_out = "temp_test_out.pdf"

    def tearDown(self):
        for f in (self.temp_in, self.temp_out):
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

    @patch("translator.translate_blocks")
    def test_translate_pdf_file_mock_flow(self, mock_translate):
        # Create a test PDF with 2 pages:
        # Page 1: Native text
        # Page 2: Empty page
        doc = pymupdf.open()
        p1 = doc.new_page(width=595, height=842)
        p1.insert_text((50, 50), "Section Title: Cerebral Palsy Management", fontsize=14)
        p1.insert_textbox(pymupdf.Rect(50, 80, 400, 150), "Clinical guidelines for pediatric neurological intervention.", fontsize=11)
        
        p2 = doc.new_page(width=595, height=842) # Completely empty page
        doc.save(self.temp_in)
        doc.close()

        # Mock translate response
        mock_translate.side_effect = lambda items, *args, **kwargs: {
            item["id"]: f"翻譯結果: {item['text'][:10]}" for item in items
        }

        progress_records = []
        def on_prog(cur, tot, phase, msg):
            progress_records.append((cur, tot, phase, msg))

        res = translate_pdf_file(
            input_path=self.temp_in,
            output_path=self.temp_out,
            provider="gemini",
            api_key="mock-key",
            target_lang_code="zh-TW",
            enable_ocr=True,
            progress_callback=on_prog
        )

        self.assertEqual(res["total_pages"], 2)
        self.assertGreater(res["translated_blocks"], 0)
        self.assertTrue(os.path.exists(self.temp_out))

        # Verify output PDF can be opened and parsed
        out_doc = pymupdf.open(self.temp_out)
        self.assertEqual(len(out_doc), 2)
        p1_out_text = out_doc[0].get_text()
        self.assertIn("翻譯結果", p1_out_text)
        self.assertNotIn("Cerebral Palsy Management", p1_out_text)
        out_doc.close()

        # Verify progress callback was triggered
        self.assertTrue(len(progress_records) > 0)

    def test_empty_pdf_no_text_no_ocr(self):
        doc = pymupdf.open()
        doc.new_page(width=400, height=400)
        doc.save(self.temp_in)
        doc.close()

        with self.assertRaises(ValueError) as ctx:
            translate_pdf_file(
                input_path=self.temp_in,
                output_path=self.temp_out,
                provider="gemini",
                api_key="mock-key",
                enable_ocr=False
            )
        self.assertIn("未偵測到文字層", str(ctx.exception))

    def test_cp_pdf_analysis(self):
        """Verify CP.pdf can be parsed and analyzed by analyze_pdf_pages."""
        cp_path = "CP.pdf"
        if not os.path.exists(cp_path):
            self.skipTest("CP.pdf not found in workspace")

        doc = pymupdf.open(cp_path)
        self.assertEqual(len(doc), 94)
        pages_meta = analyze_pdf_pages(doc)
        self.assertEqual(len(pages_meta), 94)

        # Check page types distribution
        type_counts = {}
        for p in pages_meta:
            t = p["page_type"]
            type_counts[t] = type_counts.get(t, 0) + 1

        print("\nCP.pdf page classification summary:", type_counts)
        self.assertIn("text", type_counts)
        self.assertIn("scan", type_counts)
        self.assertIn("mixed", type_counts)
        doc.close()


    def test_ocr_failure_recovery(self):
        """Verify that OCR failure or exception on a page is recovered gracefully."""
        doc = pymupdf.open()
        p = doc.new_page(width=300, height=300)
        
        with patch("translator.get_ocr_engine") as mock_engine:
            # Simulate OCR engine throwing an unexpected exception
            mock_bad = MagicMock(side_effect=RuntimeError("Simulated OCR Model Crash"))
            mock_engine.return_value = mock_bad
            
            regions = ocr_page_image_regions(p)
            self.assertEqual(regions, [], "OCR failure should gracefully return empty list without crashing")
        doc.close()


class TestFastAPIEndpoints(unittest.TestCase):
    """Test app.py routes and server endpoints."""

    @classmethod
    def setUpClass(cls):
        """Start a test instance of uvicorn in background thread for live HTTP probing."""
        import uvicorn
        import threading
        import requests

        cls.port = 8765
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        cls.config = uvicorn.Config(app.app, host="127.0.0.1", port=cls.port, log_level="error")
        cls.server = uvicorn.Server(cls.config)
        cls.thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.thread.start()

        cls.session = requests.Session()
        cls.session.trust_env = False

        # Wait for server ready
        for _ in range(30):
            try:
                r = cls.session.get(cls.base_url)
                if r.status_code == 200:
                    break
            except Exception:
                time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        cls.server.should_exit = True
        cls.thread.join(timeout=1.0)

    def test_live_serve_index_html(self):
        resp = self.session.get(self.base_url)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/html", resp.headers.get("content-type", ""))
        self.assertIn("文脈 FolioTrans", resp.text)
        self.assertIn("下載譯文 PDF", resp.text)

    def test_live_status_404(self):
        resp = self.session.get(f"{self.base_url}/api/status/invalid_task_id_12345")
        self.assertEqual(resp.status_code, 404)
        data = resp.json()
        self.assertIn("detail", data)

    def test_live_upload_validation_non_pdf(self):
        files = {"file": ("test.txt", b"plain text", "text/plain")}
        data = {"provider": "gemini", "api_key": "test"}
        resp = self.session.post(f"{self.base_url}/api/translate", files=files, data=data)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("請上傳 .pdf", resp.json()["detail"])

    @patch("app.threading.Thread")
    def test_live_upload_valid_pdf_starts_task(self, mock_thread_cls):
        mock_thread = MagicMock()
        mock_thread_cls.return_value = mock_thread

        # Create small test PDF
        doc = pymupdf.open()
        p = doc.new_page(width=400, height=400)
        p.insert_text((50, 50), "Live test PDF", fontsize=12)
        pdf_bytes = doc.tobytes()
        doc.close()

        files = {"file": ("test_doc.pdf", pdf_bytes, "application/pdf")}
        data = {
            "provider": "ollama",
            "api_key": "",
            "target_lang": "zh-TW",
            "enable_ocr": "false"
        }
        resp = self.session.post(f"{self.base_url}/api/translate", files=files, data=data)
        self.assertEqual(resp.status_code, 200)
        res_json = resp.json()
        self.assertIn("task_id", res_json)
        self.assertEqual(res_json["total_pages"], 1)

        task_id = res_json["task_id"]
        # Poll status
        status_resp = self.session.get(f"{self.base_url}/api/status/{task_id}")
        self.assertEqual(status_resp.status_code, 200)
        self.assertIn("status", status_resp.json())

        # Check pages metadata endpoint
        pages_resp = self.session.get(f"{self.base_url}/api/pages/{task_id}")
        self.assertEqual(pages_resp.status_code, 200)
        self.assertEqual(pages_resp.json()["total_pages"], 1)

        # Check page image endpoint
        img_resp = self.session.get(f"{self.base_url}/api/page-image/{task_id}/1?type=original")
        self.assertEqual(img_resp.status_code, 200)
        self.assertEqual(img_resp.headers.get("content-type"), "image/png")
        self.assertGreater(len(img_resp.content), 100)

    def test_index_html_exists_and_content(self):
        self.assertTrue(os.path.exists("index.html"))
        with open("index.html", "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("文脈 FolioTrans", content)
        self.assertIn("下載譯文 PDF", content)
        self.assertIn("開始雙階段翻譯", content)

    def test_serve_ui_function(self):
        resp = app.serve_ui()
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.path, os.path.abspath("index.html"))

    def test_status_endpoint_non_existent(self):
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            app.get_task_status("non_existent_task_id")
        self.assertEqual(ctx.exception.status_code, 404)

    def test_pages_metadata_non_existent(self):
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            app.get_pages_metadata("non_existent_task_id")
        self.assertEqual(ctx.exception.status_code, 404)

    def test_download_not_completed(self):
        from fastapi import HTTPException
        task_id = "test_task_in_progress"
        app.tasks[task_id] = {"status": "processing"}
        try:
            with self.assertRaises(HTTPException) as ctx:
                app.download_translated_pdf(task_id)
            self.assertEqual(ctx.exception.status_code, 400)
        finally:
            app.tasks.pop(task_id, None)

    @patch("app.threading.Thread")
    def test_live_export_and_resume_endpoints(self, mock_thread_cls):
        mock_thread = MagicMock()
        mock_thread_cls.return_value = mock_thread

        # Create a mock completed task
        test_task_id = "export_test_task_123"
        test_in = os.path.join(app.TASKS_DIR, f"{test_task_id}_in.pdf")
        test_out = os.path.join(app.TASKS_DIR, f"{test_task_id}_out.pdf")

        # Create dummy PDFs
        d1 = pymupdf.open()
        p1 = d1.new_page(width=500, height=300)
        p1.insert_text((50, 50), "Original PDF for export test", fontsize=12)
        d1.save(test_in)
        d1.close()

        d2 = pymupdf.open()
        p2 = d2.new_page(width=500, height=300)
        p2.insert_text((50, 50), "匯出測試翻譯後 PDF", fontsize=12, fontname="china-ts")
        d2.save(test_out)
        d2.close()

        app.tasks[test_task_id] = {
            "status": "completed",
            "input_filename": "paper.pdf",
            "input_path": test_in,
            "output_path": test_out,
            "pairs": [{"page": 1, "source": "Original", "target": "譯文"}]
        }

        try:
            # 1. Test export format=pdf
            resp_pdf = self.session.get(f"{self.base_url}/api/export/{test_task_id}?format=pdf")
            self.assertEqual(resp_pdf.status_code, 200)
            self.assertEqual(resp_pdf.headers.get("content-type"), "application/pdf")
            self.assertIn("paper_translated.pdf", resp_pdf.headers.get("content-disposition", ""))

            # 2. Test export format=bilingual
            resp_bi = self.session.get(f"{self.base_url}/api/export/{test_task_id}?format=bilingual")
            self.assertEqual(resp_bi.status_code, 200)
            self.assertEqual(resp_bi.headers.get("content-type"), "application/pdf")
            self.assertIn("paper_bilingual.pdf", resp_bi.headers.get("content-disposition", ""))

            # 3. Test export format=docx
            resp_docx = self.session.get(f"{self.base_url}/api/export/{test_task_id}?format=docx")
            self.assertEqual(resp_docx.status_code, 200)
            self.assertIn("wordprocessingml", resp_docx.headers.get("content-type", ""))
            self.assertIn("paper_bilingual.docx", resp_docx.headers.get("content-disposition", ""))

            # 4. Test export invalid format -> 400
            resp_bad = self.session.get(f"{self.base_url}/api/export/{test_task_id}?format=unknown_fmt")
            self.assertEqual(resp_bad.status_code, 400)

            # 5. Test resume endpoint
            resp_res = self.session.post(f"{self.base_url}/api/resume/{test_task_id}")
            self.assertEqual(resp_res.status_code, 200)
            self.assertEqual(resp_res.json()["status"], "resumed")

        finally:
            app.tasks.pop(test_task_id, None)
            for f in (test_in, test_out, os.path.join(app.TASKS_DIR, f"{test_task_id}_bilingual.pdf"), os.path.join(app.TASKS_DIR, f"{test_task_id}_bilingual.docx")):
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except Exception:
                        pass


class TestCheckpointAndResume(unittest.TestCase):
    """Test disk checkpointing, persistence, and skipping already translated batches."""

    def setUp(self):
        self.test_dir = "temp_test_checkpoint"
        os.makedirs(self.test_dir, exist_ok=True)
        self.cp_path = os.path.join(self.test_dir, "checkpoint.json")

    def tearDown(self):
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_save_and_load_checkpoint(self):
        data = {
            "native_translations": {0: "文字一", 1: "文字二"},
            "ocr_translations": {0: "OCR一"},
            "pairs": [{"page": 1, "source": "A", "target": "甲"}]
        }
        save_checkpoint(self.cp_path, data)
        self.assertTrue(os.path.exists(self.cp_path))

        loaded = load_checkpoint(self.cp_path)
        self.assertEqual(loaded["native_translations"], {0: "文字一", 1: "文字二"})
        self.assertEqual(loaded["ocr_translations"], {0: "OCR一"})
        self.assertEqual(len(loaded["pairs"]), 1)

    @patch("translator.translate_blocks")
    def test_checkpoint_skips_cached_batches(self, mock_translate):
        test_in = os.path.join(self.test_dir, "in.pdf")
        test_out = os.path.join(self.test_dir, "out.pdf")

        # Create 1 page with 2 blocks
        doc = pymupdf.open()
        p = doc.new_page(width=500, height=400)
        p.insert_text((50, 50), "First block text", fontsize=12)
        p.insert_text((50, 100), "Second block text", fontsize=12)
        doc.save(test_in)
        doc.close()

        # Pre-seed checkpoint with translations for both blocks
        pre_seed = {
            "native_translations": {0: "已翻譯第一段", 1: "已翻譯第二段"},
            "ocr_translations": {}
        }
        save_checkpoint(self.cp_path, pre_seed)

        # Run translation with seeded checkpoint
        res = translate_pdf_file(
            input_path=test_in,
            output_path=test_out,
            provider="gemini",
            api_key="mock-key",
            checkpoint_path=self.cp_path
        )

        # Verify translate_blocks was NEVER called because all items were cached in checkpoint
        mock_translate.assert_not_called()
        self.assertEqual(res["translated_blocks"], 2)

        # Check output PDF contains pre-seeded translations
        out_doc = pymupdf.open(test_out)
        out_text = out_doc[0].get_text()
        out_doc.close()
        self.assertIn("已翻譯第一段", out_text)
        self.assertIn("已翻譯第二段", out_text)


class TestLayoutEngineUpgrades(unittest.TestCase):
    """Test alignment detection, redaction padding, subtle background sampling, and auto-fit."""

    def test_detect_block_alignment(self):
        page_width = 600.0

        # Centered block: left_margin ~ right_margin ~ 200, width = 200
        center_block = {
            "bbox": [200.0, 50.0, 400.0, 70.0],
            "lines": [{"bbox": [200.0, 50.0, 400.0, 70.0]}]
        }
        align_c = detect_block_alignment(center_block, page_width)
        self.assertEqual(align_c, pymupdf.TEXT_ALIGN_CENTER)

        # Right-aligned block: right_margin = 20, left_margin = 480
        right_block = {
            "bbox": [480.0, 50.0, 580.0, 70.0],
            "lines": [{"bbox": [480.0, 50.0, 580.0, 70.0]}]
        }
        align_r = detect_block_alignment(right_block, page_width)
        self.assertEqual(align_r, pymupdf.TEXT_ALIGN_RIGHT)

        # Left-aligned block: left_margin = 50, right_margin = 350
        left_block = {
            "bbox": [50.0, 50.0, 250.0, 70.0],
            "lines": [{"bbox": [50.0, 50.0, 250.0, 70.0]}]
        }
        align_l = detect_block_alignment(left_block, page_width)
        self.assertEqual(align_l, pymupdf.TEXT_ALIGN_LEFT)

    def test_sample_background_and_text_color(self):
        # Create a white pixmap
        doc = pymupdf.open()
        p = doc.new_page(width=200, height=200)
        pix = p.get_pixmap()
        rect = pymupdf.Rect(50, 50, 150, 150)
        bg, text_col = sample_background_and_text_color(pix, rect, p.rect)
        doc.close()

        # Light background -> dark text
        self.assertGreater(bg[0], 0.8)
        self.assertLess(text_col[0], 0.5)


class TestExportCapabilities(unittest.TestCase):
    """Test bilingual PDF and docx export generation."""

    def setUp(self):
        self.tmp_dir = "temp_test_export"
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.in_pdf = os.path.join(self.tmp_dir, "input.pdf")
        self.trans_pdf = os.path.join(self.tmp_dir, "translated.pdf")
        self.bi_pdf = os.path.join(self.tmp_dir, "bilingual.pdf")
        self.bi_docx = os.path.join(self.tmp_dir, "bilingual.docx")

        # Create input and translated PDFs
        d1 = pymupdf.open()
        p1 = d1.new_page(width=400, height=300)
        p1.insert_text((50, 50), "Original Text Page 1", fontsize=12)
        d1.save(self.in_pdf)
        d1.close()

        d2 = pymupdf.open()
        p2 = d2.new_page(width=400, height=300)
        p2.insert_text((50, 50), "譯文第一頁", fontsize=12, fontname="china-ts")
        d2.save(self.trans_pdf)
        d2.close()

    def tearDown(self):
        import shutil
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_export_bilingual_pdf(self):
        export_bilingual_pdf(self.in_pdf, self.trans_pdf, self.bi_pdf)
        self.assertTrue(os.path.exists(self.bi_pdf))

        doc = pymupdf.open(self.bi_pdf)
        self.assertEqual(len(doc), 1)
        # Side-by-side spread width should be 400 + 400 = 800
        self.assertEqual(round(doc[0].rect.width), 800)
        self.assertEqual(round(doc[0].rect.height), 300)
        doc.close()

    def test_export_docx_comparison(self):
        pairs = [
            {"page": 1, "source": "Cerebral palsy is a group of disorders.", "target": "腦性麻痺是一組運動與姿勢障礙群。"},
            {"page": 1, "source": "Diagnosis and management guidelines.", "target": "診斷與管理指引。"}
        ]
        export_docx_comparison("文脈測試雙語文檔", pairs, self.bi_docx)
        self.assertTrue(os.path.exists(self.bi_docx))
        self.assertGreater(os.path.getsize(self.bi_docx), 500)

        # Verify it is a valid zip containing word/document.xml
        import zipfile
        with zipfile.ZipFile(self.bi_docx, "r") as z:
            namelist = z.namelist()
            self.assertIn("word/document.xml", namelist)
            self.assertIn("[Content_Types].xml", namelist)
            xml_content = z.read("word/document.xml").decode("utf-8")
            self.assertIn("腦性麻痺", xml_content)
            self.assertIn("Cerebral palsy", xml_content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
