# 文脈 FolioTrans - 專業 PDF 排版翻譯工作台

[English Documentation](#english-documentation) | [繁體中文說明](#繁體中文說明)

---

<a name="繁體中文說明"></a>
## 🇹🇼 繁體中文說明

**文脈 FolioTrans** 是一套專為專業學者、臨床醫護與技術文件閱讀者打造的 **AI PDF 雙階段排版翻譯工作台**。結合 **PyMuPDF 版面解析**、**RapidOCR 本地神經網路文字辨識** 與 **大語言模型 (Gemini / OpenAI / 本地 Ollama)**，能完整保留 PDF 的原始版面、多欄排版、插圖與背景樣式，同時精準置換文字。

### ✨ 核心升級亮點

1. **混合型兩階段翻譯管線 (Two-Phase Translation Pipeline)**：
   - **第一階段（向量文字層）**：精確抽取 PDF 內生文字區塊，進行語境翻譯後透過透明遮罩 (Redaction) 覆寫回原位置，支援自動字級縮放 (Auto Font Fitting)。
   - **第二階段（圖表掃描層 OCR）**：針對投影片、截圖、表格或純圖片頁面，自動調度本地 **RapidOCR (ONNX)** 引擎辨識圖中文字，翻譯後自動以微透明底色框覆貼譯文，圖文不再遺漏。
2. **多專業領域 Skills 規則庫 (Domain Skills Engine)**：
   - 內建 `skills/medical_terms.md`：專門收錄 CP(腦性麻痺)、GMFCS、MACS、ROM、CIMT 等醫學縮寫與臨床專業對應表。
   - 內建 `skills/academic_rules.md`：嚴格保護學術引用標籤 (`[1]`, `[2-4]`)、DOI、統計指標 (`p < 0.05`, `95% CI`)、圖表編號。
   - 內建 `skills/translation_rules.md`：嚴禁任何「譯者註」、「前言說明」等無關贅字，確保 1:1 乾淨輸出。
3. **專業雙聯工作台介面 (FolioTrans UI)**：
   - 左側導航列：來源/目標語言、領域切換、明體/黑體切換、各頁面型態標籤（文字型 / 掃描型 / 混合型）。
   - 中央畫布：支援「原文/譯文雙聯並排」、「譯文預覽」、「原文檢視」與多級縮放預覽。
   - 區塊標註：直觀檢查各頁圖表物件與文字塊偵測位置。
4. **全平台離線安全支援**：
   - 支援本機 Ollama 離線模型（Base URL: `http://localhost:11434/v1`），完全無需連網即可翻譯。
   - 支援 30 秒免費申請的 Google Gemini API。

---

### 🚀 快速啟動

#### 啟動工作台 (Web UI)
- **Windows**: 雙擊 `start.bat`
- **macOS / Linux**: 執行 `chmod +x start.sh && ./start.sh`

系統啟動後將自動在瀏覽器中開啟 `http://127.0.0.1:8000`。

#### 命令行模式 (CLI)
```bash
# 執行混合型兩階段翻譯 (文字 + OCR)
python app.py CP.pdf CP_translated.pdf --provider gemini --key YOUR_KEY --domain medical --lang zh-TW

# 搭配本地 Ollama 模型全離線翻譯
python app.py CP.pdf CP_translated.pdf --provider openai --base-url http://localhost:11434/v1 --key ollama --model qwen2.5:7b --domain medical
```

---

<a name="english-documentation"></a>
## 🌐 English Documentation

**FolioTrans** is a professional **two-phase AI PDF layout-preserving translation workbench**. Built with **PyMuPDF**, **RapidOCR (ONNX Runtime)**, and **LLMs (Gemini / OpenAI / Ollama)**, it translates PDF documents while strictly preserving layout geometry, background images, and typography.

### ✨ Key Features

1. **Two-Phase Pipeline**:
   - **Phase 1 (Native Text)**: Extracts selectable text, translates with context, and re-renders via redaction annotations and adaptive font-scaling.
   - **Phase 2 (Image & Scan OCR)**: Scans embedded diagrams, charts, and slide images using **RapidOCR**, translates detected labels, and overlays readable translations with adaptive backings.
2. **Domain-Specific Skill Bundles**:
   - `skills/medical_terms.md`: Medical and rehabilitation terminology (CP, GMFCS, MACS, ROM, CIMT).
   - `skills/academic_rules.md`: Protection of academic citations (`[1]`, `(Author, Year)`), statistical notation (`p < 0.05`), and figure/table references.
   - `skills/translation_rules.md`: Strict zero-commentary enforcement.
3. **FolioTrans Workbench UI**:
   - Dual-canvas layout (Side-by-side original and translated view).
   - Page inspector with thumbnail list categorizing pages into Text-based, Scanned, or Mixed mode.
   - Font switcher (Mingti serif / Heiti sans-serif) and zoom controls.

---

### 📦 Installation

```bash
pip install -r requirements.txt
```
