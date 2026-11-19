# AI PDF 排版翻譯器 (AI PDF Document Translator)

[English Documentation](#english-documentation) | [繁體中文說明](#繁體中文說明)

---

<a name="繁體中文說明"></a>
## 🇹🇼 繁體中文說明

基於 **PyMuPDF** 與 **LLM (Google Gemini / OpenAI / Ollama)** 的智慧 PDF 翻譯工具，能在替換翻譯文字的同時**完整保留 PDF 原始排版、段落框線、字體顏色與背景樣式**。

### ✨ 核心特色

1. **原版面排版保留**：
   - 採用區塊級文字覆寫（Redaction），保留原背景圖片與色彩。
   - 自動字級微調（Auto Font Fitting）與換行機制，確保翻譯文字不會超出原始邊框。
   - 針對 CJK 語言（繁體中文、簡體中文、日文、韓文）自動載入對應字體。
2. **內建專業翻譯 Skills 規則 (Prompt Enforcement)**：
   - 內建 `skills/translation_rules.md`：**嚴格禁止 AI 輸出任何多餘註解、前言或「譯者注」**，只輸出乾淨的 1:1 譯文對應。
   - 完整保留學術引用標記（如 `[1]`, `[2-4]`）、公式符號、技術專有名詞（如 PyTorch, API, GPU）與專利縮寫。
3. **多 AI 引擎支援**：
   - **Google Gemini API**（推薦：`gemini-2.5-flash`，快速且語境自然）。
   - **OpenAI API**（支援 `gpt-4o-mini`、`gpt-4o` 等）。
   - **本地離線大模型 (Ollama / LM Studio)**：完全免 API Key、零花費、100% 隱私保護。
4. **直覺現代化 Web UI**：
   - 支援拖曳上傳 PDF 檔案與即時進度條。
   - API Key 僅存於瀏覽器本地 `localStorage`，保護使用者金鑰隱私。
5. **跨平台支援**：提供 Windows 一鍵啟動批次檔 (`start.bat`) 與 macOS / Linux 啟動腳本 (`start.sh`)。

---

### 🚀 快速開始

#### Windows 使用者
雙擊專案目錄下的 `start.bat`，系統將自動啟動服務並在瀏覽器中開啟 `http://127.0.0.1:8000`。

#### macOS / Linux 使用者
開啟 Terminal，進入專案目錄後執行：
```bash
chmod +x start.sh
./start.sh
```
或直接透過 Python 啟動：
```bash
python3 app.py
```

#### 命令行批次處理模式 (CLI)
```bash
# 使用 Google Gemini
python app.py input.pdf output.pdf --provider gemini --key YOUR_GEMINI_KEY --lang zh-TW

# 使用本機離線 Ollama (免 API Key / 零花費)
python app.py input.pdf output.pdf --provider openai --key ollama --base-url http://localhost:11434/v1 --model qwen2.5:7b --lang zh-TW
```

---

### 🔑 如何取得 API Key / 串接自己的 AI

#### 方案 A：30 秒免費取得 Gemini API Key（推薦，免綁信用卡）
Google 官方提供免費使用額度：
1. 前往 [Google AI Studio](https://aistudio.google.com/app/apikey)。
2. 使用任一 Google 帳號登入。
3. 點擊 **"Create API key"** 並複製金鑰。
4. 貼回本工具的 API Key 欄位即可開始使用。

#### 方案 B：串接本機離線 AI（Ollama / LM Studio，完全免 API Key）
若您自己電腦有安裝 Ollama 等本地大模型，無需對外連線即可免費翻譯：
1. 確保 Ollama 已啟動（例如模型使用 `qwen2.5:7b`）。
2. 在工具介面設定：
   - **AI 提供商**：選擇 `OpenAI / 相容 API`
   - **API Base URL**：輸入 `http://localhost:11434/v1`
   - **API Key**：隨意輸入（例如 `ollama`）
   - **模型名稱**：輸入您的本地模型名（如 `qwen2.5:7b`）

#### 方案 C：使用內建 translate-pdf Skill（完全免 API Key，貼給網頁版 AI）
專案已內建 [skills/translate-pdf](skills/translate-pdf/SKILL.md)，適合手動將文字丟給 ChatGPT / Claude 網頁版翻譯：
1. **擷取 PDF 文字為 JSON**：
   ```bash
   python app.py input.pdf --extract texts.json
   ```
2. **丟給網頁版 AI**：複製 `texts.json` 內容貼到 ChatGPT / Claude，讓 AI 翻譯為 `{"原文": "譯文"}` 格式並存為 `translations.json`。
3. **一鍵合成回 PDF**：
   ```bash
   python app.py input.pdf output.pdf --apply translations.json --font china-ts
   ```

---

### 📦 依賴套件安裝

```bash
pip install -r requirements.txt
```

---

<a name="english-documentation"></a>
## 🌐 English Documentation

An AI-powered PDF translation tool built with **PyMuPDF** and **LLM (Google Gemini / OpenAI / Ollama)**. It replaces text in-place while **strictly preserving original layout, bounding boxes, background styles, and typography**.

### ✨ Key Features

1. **Exact Layout Preservation**:
   - Uses PDF block redaction to replace text while retaining underlying background art, images, and shading.
   - Intelligent auto-font-fitting and word wrapping to guarantee text never overflows bounding rectangles.
   - Built-in font mappings for CJK languages (Traditional/Simplified Chinese, Japanese, Korean) and Latin scripts.
2. **Embedded Translation Skill Rules**:
   - Packaged with `skills/translation_rules.md`: **Strictly prohibits extraneous AI commentary, preambles, or translator notes** (output is pure 1:1 translation).
   - Preserves citation brackets (`[1]`, `[2-4]`), math formulas, URLs, and standard tech terms (e.g. PyTorch, GPU, LLM) without awkward literal translation.
3. **Multi-Provider Support**:
   - **Google Gemini API** (Recommended: `gemini-2.5-flash`, ultra-fast and natural).
   - **OpenAI API** (`gpt-4o-mini`, `gpt-4o`, etc.).
   - **Local Offline LLMs (Ollama / LM Studio)**: Zero cloud API keys required, zero cost, completely private.
4. **Modern Web UI**:
   - Drag & drop PDF upload, real-time progress bar, and instant download.
   - API keys are saved exclusively in client-side `localStorage`.
5. **Cross-Platform**: Ready-to-use launch scripts for Windows (`start.bat`) and macOS/Linux (`start.sh`).

---

### 🚀 Quick Start

#### For Windows Users
Double-click `start.bat`. The browser will automatically open `http://127.0.0.1:8000`.

#### For macOS / Linux Users
Open your terminal in the project directory:
```bash
chmod +x start.sh
./start.sh
```
Or run directly with Python:
```bash
python3 app.py
```

#### CLI Batch Mode
```bash
# Using Google Gemini
python3 app.py input.pdf output.pdf --provider gemini --key YOUR_GEMINI_KEY --lang zh-TW

# Using Local Offline Ollama (No API Key Required)
python3 app.py input.pdf output.pdf --provider openai --key ollama --base-url http://localhost:11434/v1 --model qwen2.5:7b --lang zh-TW
```

---

### 🔑 Getting an API Key / Using Your Own AI

#### Option 1: Free Google Gemini API Key (Recommended, No Credit Card Required)
1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Sign in with any standard Google account.
3. Click **"Create API key"** and copy your key.
4. Paste the key into the Web UI.

#### Option 2: Connect to Local AI via Ollama (100% Free & Offline)
1. Run Ollama on your machine (e.g., `ollama run qwen2.5:7b`).
2. In the Web UI settings:
   - **AI Provider**: `OpenAI / Compatible API`
   - **API Base URL**: `http://localhost:11434/v1`
   - **API Key**: Enter any placeholder (e.g. `ollama`)
   - **Model Name**: Enter your local model name (e.g. `qwen2.5:7b`)

#### Option 3: Built-in translate-pdf Skill (Zero API Key, Manual Web Chatbot Workflow)
The repository includes [skills/translate-pdf](skills/translate-pdf/SKILL.md) for extracting text and pasting to ChatGPT/Claude web interfaces:
1. **Extract text to JSON**:
   ```bash
   python app.py input.pdf --extract texts.json
   ```
2. **Translate in Chatbot**: Paste `texts.json` into ChatGPT/Claude to generate a `{"Original": "Translated"}` mapping and save as `translations.json`.
3. **Apply translations back to PDF**:
   ```bash
   python app.py input.pdf output.pdf --apply translations.json --font china-ts
   ```

---

### 📦 Installation

```bash
pip install -r requirements.txt
```
