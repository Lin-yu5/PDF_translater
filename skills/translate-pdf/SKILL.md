---
name: translate-pdf
description: |
  Translate PDF documents to any language while preserving original structure, layout, and styling (colors, backgrounds, positions).
  Use when user wants to: (1) translate a PDF to another language, (2) convert PDF from one language to another, (3) create translated version of PDF document.
  Triggers: "translate PDF", "PDF翻译", "把PDF翻译成", "translate this PDF to Chinese/English/Japanese", "翻译成中文/英文"
---

# PDF Translation Skill

Translate PDF text while preserving structure, colors, and background styling.

## Automated Translation (Recommended)

Run the automated Web UI or CLI with LLM API (Gemini / OpenAI / Ollama):

```bash
# Launch Web UI
python app.py

# Or via CLI
python app.py input.pdf output.pdf --provider gemini --key YOUR_KEY --lang zh-TW
```

---

## 3-Step Manual / Agent Translation Workflow (Zero API Key Required)

If you do not have an API key and prefer using web chatbots (e.g. ChatGPT, Claude) or agent execution:

### Step 1: Extract texts
```bash
python skills/translate-pdf/scripts/extract_texts.py <input.pdf> --output texts.json
```
Review `texts.json` to see all unique text strings in the PDF.

### Step 2: Create translation mapping
Translate each text to your target language. Create a JSON mapping:

```json
{
  "Original Text 1": "Translated 1",
  "Original Text 2": "Translated 2"
}
```
Save as `translations.json`.

### Step 3: Apply translations
```bash
python skills/translate-pdf/scripts/translate_pdf.py <input.pdf> translations.json <output.pdf> --font china-ts
```

**Font options:**
| Font | Language |
|------|----------|
| `helv` | Latin (English, Spanish, Portuguese, French, German, etc.) |
| `china-ss` | Simplified Chinese |
| `china-ts` | Traditional Chinese |
| `japan` | Japanese |
| `korea` | Korean |

## Output naming
Append language suffix: `filename_EN.pdf`, `filename_ZH.pdf`, `filename_JA.pdf`
