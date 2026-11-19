@echo off
chcp 65001 >nul
cd /d "%~dp0"
title AI PDF 排版翻譯器

if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

echo ===================================================
echo           正在啟動 AI PDF 排版翻譯器...
echo ===================================================
python app.py
pause
