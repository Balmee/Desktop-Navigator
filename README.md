# AURA — Adaptive Understanding, Reasoning & Assistance

This starter implements the first development slice:
- Local text conversation through Ollama
- Explicit, on-demand screenshot capture
- Visual question answering through a local vision-language model
- No continuous monitoring and no desktop actions in this phase

## Requirements
- Windows 10/11
- Python 3.11
- Ollama for Windows
- Local Ollama model `qwen3-vl:8b`

## Install the model
Install Ollama from https://ollama.com/download/windows, then run in PowerShell:

```powershell
ollama --version
ollama pull qwen3-vl:8b
```

## Create the environment
From this directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

If activation is blocked:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run
```powershell
python -m aura doctor
python -m aura chat
python -m aura ask-screen "What application am I using, and what is visible?"
```

The screen command asks for confirmation, captures the primary monitor once, sends it to the configured Ollama endpoint and deletes the temporary screenshot afterwards.

## Important limitations
This is a foundation, not the complete AURA system. Conversation history is session-only. Voice, persistent memory, continuous observation, OCR, application context and computer control are future phases.

Screen capture is opt-in. No mouse or keyboard control exists in this version. The screenshot is sent only to the Ollama endpoint in `.env`; keep it local if local-only processing is required.
