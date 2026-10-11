<p align="center">
  <img src="sources/icon.png" width="96" alt="AINote icon">
</p>
<h1 align="center">AINote</h1>

<p align="center"><em>Turn videos into notes — then let the notes grow new ideas</em></p>

<p align="center">🌐 <a href="README.md">简体中文</a> · <a href="README_EN.md">English</a></p>

<p align="center"><a href="LICENSE"><img style="display: inline-block; vertical-align: middle;" src="https://img.shields.io/github/license/qiyu166/AINote?style=flat-square&label=license&color=22c55e" alt="MIT License"></a>&nbsp;<a href="https://github.com/qiyu166/AINote/actions/workflows/ci.yml"><img style="display: inline-block; vertical-align: middle;" src="https://img.shields.io/github/actions/workflow/status/qiyu166/AINote/ci.yml?branch=main&style=flat-square&label=CI" alt="CI status"></a>&nbsp;<img style="display: inline-block; vertical-align: middle;" src="https://img.shields.io/badge/frontend-Vanilla%20JS-E34F26?style=flat-square&logo=javascript&logoColor=white" alt="Frontend">&nbsp;<img style="display: inline-block; vertical-align: middle;" src="https://img.shields.io/badge/backend-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white" alt="Backend">&nbsp;<img style="display: inline-block; vertical-align: middle;" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+"></p>

> A local-first **"video → structured notes → Obsidian vault"** service.
> Forked from [like-attract/video-to-note](https://github.com/like-attract/video-to-note) (VideoToNo v1.4.5),
> focused on three **thinking modes** — 🌱 Sprout / ⚔️ Interrogate / 🎯 Quiz — and **Obsidian backlink integration**.

Turning a Bilibili / Douyin / local video into a timestamped Markdown note is only the starting point.
AINote cares about **what happens after the note exists**: let it grow new ideas (Sprout), let it be challenged (Interrogate),
let it actually stick (Quiz) — and write all of that straight into your Obsidian vault, wired together with backlinks.

---

## 🆚 Differences from upstream VideoToNo

AINote is not a re-skin. Everything below is new or rewritten in this fork:

| Feature | Description |
|---|---|
| 🌱 **Sprout** | Implements the "Dedao Brain" sprout pattern: **seed → cross-domain collision → insight**. Distils the note's core idea, then hunts for structurally analogous ideas in other fields, producing a "new-knowledge card" |
| ⚔️ **Interrogate** | Finds holes and **forces alternative solutions**. Names the logical fallacies (survivorship bias, hasty generalisation, correlation-as-causation, …), and **attaches an original-video timestamp to every claim** so the criticism is one click away from the source |
| 🎯 **Quiz** | Turns the note into questions you can test yourself with. Answers are folded behind `<details>` and **carry a timestamp back to the video** — closing the loop from "watched it → have notes" to "can I answer without looking?" |
| 🧠 **Aggregate sprout** | Select several past notes and **find the seed that recurs across all of them**, then run the cross-domain collision on that shared seed. A seed that keeps resurfacing across videos is what you actually care about |
| 🔗 **Obsidian backlinks** | Thinking artefacts are written as their own `.md` files into your vault and **bidirectionally linked** with the source note (`[[note]]` / `[[… · 发芽报告]]`), so they show up immediately in the Backlinks pane. Idempotent — never appended twice |
| ⏱️ **Clickable timestamps** | `[02:29-02:57]` in a note becomes a link that jumps to that moment in the video. Sharing/tracking parameters are stripped, and timestamps inside code blocks are left alone |
| 🎙️ **Groq cloud transcription** | New `backend/groq_asr.py`: transcribe with Groq's whisper-large-v3 family — no local model download, and far faster/more accurate than the tiny tier |
| 🗑️ **Task deletion** | Delete individual or multiple past tasks (completed ones included). **Only the working directory is removed — notes already archived in your vault are untouched** |
| 📄 **History paging** | A "Load more" button pages through history instead of stopping at the hard-coded 20-item limit |
| 🛡️ **Local hardening** | ① Host-header validation (DNS-rebinding defence) ② `note_path` confined to the vault directory (arbitrary-file-write defence) ③ atomic note writes + `.bak` backup (so the only copy of a note can't be corrupted) ④ yt-dlp version ceiling |

### Removed relative to upstream

To stay lean, this fork removes the following upstream modules (and cleaned them out of the docs):

- `backend/mcp_server.py` — MCP integration
- `backend/update_check.py` — update checking
- `backend/douyin_login.py` — Douyin login (Douyin scraping was retired due to platform risk control)
- System-tray residency

---

## 🖼️ Screenshots

![Interface](sources/preview.png)

---

## 🚀 Install and run

### Requirements

- Python **3.10+**（developed and verified on 3.12）
- No manual FFmpeg install needed: audio/video decoding goes through PyAV, model downloads through HuggingFace

### Quick start

```bash
git clone https://github.com/qiyu166/AINote.git
cd AINote

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r backend/requirements.txt

cp .env.example .env             # optional: configure the archive directory
python launcher.py
```

The browser opens automatically at `http://127.0.0.1:8000`.

---

## ⚙️ Configuration

Edit `.env` in the project root (already excluded by `.gitignore`, so it is never committed):

| Variable | Description | Default |
|---|---|---|
| `HOST` | Bind address — **keep this at 127.0.0.1** | `127.0.0.1` |
| `PORT` | Port | `8000` |
| `MAX_UPLOAD_MB` | Local upload size cap | `2048` |
| `MAX_CONCURRENT_TASKS` | Concurrent tasks | `1` |
| `TASK_HISTORY_LIMIT` | Retained history entries | `100` |
| `NOTES_OUTPUT_DIR` | **Archive directory** (point it at your Obsidian vault) | empty (`workspace/notes/`) |
| `VIDEOTONOTES_WORKSPACE` | Working directory | `workspace/` inside the project |

### Wiring it into Obsidian (recommended)

```env
NOTES_OUTPUT_DIR=/Users/you/Documents/ObsidianVault
```

Once set:

1. Notes are archived into that directory, screenshots are copied to `<note>.assets/` and rewritten as relative references — so **the vault is self-contained** and survives deleting the working directory or moving machines
2. Clicking "🌱 Thinking" writes the Sprout / Interrogate / Quiz artefact **directly into the vault**, cross-linked with the source note
3. Open the source note → **Backlinks** pane → the thinking artefact is listed there

### Configuring models

Enter your API key under **Model configuration** in the left sidebar. DeepSeek, OpenAI, Groq, Qwen, Zhipu, Moonshot and any OpenAI-compatible gateway are supported.

- "Save to this machine" stores the key on the backend, **keyed by endpoint address** (so a key can never be sent to a different gateway by mistake)
- Groq transcription needs its own Groq key

> **Honest note on key storage**: on Windows keys are encrypted with DPAPI; **on macOS / Linux they fall back to plaintext**, protected only by file permissions (`0600`).
> The `/api/health` endpoint reports `llm_key_storage` truthfully so you can always tell which mode you are in.

---

## 🔁 Processing pipeline

```
Video link / local file
      ↓
Fetch metadata → platform captions / on-device transcription (Whisper, Paraformer, or Groq cloud)
      ↓
Timeline alignment → screenshot extraction (optional)
      ↓
LLM writes the structured note
      ↓
Timestamps → clickable links → archived to the vault
      ↓
        ┌───────────────┴───────────────┐
        ↓                               ↓
   🌱 Three thinking modes           Manual review
   (Sprout / Interrogate / Quiz)
        ↓
  Written into Obsidian + backlinks
```

---

## 🌍 Supported inputs

| Source | Platform captions | On-device transcription | Screenshots |
|---|---|---|---|
| Bilibili (multi-part included) | ✅ | ✅ | ✅ |
| Local video / audio | — | ✅ | ✅ |
| YouTube and other yt-dlp sites | Site-dependent | ✅ | ✅ |
| Douyin | ❌ retired (needs a browser context; risk control outweighs the benefit) | ✅ save locally and upload instead | ✅ |

---

## ❓ FAQ

**Why does it need a Bilibili cookie?**
Bilibili applies risk control to some endpoints (the video page can return 412). Signing in via QR code markedly improves metadata and caption success rates. Cookies are obtained through a dedicated browser instance (CDP) and stored with the same envelope encryption.

**Will the thinking artefacts overwrite my notes?**
No. Sprout / Interrogate / Quiz are **separate files** (`<note> · 发芽报告.md`, etc.). The source note only gets **one extra backlink line appended**, and that write is backed up to `.bak` and performed via "temp file + atomic replace", so it cannot be corrupted.

**Does re-sprouting pile up files?**
No. Re-running Sprout on the same note **overwrites the same report**; aggregate sprout names the file from the selected notes, so it does not multiply when the model words the seed differently.

**Does deleting a task delete the note in Obsidian?**
No. Deletion only clears the working directory (audio, transcript, screenshots). The vault copy is untouched.

**What happens when there are no captions?**
It falls back to on-device transcription (or whatever cloud transcription you selected).

**Why is the first transcription so slow?**
The first Whisper run downloads a model (roughly 500 MB for small, 3 GB for large-v3). Choosing Groq cloud transcription skips this entirely.

**The model returns a 400 `unsupported_parameter`?**
Some gateways reject the `thinking` / `reasoning_effort` parameters. The app degrades and retries automatically; if it persists, turn off deep thinking on the model configuration page.

---

## 🧪 Development

```bash
pip install -r backend/requirements-dev.txt
python -m pytest tests/ -q
```

Current suite: **358 passed / 2 skipped**.

### Project layout

```
backend/
  main.py            FastAPI app: routes, task pipeline, archiving, security middleware
  llm_summarizer.py  LLM calls and per-mode prompts (sprout / interrogate / quiz / aggregate)
  groq_asr.py        Groq cloud transcription (new in this fork)
  whisper_asr.py     local faster-whisper
  paraformer_asr.py  sherpa-onnx Chinese transcription
  video_processor.py yt-dlp download, captions, screenshots
  config_store.py    local configuration and key storage
  secret_box.py      key encryption at rest (DPAPI / plaintext fallback)
frontend/            vanilla-JS single-page UI
tests/               pytest suite
```

---

## 📄 License

This project is a derivative work based on **[like-attract/video-to-note](https://github.com/like-attract/video-to-note)**
(VideoToNo v1.4.5) and is distributed under the original **MIT License**.

```
Copyright (c) 2026 like-attract (original VideoToNo author, upstream project video-to-note)
Copyright (c) 2026 qiyu166 (AINote derivative work)
```

The MIT licence permits use, modification, distribution and commercial use, **provided the copyright and licence notices above are retained**.
See [LICENSE](LICENSE) for the full text.

### Acknowledgements

- Upstream: [like-attract/video-to-note](https://github.com/like-attract/video-to-note) — the solid local-first architecture, transcription pipeline and note generation all come from there
- The sprout / interrogate / critique / create / review methodology is inspired by **Dedao Brain (得到大脑)** and the "Dedao-style thinking skills" pack on ClawHub (MIT-0)

> AINote is not affiliated with any of the projects above; it is an independent derivative. Please file issues at [this repository](https://github.com/qiyu166/AINote/issues).
