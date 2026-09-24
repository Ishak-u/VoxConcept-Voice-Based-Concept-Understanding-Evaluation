# 🎙️ Voice-Based Concept Understanding Analyser (VBCUA)

> An AI-powered Streamlit application that evaluates a student's conceptual understanding from a
> spoken explanation, using speech-to-text, semantic similarity analysis and audio-feature-based
> fluency scoring.

**Operating cost: ₹0 / $0.** Every component is free and open source, and the application runs
end-to-end on a normal laptop with no API key, no credit card and no cloud account. Google Gemini
is supported as an *optional* enhancement on its permanently free tier.

---

## 📌 1. Project Overview

The Voice-Based Concept Understanding Analyser assesses how well someone understands a concept by
analysing the way they explain it out loud.

A recording is transcribed locally with **OpenAI Whisper**, its meaning is compared against an
expected answer using **sentence-transformer embeddings**, and its delivery (pauses, energy,
speaking pace, filler words) is measured with **Librosa**. Those signals are combined into a
weighted score and a letter grade, presented on an interactive **Streamlit** dashboard and
exportable as a formatted **ReportLab** PDF. Every evaluation is stored in a local **SQLite**
database so past performance can be reviewed and compared.

When a free **Google Gemini** API key is configured, Gemini adds qualitative examiner-style
feedback on top of the locally measured scores. Without a key, an equally structured rule-based
evaluator derives feedback from the same measured signals — nothing is stubbed, faked or disabled.

Developed as part of the **APSCHE Internship 2026 — Google Cloud Generative AI Program**.

---

## ✨ 2. Features

| Feature | Status | Description |
|---|---|---|
| 🎤 Audio upload | Working | mp3, wav, m4a, ogg, flac, webm with size/duration/format validation |
| 📝 Speech-to-text | Working | OpenAI Whisper, running fully locally; results cached by audio hash |
| 🧠 Semantic similarity | Working | Sentence-transformer embeddings, with a TF-IDF + keyword-recall fallback |
| 🎵 Audio feature analysis | Working | Duration, pause ratio, long-pause count, energy consistency, waveform envelope |
| 🗣 Fluency scoring | Working | Filler-word rate, speaking pace (wpm), pause control, energy consistency |
| 📊 Weighted scoring & grading | Working | Configurable semantic/fluency weights; A+ → F grade bands |
| 🤖 AI feedback | Working | Gemini strengths/weaknesses/suggestions (optional), cached and rate-limited |
| 🛟 Offline evaluator | Working | Rule-based feedback from real measured signals when Gemini is unavailable |
| 📈 Dashboard | Working | Real waveform plot, score-component bars, delivery metrics, history trend |
| 🗂 History | Working | Browse, paginate, inspect, export to CSV, re-download PDFs, delete records |
| 📄 PDF reports | Working | Formatted A4 report with metrics table, feedback and transcript |
| ♻️ Caching & rate limits | Working | Transcript cache, AI response cache, per-minute and per-day API ceilings |
| 🔐 Configuration | Working | All settings via environment variables or Streamlit secrets; no hard-coded keys |
| 🧪 Test suite | Working | 118 pytest tests covering scoring, database, audio, parsing, PDF and pipeline |

---

## 🛠️ 3. Technology Stack

Unchanged from the original project design.

| Layer | Technology |
|---|---|
| Programming language | Python 3.9+ |
| Web framework / frontend | Streamlit |
| Speech recognition | OpenAI Whisper (local) |
| Semantic analysis | Sentence-Transformers (`all-MiniLM-L6-v2`) / Google Gemini API |
| Audio processing | Librosa, SoundFile (stdlib `wave` fallback) |
| Data processing | NumPy, Pandas |
| Visualization | Matplotlib |
| Report generation | ReportLab |
| Database | SQLite (Python standard library) |
| Testing | Pytest |
| Version control | Git & GitHub |

---

## 📂 4. Project Structure

```text
Voice-Based-Concept-Understanding-Analyser/
│
├── Source Code/                    # The application
│   ├── app.py                      # Streamlit entry point + Home / Analysis / Report pages
│   ├── upload.py                   # Upload page (audio + expected answer, validation)
│   ├── dashboard.py                # Dashboard page (charts, metrics, history analytics)
│   ├── results.py                  # History page (browse, export, delete, re-download)
│   ├── report.py                   # ReportLab PDF generation
│   ├── ui.py                       # Shared theme, cards, empty states, charts
│   ├── analysis.py                 # End-to-end evaluation pipeline
│   ├── transcription.py            # Whisper wrapper (lazy load, caching, ffmpeg checks)
│   ├── semantics.py                # Sentence-transformer similarity + lexical fallback
│   ├── scoring.py                  # Pure scoring logic (similarity, fluency, grades, feedback)
│   ├── audio_features.py           # Librosa / wave feature extraction
│   ├── gemini_client.py            # Gemini calls, caching, rate limiting, response parsing
│   ├── database.py                 # SQLite schema, migrations, CRUD, caches, analytics
│   ├── config.py                   # Environment / secrets driven configuration
│   ├── data/                       # SQLite database lives here (git-ignored)
│   ├── .env.example                # Template for local configuration
│   ├── .gitignore
│   └── .streamlit/
│       ├── config.toml             # Theme + upload limits
│       └── secrets.toml.example    # Alternative to .env (Streamlit Cloud)
│
├── tests/                          # Pytest suite
│   ├── conftest.py
│   ├── test_scoring.py
│   ├── test_database.py
│   ├── test_audio_features.py
│   ├── test_gemini_parsing.py
│   ├── test_config.py
│   ├── test_analysis.py
│   ├── test_report.py
│   └── test_pipeline.py
│
├── 1. Brainstorming & Ideation/    # Project documentation (unchanged)
├── 2. Requirement Analysis/
├── 3. Project Design Phase/
├── 4. Project Planning Phase/
├── 5. Project Development Phase/
├── 6. Project Testing/
├── 7. Project Documentataion/
├── 8. Project Demonstration/
│
├── requirements.txt
├── pytest.ini
├── run.sh                          # One-command launcher (macOS / Linux)
├── run.bat                         # One-command launcher (Windows)
├── README.md
└── .gitignore
```

---

## 🔄 5. Project Workflow

```text
Audio upload + expected answer
        │
        ▼
Validation (format, size, duration)
        │
        ▼
Audio feature extraction (Librosa) ──► duration, pause ratio, energy, waveform
        │
        ▼
Speech-to-text (Whisper, cached)  ──► transcript
        │
        ▼
Semantic similarity (embeddings)  ──► similarity %
        │
        ▼
Fluency scoring                   ──► fillers, pace, pauses, energy
        │
        ▼
Feedback  ──► Gemini (if configured, cached) OR offline rule-based evaluator
        │
        ▼
Weighted score + grade  ──►  SQLite  ──►  Dashboard  ──►  PDF report
```

---

## ✅ 6. Requirements / Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.9 – 3.12 | 3.11 recommended |
| pip | Ships with Python |
| ~3 GB free disk | PyTorch, Whisper model (~140 MB for `base`), MiniLM (~90 MB) |
| ffmpeg | **Recommended.** Required by Whisper to decode mp3/m4a/ogg. Without it only `.wav` works |
| Internet (first run only) | To install packages and download the models once |
| Google Gemini API key | **Optional.** Free, no credit card: https://aistudio.google.com/apikey |

Installing ffmpeg:

```bash
# Windows (Chocolatey)
choco install ffmpeg

# macOS (Homebrew)
brew install ffmpeg

# Debian / Ubuntu
sudo apt update && sudo apt install ffmpeg
```

---

## ⚙️ 7. Installation

```bash
# 1. Clone
git clone https://github.com/PatiGeetha/Voice-Based-Concept-Understanding-Analyser.git
cd Voice-Based-Concept-Understanding-Analyser

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

> A CPU-only PyTorch install is sufficient and much smaller. If the default `torch` wheel is too
> large for your connection, install it first with:
> `pip install torch --index-url https://download.pytorch.org/whl/cpu`

There is **no NLTK download step** — the stop-word list is built into `scoring.py`.

---

## 🔑 8. Environment Variable Setup

Every variable is optional. **With no configuration at all the app runs fully offline at zero cost.**

```bash
cp "Source Code/.env.example" "Source Code/.env"
```

Then edit `Source Code/.env`:

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | *(empty)* | Optional free Gemini key. Empty → offline evaluator is used |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Preferred model; known free models are tried as fallbacks |
| `WHISPER_MODEL` | `base` | `tiny`, `base`, `small`, `medium`, `large` |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local similarity model |
| `DATABASE_PATH` | `data/results.db` | Relative paths resolve against `Source Code/` |
| `MAX_AUDIO_MB` | `25` | Upload size ceiling |
| `MAX_AUDIO_SECONDS` | `300` | Recording length ceiling |
| `GEMINI_CALLS_PER_MINUTE` | `10` | Self-imposed rate limit |
| `GEMINI_CALLS_PER_DAY` | `180` | Self-imposed daily budget |
| `AI_CACHE_TTL_HOURS` | `720` | How long Gemini replies are reused |
| `SEMANTIC_WEIGHT` | `0.7` | Content weight in the final score |
| `FLUENCY_WEIGHT` | `0.3` | Delivery weight in the final score |
| `OFFLINE_MODE` | `false` | `true` forces the local evaluator even with a key present |
| `HISTORY_PAGE_SIZE` | `25` | Rows per page on the History screen |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

Streamlit secrets work too — copy `Source Code/.streamlit/secrets.toml.example` to
`secrets.toml` instead. Both `.env` and `secrets.toml` are git-ignored and must never be committed.

---

## 🗄️ 9. Database Setup

**No manual setup is needed.** SQLite is part of the Python standard library, and
`Source Code/database.py` creates and migrates the schema automatically on first launch at
`Source Code/data/results.db`.

Tables created:

| Table | Purpose |
|---|---|
| `results` | One row per evaluation (scores, grade, transcript, feedback, metrics) |
| `transcripts` | Whisper transcript cache, keyed by SHA-256 of the audio |
| `ai_cache` | Gemini response cache, keyed by SHA-256 of the prompt |
| `api_usage` | Timestamps of Gemini calls, used for client-side rate limiting |
| `meta` | Schema version |

An older `results.db` from a previous version is **migrated in place** — existing rows are kept and
new columns are added. All queries are parameterised, so SQL injection is structurally impossible.

To reset the history, delete `Source Code/data/results.db` or use **Clear all history** on the
History page.

---

## ▶️ 10. Running the Application

This is a single-process Streamlit application: the "backend" (pipeline, database, models) and the
"frontend" (Streamlit pages) run together in one command. There is no separate server to start.

```bash
streamlit run "Source Code/app.py"
```

Then open http://localhost:8501 (Streamlit opens it automatically).

One-command alternative, which also creates the virtual environment and installs dependencies:

```bash
./run.sh        # macOS / Linux
run.bat         # Windows
```

**First run** downloads the Whisper model (~140 MB) and the MiniLM embedding model (~90 MB) once.
Later runs start in seconds.

### Using the app

1. **🎤 Upload Audio** — choose a recording, type the expected answer, press *Save input*.
2. **🧠 AI Analysis** — press *Run analysis* and watch the progress steps.
3. **📊 Dashboard** — waveform, score components, delivery metrics, history trend.
4. **📄 Report** — download the PDF.
5. **🗂 History** — revisit, export or delete past evaluations.

---

## 🧪 11. Testing

```bash
pytest                    # run everything
pytest -v                 # verbose
pytest tests/test_scoring.py
```

The suite has **118 tests** and runs in a few seconds — Whisper is stubbed in the pipeline tests so
no model download is required. Coverage includes:

| File | What it verifies |
|---|---|
| `test_scoring.py` | Tokenisation, lexical similarity, filler detection, fluency, grade bands, weighted score, offline feedback |
| `test_database.py` | Schema creation, **legacy database migration without data loss**, CRUD, pagination, caches, rate-limit accounting, analytics, SQL-injection safety |
| `test_audio_features.py` | WAV loading, stereo downmix, duration accuracy, silence detection, corrupt-file handling, waveform downsampling |
| `test_gemini_parsing.py` | Prompt building, plain and markdown response parsing, malformed responses, rate limiting, cache-hit avoiding API calls |
| `test_config.py` | Defaults, overrides, invalid values, weight normalisation, **no hard-coded secrets** |
| `test_analysis.py` | Input validation, similarity fallback, record round-trip |
| `test_report.py` | Valid PDF bytes, minimal/empty results, HTML escaping, very long text, non-ASCII |
| `test_pipeline.py` | Full end-to-end runs, persistence, progress callbacks, good vs poor answer scoring, undecodable audio, duration limits, transcription failure, running without a database |

### Manual verification checklist

- [ ] App starts with **no** `.env` file (offline mode, no crash)
- [ ] App starts with a Gemini key present
- [ ] Upload rejects an oversized file and a `.txt` file
- [ ] Upload rejects an empty or one-word expected answer
- [ ] Analysis completes and shows score, similarity, fluency and grade
- [ ] Dashboard renders the real waveform and component bars
- [ ] PDF downloads and opens
- [ ] History lists the run, exports CSV and deletes a record
- [ ] Re-running the same audio reuses the cached transcript (see *Analysis notes*)

---

## 💸 12. FREE-TIER / $0 COST SETUP

**Total expected operating cost: ₹0 / $0.** The application requires no paid service, no
subscription, no trial credits and no credit card. It can be run entirely offline after the first
model download.

### External services used

#### 1. OpenAI Whisper — speech-to-text

1. **Used for:** converting the uploaded recording into text.
2. **Why it is free:** Whisper is MIT-licensed open-source software. It runs **locally** on your
   own CPU. This is *not* the paid OpenAI audio API — no account, key or network call is involved.
3. **Free-tier limits:** none. The only cost is local CPU time and ~140 MB of disk for the `base`
   model (downloaded once from a public URL).
4. **How the app is optimised:** the model is loaded lazily and memoised per process; transcripts
   are cached in SQLite by the SHA-256 of the audio, so re-analysing the same recording never
   re-runs the model; uploads are capped at 25 MB / 5 minutes; `fp16` is disabled for correct CPU
   inference.
5. **When the limit is reached:** not applicable. If the machine is slow, set `WHISPER_MODEL=tiny`
   in `.env` for roughly 3× faster transcription.

#### 2. Sentence-Transformers `all-MiniLM-L6-v2` — semantic similarity

1. **Used for:** measuring how closely the spoken answer's *meaning* matches the expected answer.
2. **Why it is free:** Apache-2.0 open-source model, ~90 MB, downloaded once from Hugging Face's
   free public model hub and then run locally. No account or token is required.
3. **Free-tier limits:** none for inference. Hugging Face applies generous anonymous download rate
   limits, which affect only the one-time download.
4. **How the app is optimised:** the model is loaded once per process and memoised; only two short
   texts are encoded per evaluation; after the first download everything works offline.
5. **When the limit is reached:** if the model cannot be downloaded or loaded, the app
   automatically falls back to the built-in TF-IDF + keyword-recall metric in `scoring.py`, which
   needs no model at all. A note is shown in *Analysis notes*; the app never fails.

#### 3. Google Gemini API — **optional** qualitative feedback

1. **Used for:** examiner-style strengths, weaknesses and suggestions layered on top of the locally
   computed scores. **The app is fully functional without it.**
2. **Why it is free:** Google AI Studio provides a permanently free tier for Gemini Flash models.
   A key is created at https://aistudio.google.com/apikey with a Google account only — **no credit
   card and no billing account are required**, and the free tier is not a time-limited trial.
3. **Free-tier limits:** Google publishes per-model request-per-minute, request-per-day and
   token-per-minute ceilings for Flash models (historically in the region of ~10–15 RPM and a few
   hundred requests per day). Google may adjust these, so check the current figures at
   https://ai.google.dev/gemini-api/docs/rate-limits. This project's defaults sit deliberately
   below the published values.
4. **How the app is optimised:**
   - Every prompt is hashed and cached in the `ai_cache` table for 30 days — repeating an
     evaluation costs **zero** API calls.
   - A client-side limiter (`api_usage` table) enforces `GEMINI_CALLS_PER_MINUTE` (default 10) and
     `GEMINI_CALLS_PER_DAY` (default 180) and refuses to call the API beyond them.
   - Exactly **one** API call is made per new evaluation — never per page view, never on a timer.
   - There is no polling, no background job and no streaming.
   - The prompt is compact: the expected answer, the transcript and a short list of pre-computed
     numeric signals.
   - All scoring is computed locally, so Gemini is never on the critical path.
5. **When the limit is reached:** a quota or rate-limit error is caught, the app logs it, shows a
   note, and silently falls back to the offline rule-based evaluator. **The evaluation still
   completes with a real score, a real grade and real feedback.** Setting `OFFLINE_MODE=true`
   disables Gemini entirely.

#### 4. SQLite — storage

1. **Used for:** evaluation history, transcript cache, AI response cache, API usage accounting.
2. **Why it is free:** SQLite is public-domain software bundled with Python. It is a local file —
   there is no database server, no hosting and no account.
3. **Free-tier limits:** none. Limited only by local disk (a typical evaluation row is a few KB).
4. **How the app is optimised:** short-lived connections, WAL journaling, indexes on
   `created_at` and API usage, pagination on the History page, `api_usage` rows older than two days
   are pruned automatically, and waveform arrays are deliberately **not** persisted.
5. **When the limit is reached:** not applicable. Use **Clear all history** or delete the `.db`
   file if it ever grows large.

#### 5. Streamlit — web interface

1. **Used for:** the entire user interface.
2. **Why it is free:** Apache-2.0 open source, run locally with `streamlit run`.
3. **Free-tier limits:** none for local use.
4. **How the app is optimised:** heavy resources are cached with `@st.cache_resource`; models load
   lazily so browsing the Home page costs nothing; `maxUploadSize` is capped at 30 MB.
5. **When the limit is reached:** not applicable.

### Optional free deployment

If you want to host it, **Streamlit Community Cloud** (https://share.streamlit.io) is free with a
GitHub account and no credit card. Add `GEMINI_API_KEY` under *App → Settings → Secrets* rather
than committing it. Be aware of the free-tier resource ceiling: use `WHISPER_MODEL=tiny`, since
`base` and larger models may exceed the container's memory. The app is also perfectly usable purely
on localhost, which is the recommended and fully free option.

### Free-tier limitations to be aware of

- The first run downloads ~1.5–2.5 GB of Python packages (mostly PyTorch) plus ~230 MB of models.
- Local transcription speed depends on your CPU: roughly 5–20 s for a 1-minute clip with the `base`
  model, faster with `tiny`.
- Without ffmpeg only `.wav` uploads can be decoded.
- Without a Gemini key, feedback is rule-based rather than free-form prose — it is still specific
  and derived from the real measured signals, but it is not natural-language examiner commentary.
- Gemini free-tier quotas are set by Google and can change; the app degrades gracefully when they
  are hit.

---

## 🩺 13. Common Errors and Fixes

| Symptom | Cause | Fix |
|---|---|---|
| `KeyError: 'GEMINI_API_KEY'` on startup | Older version read secrets at import time | Fixed — the key is now optional and resolved lazily |
| Sidebar shows **ffmpeg: missing** | ffmpeg not on PATH | Install ffmpeg (section 6) and restart, or upload `.wav` files |
| `FileNotFoundError: ... ffmpeg` during analysis | Same as above | Same as above |
| *"openai-whisper is not installed"* | Dependencies not installed | `pip install -r requirements.txt` with the venv active |
| First analysis hangs for minutes | Models downloading | Wait for the one-time download; watch the terminal. Use `WHISPER_MODEL=tiny` to reduce it |
| *"No speech was detected in the recording"* | Silent or corrupted audio | Re-record with a working microphone |
| *"The Gemini free-tier quota is currently exhausted"* | Daily/minute quota hit | Nothing to do — the offline evaluator takes over. Retry later or set `OFFLINE_MODE=true` |
| *"Local rate limit reached"* | Self-imposed cap hit | Raise `GEMINI_CALLS_PER_MINUTE` / `GEMINI_CALLS_PER_DAY` in `.env`, or wait |
| Note: *"Sentence-transformer embeddings were unavailable"* | Model could not be downloaded/loaded | Check connectivity for the one-time download; the lexical metric is used meanwhile |
| Note: *"Audio signal analysis was unavailable"* | Librosa could not decode the file | Install ffmpeg, or upload a PCM `.wav` |
| `torch` install fails / too large | Default wheel includes CUDA | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| `sqlite3.OperationalError: database is locked` | Several app copies writing at once | Close the extra instances; connections use WAL and a 15 s timeout |
| `AxiosError: Request failed with status code 403` on upload | Streamlit XSRF vs a proxy | Access the app directly at `localhost:8501` |
| Port 8501 already in use | Another Streamlit instance | `streamlit run "Source Code/app.py" --server.port 8502` |

Set `LOG_LEVEL=DEBUG` in `.env` for verbose developer logs. Users always see friendly messages;
raw tracebacks go to the terminal only.

---

## 🚀 14. Production / Deployment Notes

- **Secrets:** never commit `.env` or `.streamlit/secrets.toml` (both are git-ignored). On hosted
  platforms use the platform's secret manager.
- **Streamlit config:** `showErrorDetails = false` keeps internal tracebacks out of the browser;
  `enableXsrfProtection = true` is on; `gatherUsageStats = false` disables telemetry.
- **Resources:** Whisper is CPU-bound. For multiple concurrent users, use `WHISPER_MODEL=tiny` or a
  machine with more cores.
- **Database:** SQLite suits single-instance deployments. For multi-instance hosting, point
  `DATABASE_PATH` at a shared volume or migrate `database.py` to a networked engine — the data
  layer is isolated behind the `Database` class for exactly this reason.
- **Persistence:** on ephemeral hosts the SQLite file is lost on restart. Mount a volume if history
  must survive.
- **Privacy:** audio is written to a temporary file, processed and deleted immediately in a
  `finally` block. Audio is never uploaded anywhere. Only the transcript and the expected answer are
  sent to Gemini, and only when a key is configured.
- **Scaling costs:** the architecture stays at $0 because the expensive parts (ASR, embeddings) run
  locally. Keep it that way — do not move transcription to a paid API.

---

## 📜 15. Academic Information

- **Project title:** Voice-Based Concept Understanding Analyser
- **Domain:** Artificial Intelligence · Machine Learning · Speech Processing · NLP
- **Institution:** Aditya College of Engineering and Technology (ACET), Surampalem
- **Department:** Computer Science and Engineering (CSE)
- **Programme:** APSCHE (Smart Bridge) Internship 2026 — Google Cloud Generative AI

**Team**

1. Geetha Pati
2. Vijay Kumar Nangana
3. Pavani Lakshmi Gonthina
4. Tejaswini Veera Yenugula
5. Poojitha Pravallika Pedapatruni

---

## 🔮 16. Future Enhancements

- Multi-language speech support (Whisper already detects language)
- Real-time / streaming voice evaluation
- Emotion and sentiment analysis
- User authentication and teacher/student roles
- Per-student longitudinal analytics
- Noise reduction before transcription
- Mobile application front end

---

## 📄 17. License & Acknowledgements

Developed for educational and academic purposes.

Thanks to APSCHE (Smart Bridge), Aditya College of Engineering and Technology, Streamlit, OpenAI
Whisper, Sentence-Transformers, Google Gemini, Librosa, ReportLab, FFmpeg and the wider Python
open-source community.

---

### Thank you for visiting our repository!
