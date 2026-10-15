# Voice-Based Concept Understanding Analyser (VBCUA)

VBCUA is a Streamlit application for evaluating spoken explanations. It combines local speech transcription, semantic comparison, audio-delivery metrics, feedback, and a history of previous evaluations. It also includes a local PDF retrieval pipeline, Gemini-assisted concept analysis, Tutor Mode, and a browser-based Real-time Coach.

## What the application does

A student can provide a recording and an expected answer, or use retrieved study material as context. VBCUA transcribes the recording, evaluates the explanation and delivery, then presents a score, grade, feedback, and supporting metrics.

### Main features

- **Audio evaluation:** upload supported audio, validate it, and analyse the recording.
- **Local speech-to-text:** OpenAI Whisper runs locally; the final evaluation uses the Whisper transcript.
- **Semantic comparison:** Sentence-Transformers compares the transcript with the reference answer. A local TF-IDF/keyword-based fallback is available if embeddings cannot be used.
- **Delivery analysis:** Librosa-based features include duration, pauses, silence ratio, and energy consistency. Transcript-based metrics include filler words and speaking pace.
- **Weighted score and grade:** the default overall score combines semantic similarity (70%) and fluency (30%). The weights can be configured.
- **Feedback:** optional Gemini feedback provides examiner-style strengths, weaknesses, and suggestions. A rule-based local evaluator is available when Gemini is not configured or cannot be used.
- **PDF-based RAG:** extract text from study PDFs, split it into overlapping chunks, embed the chunks locally, store them in SQLite, and retrieve relevant passages for a topic or question.
- **Concept-level analysis:** when Gemini is enabled and retrieved PDF context is supplied, Gemini can extract a grounded concept map. VBCUA then compares the student's transcript with each concept locally.
- **Tutor Mode:** asks targeted follow-up questions about concepts that appear weak, using the available concept map and study context. Local scoring/question fallbacks are available when Gemini is unavailable.
- **Real-time Mode:** provides a browser-side live speech preview and approximate live metrics. The submitted recording is then evaluated using Whisper; browser recognition is not the authoritative transcript.
- **Dashboard and history:** review saved evaluations and metrics, browse previous results, and export available history data.
- **PDF reports:** generate a report containing evaluation results and feedback.
- **Local persistence and caching:** SQLite stores evaluation history, transcript/AI caches, API usage information, and the PDF document/chunk index.

## Architecture and workflow

### Spoken-answer evaluation

```text
Audio recording + reference answer (or retrieved PDF context)
                         |
                         v
               Input validation
                         |
             +-----------+-----------+
             |                       |
             v                       v
     Audio feature analysis     Local Whisper
     duration / pauses /        transcription
     energy / waveform               |
             |                       v
             |                 Transcript text
             |                       |
             +-----------+-----------+
                         |
                         v
               Semantic comparison
                         |
                         v
                Fluency evaluation
                         |
                         v
          Score, grade, and feedback
          (local rules or optional Gemini)
                         |
                         v
             SQLite / results / dashboard
                         |
                         v
                    PDF report
```

### PDF retrieval (RAG)

```text
Study PDF
   |
   v
Extract page text with pypdf
   |
   v
Create overlapping text chunks
(default: 900 words, 150-word overlap)
   |
   v
Generate local Sentence-Transformer embeddings
   |
   v
Store document metadata, page references,
chunk text, and embeddings in SQLite
   |
   v
Embed the user's topic/question and retrieve
the most relevant chunks (default top 5)
   |
   v
Display retrieved passages and source pages
   |
   +--> Optional Gemini concept extraction
        using retrieved context
                 |
                 v
        Local transcript-to-concept scoring
```

PDF ingestion and retrieval are local and do not require Gemini. Gemini is used for enhanced feedback and for grounded concept extraction when enabled. Text extraction does not perform OCR, so scanned or image-only PDFs may need OCR before they can be indexed.

## Technology stack

| Area | Technology |
|---|---|
| Language | Python |
| User interface | Streamlit |
| Speech transcription | OpenAI Whisper, running locally |
| Semantic embeddings | Sentence-Transformers (`sentence-transformers/all-MiniLM-L6-v2` by default) |
| Lexical fallback | Local TF-IDF and keyword-recall logic |
| Audio analysis | Librosa, NumPy, SoundFile |
| PDF text extraction | pypdf |
| Optional generative AI | Google Gemini through the `google-genai` SDK |
| Persistence and caches | SQLite |
| Charts | Matplotlib |
| PDF reports | ReportLab |
| Tests | pytest |

The application is a Streamlit app; the supplied source does not define a separate FastAPI service or a separate backend server.

## Project structure

```text
VBCUA/
├── README.md
├── requirements.txt
├── run.bat
├── test_phase2_concepts.py
├── test_phase2_gemini_resilience.py
├── test_rag_phase1.py
└── Source Code/
    ├── app.py                 # Streamlit entry point and navigation
    ├── upload.py              # Audio input and optional PDF retrieval UI
    ├── analysis.py            # Evaluation pipeline and result model
    ├── transcription.py       # Local Whisper transcription
    ├── audio_features.py     # Audio feature extraction
    ├── semantics.py           # Semantic similarity and embedding model
    ├── scoring.py             # Fluency, weighted score, grade, local feedback
    ├── rag.py                 # PDF extraction, chunking, embedding, retrieval
    ├── tutor.py               # Tutor Mode follow-up logic
    ├── realtime_mode.py       # Browser live coach and submitted-recording flow
    ├── gemini_client.py       # Optional Gemini calls, parsing, caching and limits
    ├── database.py            # SQLite schema and persistence
    ├── dashboard.py           # Evaluation dashboard
    ├── results.py             # History and result actions
    ├── report.py              # PDF report generation
    ├── ui.py                  # Shared UI elements and theme
    ├── config.py              # Environment/secrets-based configuration
    └── .streamlit/
        ├── config.toml
        └── secrets.toml.example
```

The SQLite database is created locally under `Source Code/data/` by default. It is runtime data, not source code.

## Requirements

- Python 3.11 is recommended. The Windows launcher attempts to create a Python 3.11 virtual environment.
- `pip`
- FFmpeg on `PATH` for decoding formats such as MP3, M4A, OGG, and other formats that require it. WAV is the simplest format to test first.
- Internet access for installing Python packages and downloading the Whisper and embedding models on first use.
- A Google Gemini API key only if you want Gemini-enhanced feedback and Gemini-based concept extraction. Core local evaluation and local PDF retrieval can run without it.

Model downloads and local inference require disk space and can be CPU-intensive, particularly on the first run.

## Installation and run

### Windows

1. Download or clone the repository.
2. Open a terminal in the repository root.
3. Install the dependencies and launch the app:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run "Source Code/app.py"
```

Alternatively, run `run.bat` from the repository root. It creates a Python 3.11 virtual environment if needed, installs requirements, and starts Streamlit.

If PowerShell blocks virtual-environment activation, you can run the commands from Command Prompt using `venv\Scripts\activate.bat`, or invoke `venv\Scripts\python.exe` directly.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run "Source Code/app.py"
```

When Streamlit starts, open the local URL shown in the terminal (usually `http://localhost:8501`).

## Configuration

Configuration is optional. Settings can be supplied as environment variables, through a local `.env` file in `Source Code/`, or through Streamlit secrets. Do not commit API keys or secret files.

Create `Source Code/.env` if you want to configure the application:

```dotenv
# Optional: enables Gemini feedback and grounded concept extraction
GEMINI_API_KEY=

# Optional: comma-separated model preference list; GEMINI_MODEL is also accepted
GEMINI_MODELS=

# Local model choices
WHISPER_MODEL=base
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Local SQLite database
DATABASE_PATH=data/results.db

# Input limits
MAX_AUDIO_MB=25
MAX_AUDIO_SECONDS=300

# Overall score weights (normalised by the application)
SEMANTIC_WEIGHT=0.7
FLUENCY_WEIGHT=0.3

# Optional behaviour and limits
OFFLINE_MODE=false
GEMINI_CALLS_PER_MINUTE=10
GEMINI_CALLS_PER_DAY=180
AI_CACHE_TTL_HOURS=720
HISTORY_PAGE_SIZE=25
```

Leave `GEMINI_API_KEY` empty to use local scoring and rule-based feedback. Set `OFFLINE_MODE=true` to disable Gemini even if a key is present. Provider model availability and API quotas can change over time.

## How scoring works

The default final score is:

```text
Final score = (semantic similarity × 0.70) + (fluency score × 0.30)
```

The application normalises configured semantic and fluency weights before combining them.

Fluency combines the available delivery signals, with missing signals omitted and the remaining weights re-normalised:

| Signal | Default component weight |
|---|---:|
| Filler-word control | 35% |
| Speaking pace | 25% |
| Pause ratio | 25% |
| Energy consistency | 15% |

The pace heuristic gives its best score in the approximate range of 110–160 words per minute. These are application heuristics, not a formal assessment of a student's ability.

| Score | Grade |
|---:|:---|
| 90–100 | A+ |
| 80–89.99 | A |
| 70–79.99 | B |
| 60–69.99 | C |
| 50–59.99 | D |
| Below 50 | F |

Gemini feedback does not replace the locally calculated score. When Gemini is unavailable, the app can generate feedback from measured semantic, keyword, and delivery signals.

## Using PDF RAG

1. Open **Upload Audio**.
2. Upload a study PDF and index it.
3. Select the indexed document.
4. Enter a topic or question and retrieve relevant passages.
5. Review the retrieved excerpts and source page references.
6. Provide a recording and a reference answer, or use retrieved context as the basis for analysis.
7. Run **AI Analysis**. With Gemini enabled, the app can extract a concept map from the retrieved context and score the transcript against individual concepts.
8. If a concept map is available, open **Tutor Mode** for targeted follow-up questions.

Retrieval is based on local embeddings and cosine similarity over stored chunks. The default chunk size is 900 words with 150 words of overlap, and the default retrieval count is five chunks. These are defaults in the source and may be changed in code.

## Real-time Mode notes

Real-time Mode uses browser speech recognition for a live preview and approximate metrics such as elapsed time, word count, speaking pace, and filler count. Browser support varies. The submitted recording is analysed by the normal Whisper-based pipeline; the browser preview is not treated as the authoritative transcript.

## Data, privacy, and API use

- Audio transcription and embedding inference run locally after required models are installed.
- PDF extraction, chunking, embedding, and retrieval are local.
- When Gemini is enabled, relevant text inputs (such as the transcript, reference answer, or retrieved study context) may be sent to Google's Gemini API for the requested AI operation. Do not use Gemini-enabled features with material you are not permitted to share.
- SQLite stores evaluation records, transcript and AI caches, API usage counters, and indexed PDF chunks/embeddings.
- Keep `.env` and Streamlit secret files out of version control. The provided `.gitignore` excludes these files and local database data.

## Tests

The repository includes these test files:

- `test_rag_phase1.py`
- `test_phase2_concepts.py`
- `test_phase2_gemini_resilience.py`

Run the available tests from the repository root:

```bash
python -m pytest -q
```

The README does not claim a fixed passing-test count; results depend on the current source, installed dependencies, and test environment.

## Troubleshooting

| Problem | What to check |
|---|---|
| Whisper or model import fails | Activate the intended virtual environment and install `requirements.txt`. |
| MP3/M4A/OGG decoding fails | Install FFmpeg, add its `bin` directory to `PATH`, and restart the terminal. Try WAV to isolate decoding issues. |
| First evaluation is slow | Whisper and the embedding model may be downloading or loading for the first time. Try `WHISPER_MODEL=tiny` on a slower machine. |
| Embedding model is unavailable | Check internet access for the initial download. The semantic scorer has a lexical fallback. |
| Gemini quota or request fails | Check the API key and current provider limits, or set `OFFLINE_MODE=true` to use local feedback. |
| PDF retrieval returns no useful passages | Confirm the PDF contains selectable text, try a more specific query, and check the retrieved source excerpts. Scanned PDFs require OCR before ingestion. |
| Browser live recognition is unavailable | Use a supported browser or use the standard audio-upload workflow. The final transcript is generated by Whisper. |

## Project information

- **Project:** Voice-Based Concept Understanding Analyser (VBCUA)
- **Domain:** Artificial Intelligence, speech processing, and natural-language processing
- **Programme noted in source:** APSCHE Internship 2026 — Google Cloud Generative AI

## Limitations

- PDF extraction is text-based and does not include OCR for scanned pages.
- Gemini-dependent concept extraction and enhanced feedback require a valid API key and available provider quota.
- Speech pace and delivery metrics are heuristics; they should be interpreted as learning feedback, not as a definitive measure of understanding.
- Real-time browser recognition is a preview; the final analysis uses Whisper on the submitted recording.
