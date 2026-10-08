# AI Resume Screening & Ranking System

An SDE-intern coding-assignment implementation for screening ~50 resumes with deterministic hard eligibility, explainable 100-point ranking, optional LLM enrichment, lightweight public GitHub enrichment, and failure-isolated batch processing.

The design intentionally stays small: CLI only, no frontend, no database, no vector database, no authentication, and no SaaS layer. The assignment explicitly prioritizes correctness, explainability, reliability, and testability over UI polish.

## Architecture

```text
resume folder
    |
    v
parsers.py  ---> normalized ResumeData
    |
    +----> eligibility.py  (deterministic hard filter)
    |             |
    |             +---- reject with explicit reasons
    |
    +----> llm.py --------> optional structured extraction / project analysis
    |
    +----> github.py -----> optional public GitHub enrichment + per-run cache
    |
    v
scoring.py  ---> evidence + penalties + 100-point score
    |
    v
pipeline.py ---> deterministic ranking + batch summary
    |
    v
results.json
```

## Setup

Python 3.11+ is recommended.

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env   # Windows
# cp .env.example .env   # macOS/Linux
```

No API key is required for the default deterministic/mock mode.

## Run

```bash
python main.py --input ./resumes --output ./output/results.json
```

The supplied dataset is already copied into `./resumes`. The sample generated output is in `./output/results.json`. The bundled sample was generated with GitHub enrichment disabled because the build environment has no outbound GitHub API access; enabling it at runtime populates the GitHub score/status.

## Hard eligibility

Eligibility is deterministic and happens before ranking. Both requirements must pass:

1. Genuine Python evidence in skills, project/work technology, internship/work technology, or implementation language.
2. Meaningful AI/LLM/RAG/agentic evidence.

A Java/React-only profile is rejected when Python evidence is absent. AI framework names in a skills list can help satisfy the minimum AI signal, but they do not receive high project-depth credit without project/work implementation evidence. This mirrors the assignment's hard-filter and project-quality requirements.

## Scoring

The implementation uses the required baseline weights:

| Category | Max |
|---|---:|
| AI / Agentic / RAG Project Depth | 40 |
| Python & Backend Engineering | 30 |
| Cloud / Deployment / Full Stack | 15 |
| GitHub Activity | 10 |
| Engineering Depth Signals | 5 |
| **Total** | **100** |

AI project scoring rewards retrieval/RAG, agents/orchestration, state/memory, evaluation, data processing/product logic, and backend integration. Thin LLM/API wrappers can receive a deterministic `-10` penalty with an explicit reason. The assignment calls for approximately 5–15 point shallow-project penalties.

Python/backend scoring prefers project/work evidence over skills-only keywords. Cloud/full-stack points require evidence from projects/work, not just a standalone skills keyword.

## GitHub enrichment

When a GitHub profile is present, the system extracts the username and calls public GitHub REST endpoints for the user profile, recent public events, and recently updated repositories. The score is capped at 10: up to 5 for recent activity and up to 5 for maintained/relevant repositories. GitHub never affects hard eligibility. The implementation caches the username result for the current run and records `success`, `rate_limited`, `api_error`, `invalid_url`, `private_or_unavailable`, or `not_provided` instead of crashing the batch.

A GitHub token can be supplied as `GITHUB_TOKEN`. GitHub supports authenticated requests for a higher primary rate limit; the implementation therefore keeps requests serial and handles 403/429 responses without failing the batch.

## LLM usage

`llm.py` defines a provider adapter with:

- `MockProvider`: deterministic, reproducible, and used by default.
- `OpenAIProvider`: optional structured JSON extraction/project-quality analysis behind the same interface.

Hard eligibility is never delegated to the LLM. LLM failures fall back to deterministic extraction/scoring for the affected candidate and increment `llm_failures` in the batch summary.

Set:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your-key
MODEL_NAME=gpt-4o-mini
```

For tests and offline runs, keep `LLM_PROVIDER=mock`.

## Batch reliability and duplicates

Every supported resume is processed independently. A malformed PDF, DOCX, or TXT file becomes a failure record and the rest of the batch continues. Exact duplicate files are detected by SHA-256 content hash and are not processed twice. Unsupported files are ignored.

PDF is required; DOCX and TXT are also supported as a bonus.

## Output

`results.json` contains:

- ranked eligible candidates
- rejected candidates with explicit rejection reasons
- score breakdown and evidence
- project summary, strengths, concerns, and penalties
- LLM/GitHub status
- batch summary including parsing failures, duplicates, LLM failures, GitHub outcomes, and processing duration

Tie-breaks are deterministic: total score, AI depth, Python/backend score, engineering depth, candidate name alphabetically.

## Tests

```bash
pytest
```

The tests cover:

- Python + AI eligibility
- Java/React-only rejection
- Python without AI rejection
- both missing eligibility reasons
- shallow-vs-strong AI project scoring
- 100-point caps
- batch continuation after an unreadable file

The assignment explicitly asks for tests around eligibility, scoring, batch resilience, and failure isolation.

## Design Decisions

**Deterministic filter first.** This prevents a persuasive LLM explanation from turning an irrelevant candidate into an eligible one.

**Evidence before keywords.** Skills lists are useful for detection, but project/work descriptions receive more scoring credit.

**Simple additive scoring.** The five required categories map directly to the assignment weights. This makes score decisions easy to audit.

**Graceful degradation.** LLM and GitHub are enrichment layers; neither is allowed to crash the batch.

**Serial GitHub calls + in-memory cache.** With ~50 resumes, a simple bounded request pattern is easier to reason about and less likely to trigger secondary limits than aggressive concurrency.

## Trade-offs

- Parsing is text-first rather than OCR-heavy; image-only/scanned resumes may produce little text.
- Project section segmentation uses conservative heuristics because resume layouts vary widely.
- GitHub activity reflects public API-visible activity and is deliberately a small positive signal, not a quality verdict.
- The default deterministic scorer is intentionally explainable; an LLM is optional rather than mandatory.

## If I Had More Time

1. Add a small evaluation fixture set with hand-labeled expected eligibility and score ranges.
2. Add optional persistent GitHub/LLM cache keyed by username/content hash.
3. Add stronger section-aware parsing and OCR fallback for scanned PDFs.
4. Add an optional FastAPI wrapper after the CLI pipeline is fully stabilized.

## Deliverables included

- `main.py`
- `src/resume_screening/*` modular implementation
- `tests/*`
- `requirements.txt`
- `.env.example`
- `README.md`
- `resumes/*.pdf` supplied sample set
- `output/results.json` generated from the supplied set
