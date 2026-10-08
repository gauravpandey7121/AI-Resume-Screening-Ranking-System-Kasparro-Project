from __future__ import annotations

import re
from dataclasses import dataclass

from .extraction import AI_TERMS, BACKEND_TERMS, CLOUD_TERMS, ENGINEERING_TERMS
from .models import Penalty, ResumeAnalysis, ResumeData, ScoreBreakdown


@dataclass(frozen=True)
class ScoreResult:
    breakdown: ScoreBreakdown
    penalties: list[Penalty]
    evidence: dict[str, list[str]]
    project_summary: str
    strengths: list[str]
    concerns: list[str]


def _blob(resume: ResumeData, include_skills: bool = True) -> str:
    chunks = [resume.sections.get("projects", ""), resume.sections.get("experience", "")]
    if include_skills:
        chunks.append(resume.sections.get("skills", ""))
    if not any(chunk.strip() for chunk in chunks):
        chunks.append(resume.raw_text)
    return "\n".join(chunks).casefold()


def _contains(blob: str, terms: set[str]) -> bool:
    return any(term in blob for term in terms)


def _lines_with_terms(resume: ResumeData, terms: set[str], max_items: int = 8) -> list[str]:
    text = _blob(resume, include_skills=False)
    lines = [re.sub(r"\s+", " ", x).strip(" -•●") for x in text.splitlines() if len(x.strip()) > 12]
    out: list[str] = []
    for line in lines:
        if _contains(line, terms):
            # Preserve original-ish casing by finding the same line from raw sections.
            out.append(line)
        if len(out) >= max_items:
            break
    return _dedupe(out)


def _project_quality(resume: ResumeData) -> tuple[int, list[str], bool, list[Penalty]]:
    candidates: list[tuple[int, str, list[str], bool]] = []
    project_inputs = []
    if resume.projects:
        for project in resume.projects:
            project_inputs.append((project.name or "Project", f"{project.name} {' '.join(project.technologies)} {project.description} {' '.join(project.evidence)}"))
    else:
        # Preserve only implementation-looking AI sentences when section extraction is unavailable.
        raw_candidates = []
        action_re = re.compile(r"\b(built|developed|implemented|designed|engineered|created|trained|deployed|integrated|leveraged|orchestrated|automated|optimized|using|uses|utilizing|leverages)\b", re.I)
        ai_terms = AI_TERMS
        # AI_TERMS are imported at module level; use the sentence text directly here.
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", resume.raw_text):
            clean = re.sub(r"\s+", " ", sentence).strip(" -•●")
            low = clean.casefold()
            if clean and (action_re.search(clean) or any(term in low for term in ("rag", "agent", "deep learning", "machine learning", "llm"))) and any(term in low for term in ai_terms):
                raw_candidates.append(clean)
            if len(raw_candidates) >= 10:
                break
        if raw_candidates:
            project_inputs.append(("Resume AI evidence", " ".join(raw_candidates)))

    for project_name, raw in project_inputs:
        text = raw.casefold()
        score = 0
        evidence: list[str] = []
        if _contains(text, {"rag", "retrieval-augmented", "retrieval augmented", "vector search", "vector database", "vector db", "embedding", "embeddings", "chromadb", "pgvector", "elasticsearch"}):
            score += 10; evidence.append("retrieval/RAG or vector-search evidence")
        if _contains(text, {"agent", "agentic", "langgraph", "multi-agent", "tool calling", "orchestration"}):
            score += 10; evidence.append("agentic workflow, orchestration, or tool-calling evidence")
        if _contains(text, {"state", "memory", "checkpoint", "session", "persistent", "workflow", "multi-step"}):
            score += 5; evidence.append("stateful or multi-step workflow evidence")
        if _contains(text, {"evaluation", "evals", "eval suite", "metrics", "accuracy", "precision", "recall", "benchmark"}):
            score += 5; evidence.append("evaluation/measurement evidence")
        if _contains(text, {"chunk", "document", "pipeline", "index", "data processing", "processing", "business logic", "production"}):
            score += 5; evidence.append("data-processing or product/business-logic evidence")
        if _contains(text, {"machine learning", "deep learning", "tensorflow", "pytorch", "computer vision", "nlp", "neural network", "model training", "model evaluation"}):
            score += 10; evidence.append("custom ML/deep-learning implementation evidence")
        if _contains(text, {"api", "backend", "fastapi", "flask", "django", "database", "postgresql", "redis"}):
            score += 5; evidence.append("backend or integration implementation evidence")
        weak_wrapper = (
            _contains(text, {"openai api", "chatgpt api", "llm api"})
            and not _contains(text, {"rag", "retrieval", "agent", "workflow", "state", "tool", "evaluation", "pipeline", "database"})
        )
        candidates.append((min(score, 40), project_name, evidence, weak_wrapper))

    if not candidates:
        return 0, [], False, []
    candidates.sort(reverse=True)
    best_score, best_name, best_ev, shallow = candidates[0]
    penalties: list[Penalty] = []
    if shallow:
        penalty = -10
        penalties.append(Penalty(
            type="shallow_ai_project",
            points=penalty,
            reason="Project appears to be primarily an LLM/API wrapper with limited workflow, retrieval, state, evaluation, or product-logic evidence.",
        ))
        best_score = max(0, best_score + penalty)
    return best_score, [f"{best_name}: {ev}" for ev in best_ev], shallow, penalties


def score_resume(resume: ResumeData, analysis: ResumeAnalysis, github_score: int = 0, github_evidence: list[str] | None = None) -> ScoreResult:
    blob = _blob(resume, include_skills=False)
    skill_blob = resume.sections.get("skills", "").casefold()
    evidence: dict[str, list[str]] = {"ai": [], "python": [], "backend": [], "cloud": [], "engineering": [], "github": []}
    penalties: list[Penalty] = []

    ai_score, project_evidence, _, project_penalties = _project_quality(resume)
    penalties.extend(project_penalties)
    evidence["ai"].extend(project_evidence)
    evidence["ai"].extend(analysis.ai_evidence[:5])
    evidence["ai"] = _dedupe(evidence["ai"])[:8]

    python_score = 0
    if re.search(r"\bpython\b", blob):
        python_score += 8
        evidence["python"].extend(_lines_with_terms(resume, {"python"}, 4))
    elif re.search(r"\bpython\b", skill_blob):
        python_score += 4
        evidence["python"].append("Python is listed in technical skills, but project/work implementation detail is limited.")

    framework_hits = [term for term in ("fastapi", "flask", "django") if term in blob]
    if framework_hits:
        python_score += min(8, 3 + len(framework_hits) * 2)
        evidence["backend"].extend(_lines_with_terms(resume, set(framework_hits), 3))
    data_hits = [term for term in ("postgresql", "redis", "mysql", "mongodb", "sql") if term in blob]
    if data_hits:
        python_score += min(5, 2 + len(data_hits))
        evidence["backend"].extend(_lines_with_terms(resume, set(data_hits), 3))
    if _contains(blob, {"async", "asynchronous", "background", "concurrency", "celery", "kafka"}):
        python_score += 4
        evidence["backend"].extend(_lines_with_terms(resume, {"async", "asynchronous", "background", "concurrency", "celery", "kafka"}, 2))
    if _contains(blob, {"rest api", "rest apis", "backend", "microservice", "error handling", "retry", "circuit breaker"}):
        python_score += 5
        evidence["backend"].extend(_lines_with_terms(resume, {"rest api", "rest apis", "backend", "microservice", "error handling", "retry", "circuit breaker"}, 3))
    python_score = min(30, python_score)

    cloud_score = 0
    if _contains(blob, {"docker", "deployment", "deployed"}):
        cloud_score += 5; evidence["cloud"].extend(_lines_with_terms(resume, {"docker", "deployment", "deployed"}, 3))
    if _contains(blob, {"gcp", "google cloud", "aws", "azure", "s3", "ec2", "lambda"}):
        cloud_score += 5; evidence["cloud"].extend(_lines_with_terms(resume, {"gcp", "google cloud", "aws", "azure", "s3", "ec2", "lambda"}, 3))
    if _contains(blob, {"react", "next.js", "nextjs"}):
        cloud_score += 3; evidence["cloud"].extend(_lines_with_terms(resume, {"react", "next.js", "nextjs"}, 2))
    if _contains(blob, {"ci/cd", "github actions", "jenkins", "continuous integration", "continuous deployment"}):
        cloud_score += 2; evidence["cloud"].extend(_lines_with_terms(resume, {"ci/cd", "github actions", "jenkins", "continuous integration", "continuous deployment"}, 2))
    cloud_score = min(15, cloud_score)

    eng_score = 0
    engineering_hits = [term for term in ENGINEERING_TERMS if term in blob]
    if engineering_hits:
        eng_score = min(5, max(1, len(engineering_hits) // 2 + 1))
        evidence["engineering"].extend(_lines_with_terms(resume, ENGINEERING_TERMS, 5))

    github_score = max(0, min(10, github_score))
    evidence["github"].extend(github_evidence or [])

    breakdown = ScoreBreakdown(
        ai_project_depth=max(0, min(40, ai_score)),
        python_backend=max(0, min(30, python_score)),
        cloud_fullstack=max(0, min(15, cloud_score)),
        github=github_score,
        engineering_depth=max(0, min(5, eng_score)),
    )

    strengths: list[str] = []
    concerns: list[str] = []
    if breakdown.ai_project_depth >= 30: strengths.append("Strong AI/agentic project depth")
    elif breakdown.ai_project_depth >= 20: strengths.append("Meaningful AI project implementation")
    if breakdown.python_backend >= 22: strengths.append("Strong Python/backend engineering evidence")
    elif breakdown.python_backend >= 14: strengths.append("Solid Python/backend evidence")
    if breakdown.cloud_fullstack >= 10: strengths.append("Hands-on cloud/deployment/full-stack evidence")
    if breakdown.github >= 7: strengths.append("Recent/maintained public GitHub activity")
    if breakdown.engineering_depth >= 4: strengths.append("Non-trivial engineering depth signals")

    if breakdown.ai_project_depth < 20: concerns.append("AI project depth is relatively shallow or implementation detail is limited")
    if breakdown.python_backend < 15: concerns.append("Limited Python/backend implementation evidence")
    if breakdown.cloud_fullstack < 6: concerns.append("Limited cloud/deployment evidence")
    concerns.extend(p.reason for p in penalties)

    summary_project = ""
    if resume.projects:
        best = sorted(resume.projects, key=lambda p: len(p.evidence), reverse=True)[0]
        summary_project = f"{best.name}: {best.description}".strip()
        if len(summary_project) > 300:
            summary_project = summary_project[:297].rstrip() + "..."
    elif analysis.projects:
        summary_project = analysis.projects[0]

    return ScoreResult(
        breakdown=breakdown,
        penalties=penalties,
        evidence={k: _dedupe(v)[:8] for k, v in evidence.items()},
        project_summary=summary_project,
        strengths=_dedupe(strengths)[:6],
        concerns=_dedupe(concerns)[:6],
    )


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = re.sub(r"\s+", " ", value).strip()
        key = clean.casefold()
        if clean and key not in seen:
            seen.add(key)
            out.append(clean)
    return out
