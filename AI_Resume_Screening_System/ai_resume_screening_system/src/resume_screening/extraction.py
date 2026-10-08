from __future__ import annotations

import re
from dataclasses import dataclass

from .models import ResumeAnalysis, ResumeData

AI_TERMS = {
    "langchain", "langgraph", "llamaindex", "google adk", "rag", "retrieval", "embedding",
    "embeddings", "vector search", "vector database", "vector db", "tool calling", "tool-calling",
    "multi-agent", "multi agent", "agentic", "agent orchestration", "llm", "large language model",
    "openai", "gemini", "claude", "hugging face", "machine learning", "deep learning", "tensorflow",
    "pytorch", "computer vision", "nlp", "natural language processing", "speech recognition", "ocr",
    "ai system", "artificial intelligence", "model evaluation", "evals", "evaluation pipeline",
}
PYTHON_TERMS = {"python", "fastapi", "flask", "django", "pyttsx3", "pydantic", "celery", "airflow"}
BACKEND_TERMS = {"fastapi", "flask", "django", "postgresql", "redis", "rest api", "rest apis", "backend", "api", "microservice", "async", "asynchronous", "background task", "queue", "kafka"}
CLOUD_TERMS = {"gcp", "google cloud", "aws", "docker", "deployment", "deployed", "ci/cd", "github actions", "azure", "cloud", "s3", "ec2", "lambda"}
ENGINEERING_TERMS = {"testing", "test", "unit tests", "architecture", "caching", "cache", "queue", "kafka", "concurrency", "async", "error handling", "retry", "circuit breaker", "observability", "monitoring", "rate limiting", "schema", "versioned", "lru", "checkpoint"}


def _sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [re.sub(r"\s+", " ", p).strip(" -•●") for p in pieces if len(p.strip()) >= 18]


def evidence_for_terms(resume: ResumeData, terms: set[str], *, project_or_experience_only: bool = False) -> list[str]:
    pools: list[str] = []
    if project_or_experience_only:
        pools.extend("\n".join(p.evidence) for p in resume.projects)
        pools.extend(resume.experience)
        # Project section text can be lost by imperfect project segmentation.
        pools.append(resume.sections.get("projects", ""))
        pools.append(resume.sections.get("experience", ""))
    else:
        pools.extend("\n".join(resume.projects[i].evidence) for i in range(len(resume.projects)))
        pools.extend(resume.experience)
        pools.append(resume.sections.get("skills", ""))
        pools.append(resume.sections.get("summary", ""))
        pools.append(resume.sections.get("projects", ""))
        pools.append(resume.sections.get("experience", ""))
    evidence: list[str] = []
    for pool in pools:
        for sentence in _sentences(pool):
            low = sentence.casefold()
            if any(term in low for term in terms):
                evidence.append(sentence)
    # Keep the output explainable and bounded.
    seen: set[str] = set()
    result: list[str] = []
    for item in evidence:
        key = item.casefold()
        if key not in seen:
            seen.add(key)
            result.append(item)
        if len(result) >= 8:
            break
    return result


def _has_python_evidence(resume: ResumeData) -> list[str]:
    evidence: list[str] = []
    skill_text = " ".join(resume.skills).casefold()
    if re.search(r"\bpython\b", skill_text):
        evidence.append("Python is listed in the candidate's technical skills.")
    evidence.extend(evidence_for_terms(resume, {"python"}, project_or_experience_only=True)[:6])
    for sentence in _sentences(resume.raw_text):
        low = sentence.casefold()
        if re.search(r"\bpython\b", low) and (
            "python-based" in low or "using python" in low or "written in python" in low
            or "implemented in python" in low or "python developer" in low
            or "programming language: python" in low or "python," in low or "python |" in low
        ):
            evidence.append(sentence)
    return _dedupe(evidence)[:8]


def _has_ai_evidence(resume: ResumeData) -> list[str]:
    project_evidence = evidence_for_terms(resume, AI_TERMS, project_or_experience_only=True)
    for project in resume.projects:
        blob = f"{project.name} {' '.join(project.technologies)} {' '.join(project.evidence)}".casefold()
        if any(term in blob for term in AI_TERMS):
            candidate = project.description or project.name
            if candidate:
                project_evidence.append(candidate)

    # Layout fallback: only count AI terms in sentences that look like implementation evidence.
    action_re = re.compile(r"\b(built|developed|implemented|designed|engineered|created|trained|deployed|integrated|leveraged|orchestrated|automated|optimized|using|uses|utilizing|leverages)\b", re.I)
    for sentence in _sentences(resume.raw_text):
        low = sentence.casefold()
        if action_re.search(sentence) and any(term in low for term in AI_TERMS):
            project_evidence.append(sentence)
        if len(project_evidence) >= 8:
            break

    # Some AI frameworks are strong enough as a minimum signal even when the resume's
    # section structure is poor. Generic words such as 'AI' or 'Machine Learning' alone
    # are not sufficient for this fallback.
    if not project_evidence:
        skill_blob = " ".join(resume.skills + [resume.sections.get("skills", "")]).casefold()
        minimum_framework_terms = {"langchain", "langgraph", "llamaindex", "google adk", "rag", "vector search", "vector database", "tool calling", "multi-agent", "agentic"}
        matched = sorted({term for term in minimum_framework_terms if term in skill_blob})
        if matched:
            project_evidence.append(f"Meaningful AI technology/framework listed in skills: {', '.join(matched[:6])}.")
    return _dedupe(project_evidence)[:8]


def deterministic_resume_analysis(resume: ResumeData) -> ResumeAnalysis:
    python_ev = _has_python_evidence(resume)
    ai_ev = _has_ai_evidence(resume)
    backend_ev = evidence_for_terms(resume, BACKEND_TERMS, project_or_experience_only=True)
    cloud_ev = evidence_for_terms(resume, CLOUD_TERMS, project_or_experience_only=True)
    eng_ev = evidence_for_terms(resume, ENGINEERING_TERMS, project_or_experience_only=True)
    return ResumeAnalysis(
        candidate_name=resume.candidate_name,
        skills=resume.skills,
        projects=[p.name for p in resume.projects if p.name],
        python_evidence=python_ev,
        ai_evidence=ai_ev,
        backend_evidence=backend_ev[:8],
        cloud_evidence=cloud_ev[:8],
        engineering_evidence=eng_ev[:8],
        github_url=resume.github_url,
    )


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        k = v.casefold().strip()
        if k and k not in seen:
            seen.add(k)
            out.append(v.strip())
    return out
