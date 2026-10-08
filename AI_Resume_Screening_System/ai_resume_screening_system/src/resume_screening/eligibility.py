from __future__ import annotations

import re

from .extraction import AI_TERMS, deterministic_resume_analysis
from .models import EligibilityResult, ResumeData


def evaluate_eligibility(resume: ResumeData) -> EligibilityResult:
    analysis = deterministic_resume_analysis(resume)
    python_evidence = analysis.python_evidence
    ai_evidence = analysis.ai_evidence

    reasons: list[str] = []
    if not python_evidence:
        reasons.append("No evidence of Python stack")
    if not ai_evidence:
        reasons.append("No AI/agentic project evidence")

    matched = []
    blob = " ".join(resume.skills + [resume.sections.get("projects", ""), resume.sections.get("experience", "")]).casefold()
    for skill in resume.skills:
        if skill.casefold() in blob:
            matched.append(skill)
    for keyword in sorted(AI_TERMS):
        if keyword in blob and keyword not in {s.casefold() for s in matched}:
            matched.append(keyword)
    # Only expose recognisable skills/technologies rather than every token.
    matched = _dedupe(matched)[:20]

    return EligibilityResult(
        eligible=not reasons,
        rejection_reasons=reasons,
        matched_skills=matched,
        python_evidence=python_evidence,
        ai_evidence=ai_evidence,
    )


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        k = re.sub(r"\s+", " ", value).strip().casefold()
        if k and k not in seen:
            seen.add(k)
            result.append(re.sub(r"\s+", " ", value).strip())
    return result
