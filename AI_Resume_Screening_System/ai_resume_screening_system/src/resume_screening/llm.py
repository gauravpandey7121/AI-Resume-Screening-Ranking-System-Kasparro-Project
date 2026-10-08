from __future__ import annotations

import json
from abc import ABC, abstractmethod

from .models import ProjectQualityAnalysis, ResumeAnalysis, ResumeData
from .extraction import deterministic_resume_analysis
from .scoring import score_resume
from .config import Settings


class LLMProvider(ABC):
    @abstractmethod
    def analyze_resume(self, resume: ResumeData) -> ResumeAnalysis:
        raise NotImplementedError

    @abstractmethod
    def analyze_projects(self, resume: ResumeData, analysis: ResumeAnalysis) -> ProjectQualityAnalysis:
        raise NotImplementedError


class MockProvider(LLMProvider):
    """Deterministic provider used by default for reproducible local/test runs."""

    def analyze_resume(self, resume: ResumeData) -> ResumeAnalysis:
        return deterministic_resume_analysis(resume)

    def analyze_projects(self, resume: ResumeData, analysis: ResumeAnalysis) -> ProjectQualityAnalysis:
        result = score_resume(resume, analysis)
        return ProjectQualityAnalysis(
            ai_depth=result.breakdown.ai_project_depth,
            python_backend_depth=result.breakdown.python_backend,
            cloud_depth=result.breakdown.cloud_fullstack,
            engineering_depth=result.breakdown.engineering_depth,
            is_shallow_wrapper=any(p.type == "shallow_ai_project" for p in result.penalties),
            evidence=result.evidence.get("ai", [])[:6],
            reasoning=result.strengths[:4] + result.concerns[:4],
        )


class OpenAIProvider(LLMProvider):
    """Small provider adapter. The rest of the pipeline never imports OpenAI directly."""

    def __init__(self, settings: Settings):
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        from openai import OpenAI

        self.client = OpenAI(api_key=settings.openai_api_key, timeout=settings.llm_timeout_seconds)
        self.model = settings.model_name

    def _json_call(self, system: str, user: str) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = response.choices[0].message.content or "{}"
        return json.loads(content)

    def analyze_resume(self, resume: ResumeData) -> ResumeAnalysis:
        system = (
            "You extract structured resume evidence. Do not decide eligibility. "
            "Return only JSON matching the requested fields. Evidence must be short, verbatim-ish, and grounded in the resume."
        )
        user = f"""Analyze this resume. Return JSON with keys: candidate_name, skills, projects, python_evidence, ai_evidence, backend_evidence, cloud_evidence, engineering_evidence, github_url.\n\nRESUME:\n{resume.raw_text[:18000]}"""
        return ResumeAnalysis.model_validate(self._json_call(system, user))

    def analyze_projects(self, resume: ResumeData, analysis: ResumeAnalysis) -> ProjectQualityAnalysis:
        system = (
            "You assess implementation depth, not framework-name presence. Favor project/work evidence, "
            "penalize thin API wrappers and tutorial-style projects. Return only JSON."
        )
        user = f"""Score this resume for the specified rubric. Return JSON with integer fields ai_depth (0-40), python_backend_depth (0-30), cloud_depth (0-15), engineering_depth (0-5), boolean is_shallow_wrapper, evidence array, reasoning array.\n\nResume:\n{resume.raw_text[:18000]}\n\nStructured extraction:\n{analysis.model_dump_json()}"""
        return ProjectQualityAnalysis.model_validate(self._json_call(system, user))


def build_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "openai":
        try:
            return OpenAIProvider(settings)
        except Exception:
            return MockProvider()
    return MockProvider()
