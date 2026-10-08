from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Project(BaseModel):
    name: str
    description: str = ""
    technologies: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class ResumeData(BaseModel):
    file_name: str
    file_hash: str
    candidate_name: str
    email: str | None = None
    phone: str | None = None
    github_url: str | None = None
    skills: list[str] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    experience: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    sections: dict[str, str] = Field(default_factory=dict)
    raw_text: str


class EligibilityResult(BaseModel):
    eligible: bool
    rejection_reasons: list[str] = Field(default_factory=list)
    matched_skills: list[str] = Field(default_factory=list)
    python_evidence: list[str] = Field(default_factory=list)
    ai_evidence: list[str] = Field(default_factory=list)


class ResumeAnalysis(BaseModel):
    candidate_name: str | None = None
    skills: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    python_evidence: list[str] = Field(default_factory=list)
    ai_evidence: list[str] = Field(default_factory=list)
    backend_evidence: list[str] = Field(default_factory=list)
    cloud_evidence: list[str] = Field(default_factory=list)
    engineering_evidence: list[str] = Field(default_factory=list)
    github_url: str | None = None


class ProjectQualityAnalysis(BaseModel):
    ai_depth: int = Field(ge=0, le=40)
    python_backend_depth: int = Field(ge=0, le=30)
    cloud_depth: int = Field(ge=0, le=15)
    engineering_depth: int = Field(ge=0, le=5)
    is_shallow_wrapper: bool = False
    evidence: list[str] = Field(default_factory=list)
    reasoning: list[str] = Field(default_factory=list)


GithubStatus = Literal[
    "not_provided",
    "success",
    "rate_limited",
    "api_error",
    "invalid_url",
    "private_or_unavailable",
]


class GithubResult(BaseModel):
    status: GithubStatus
    username: str | None = None
    score: int = Field(default=0, ge=0, le=10)
    summary: str = ""
    evidence: list[str] = Field(default_factory=list)
    recent_activity_score: int = Field(default=0, ge=0, le=5)
    maintained_repo_score: int = Field(default=0, ge=0, le=5)


class Penalty(BaseModel):
    type: str
    points: int
    reason: str


class ScoreBreakdown(BaseModel):
    ai_project_depth: int = 0
    python_backend: int = 0
    cloud_fullstack: int = 0
    github: int = 0
    engineering_depth: int = 0

    @property
    def total(self) -> int:
        return (
            self.ai_project_depth
            + self.python_backend
            + self.cloud_fullstack
            + self.github
            + self.engineering_depth
        )


class CandidateResult(BaseModel):
    rank: int | None = None
    candidate_name: str
    source_file: str
    eligible: bool
    total_score: int
    score_breakdown: ScoreBreakdown
    matched_skills: list[str] = Field(default_factory=list)
    project_summary: str = ""
    github_summary: str = ""
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    rejection_reasons: list[str] = Field(default_factory=list)
    evidence: dict[str, list[str]] = Field(default_factory=dict)
    penalties: list[Penalty] = Field(default_factory=list)
    llm_status: str = "not_used"
    github_status: GithubStatus = "not_provided"
    file_duplicate: bool = False


class BatchSummary(BaseModel):
    total_resumes: int
    successfully_parsed: int
    eligible: int
    rejected: int
    failed_or_unreadable: int
    duplicate_files: int
    llm_failures: int
    github_successes: int
    github_failures: int
    github_unavailable: int
    processing_duration_seconds: float


class ScreeningOutput(BaseModel):
    generated_at_utc: str
    batch_summary: BatchSummary
    candidates: list[CandidateResult]
