from __future__ import annotations

import hashlib
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings
from .eligibility import evaluate_eligibility
from .github import GithubEnricher
from .llm import LLMProvider, build_provider
from .models import BatchSummary, CandidateResult, ScreeningOutput
from .parsers import SUPPORTED_EXTENSIONS, parse_resume
from .scoring import score_resume


class ScreeningPipeline:
    def __init__(self, settings: Settings, llm_provider: LLMProvider | None = None, github: GithubEnricher | None = None):
        self.settings = settings
        self.llm = llm_provider or build_provider(settings)
        self.github = github or GithubEnricher(settings)

    def run(self, input_dir: Path) -> ScreeningOutput:
        start = time.perf_counter()
        files = sorted(
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        duplicate_hashes: set[str] = set()
        successful: list[tuple[Path, object]] = []
        failed = 0
        failure_records: list[CandidateResult] = []
        llm_failures = 0
        github_successes = 0
        github_failures = 0
        github_unavailable = 0

        for path in files:
            try:
                raw_hash = _sha256_path(path)
            except Exception as exc:
                failed += 1
                failure_records.append(CandidateResult(
                    rank=None, candidate_name=path.stem, source_file=path.name, eligible=False, total_score=0,
                    score_breakdown={}, rejection_reasons=[f"Resume unreadable: {exc.__class__.__name__}"],
                    concerns=[str(exc)], llm_status="not_used", github_status="not_provided",
                ))
                continue
            if raw_hash in duplicate_hashes:
                continue
            duplicate_hashes.add(raw_hash)
            try:
                resume = parse_resume(path)
            except Exception as exc:
                failed += 1
                failure_records.append(CandidateResult(
                    rank=None, candidate_name=path.stem, source_file=path.name, eligible=False, total_score=0,
                    score_breakdown={}, rejection_reasons=[f"Resume unreadable: {exc.__class__.__name__}"],
                    concerns=[str(exc)], llm_status="not_used", github_status="not_provided",
                ))
                continue
            successful.append((path, resume))

        eligible_results: list[CandidateResult] = []
        rejected_results: list[CandidateResult] = []
        for path, resume in successful:
            eligibility = evaluate_eligibility(resume)
            if not eligibility.eligible:
                rejected_results.append(CandidateResult(
                    rank=None,
                    candidate_name=resume.candidate_name,
                    source_file=resume.file_name,
                    eligible=False,
                    total_score=0,
                    score_breakdown={},
                    matched_skills=eligibility.matched_skills,
                    rejection_reasons=eligibility.rejection_reasons,
                    evidence={
                        "python": eligibility.python_evidence,
                        "ai": eligibility.ai_evidence,
                    },
                    llm_status="not_used",
                    github_status="not_provided" if not resume.github_url else "not_provided",
                ))
                continue

            llm_status = "mock"
            try:
                analysis = self.llm.analyze_resume(resume)
                project_analysis = self.llm.analyze_projects(resume, analysis)
                llm_status = "success" if self.settings.llm_provider == "openai" else "mock"
            except Exception:
                llm_failures += 1
                llm_status = "failed"
                from .extraction import deterministic_resume_analysis
                analysis = deterministic_resume_analysis(resume)
                project_analysis = None

            github_result = self.github.enrich(resume.github_url)
            if github_result.status == "success": github_successes += 1
            elif github_result.status in ("api_error", "rate_limited"): github_failures += 1
            elif github_result.status in ("not_provided", "private_or_unavailable", "invalid_url"): github_unavailable += 1

            score = score_resume(
                resume,
                analysis,
                github_score=github_result.score,
                github_evidence=github_result.evidence,
            )

            # In OpenAI mode, blend model assessment into the deterministic evidence-backed result without
            # allowing the model to control eligibility. Caps and penalties remain deterministic.
            if project_analysis is not None and self.settings.llm_provider == "openai":
                score.breakdown.ai_project_depth = min(40, max(0, round((score.breakdown.ai_project_depth + project_analysis.ai_depth) / 2)))
                score.breakdown.python_backend = min(30, max(0, round((score.breakdown.python_backend + project_analysis.python_backend_depth) / 2)))
                score.breakdown.cloud_fullstack = min(15, max(0, round((score.breakdown.cloud_fullstack + project_analysis.cloud_depth) / 2)))
                score.breakdown.engineering_depth = min(5, max(0, round((score.breakdown.engineering_depth + project_analysis.engineering_depth) / 2)))

            total = score.breakdown.total
            eligible_results.append(CandidateResult(
                rank=None,
                candidate_name=resume.candidate_name,
                source_file=resume.file_name,
                eligible=True,
                total_score=total,
                score_breakdown=score.breakdown,
                matched_skills=evaluation_matched_skills(resume),
                project_summary=score.project_summary,
                github_summary=github_result.summary,
                strengths=score.strengths,
                concerns=score.concerns,
                evidence=score.evidence,
                penalties=score.penalties,
                llm_status=llm_status,
                github_status=github_result.status,
            ))

        eligible_results.sort(key=lambda r: (
            -r.total_score,
            -r.score_breakdown.ai_project_depth,
            -r.score_breakdown.python_backend,
            -r.score_breakdown.engineering_depth,
            r.candidate_name.casefold(),
        ))
        for idx, candidate in enumerate(eligible_results, start=1):
            candidate.rank = idx

        candidates = eligible_results + rejected_results + failure_records
        # Deterministic output: ranked candidates first, then rejected/failures alphabetically.
        candidates[len(eligible_results):] = sorted(
            candidates[len(eligible_results):],
            key=lambda r: (r.candidate_name.casefold(), r.source_file.casefold()),
        )

        duration = time.perf_counter() - start
        summary = BatchSummary(
            total_resumes=len(files),
            successfully_parsed=len(successful),
            eligible=len(eligible_results),
            rejected=len(rejected_results),
            failed_or_unreadable=failed,
            duplicate_files=len(files) - len(successful) - failed,
            llm_failures=llm_failures,
            github_successes=github_successes,
            github_failures=github_failures,
            github_unavailable=github_unavailable,
            processing_duration_seconds=round(duration, 3),
        )
        return ScreeningOutput(
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
            batch_summary=summary,
            candidates=candidates,
        )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluation_matched_skills(resume) -> list[str]:
    # Keep this readable: skills are already extracted, and common technologies provide useful reviewer context.
    preferred = {"python", "fastapi", "flask", "django", "postgresql", "redis", "langchain", "langgraph", "rag", "docker", "gcp", "aws", "react", "next.js"}
    result = [s for s in resume.skills if any(token in s.casefold() for token in preferred)]
    return result[:20]
