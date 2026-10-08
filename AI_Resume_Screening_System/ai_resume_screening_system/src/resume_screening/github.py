from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from .config import Settings
from .models import GithubResult


class GithubEnricher:
    """Lightweight, serial GitHub enrichment with per-run in-memory caching."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._cache: dict[str, GithubResult] = {}
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/vnd.github+json",
            "User-Agent": "ai-resume-screening-assignment/1.0",
            "X-GitHub-Api-Version": "2026-03-10",
        })
        if settings.github_token:
            self._session.headers["Authorization"] = f"Bearer {settings.github_token}"

    @staticmethod
    def extract_username(url: str | None) -> str | None:
        if not url:
            return None
        match = re.search(r"github\.com/([A-Za-z0-9-]+)", url, re.I)
        return match.group(1) if match else None

    def enrich(self, url: str | None, *, candidate_ai_terms: set[str] | None = None) -> GithubResult:
        if not url:
            return GithubResult(status="not_provided", summary="No GitHub profile provided.")
        if not self.settings.github_enabled:
            return GithubResult(status="api_error", summary="GitHub enrichment disabled for this run.")
        username = self.extract_username(url)
        if not username:
            return GithubResult(status="invalid_url", summary="GitHub URL was present but could not be parsed.")
        if username in self._cache:
            return self._cache[username]

        try:
            # One lightweight request per username keeps a ~50-resume batch well below
            # unauthenticated GitHub's request budget while still exposing repo freshness
            # and language/topic signals.
            repos_resp = self._session.get(
                f"https://api.github.com/users/{username}/repos?per_page=30&sort=updated",
                timeout=self.settings.github_timeout_seconds,
            )
            if repos_resp.status_code in (403, 429):
                result = GithubResult(status="rate_limited", username=username, summary="GitHub API rate limit prevented enrichment.")
                self._cache[username] = result
                return result
            if repos_resp.status_code == 404:
                result = GithubResult(status="private_or_unavailable", username=username, summary="GitHub profile is unavailable or not public.")
                self._cache[username] = result
                return result
            repos_resp.raise_for_status()
            repos = repos_resp.json()

            recent_cutoff = datetime.now(timezone.utc) - timedelta(days=self.settings.github_recent_days)
            recent_repos = []
            maintained = 0
            relevant = 0
            ai_terms = {x.casefold() for x in (candidate_ai_terms or set())}
            for repo in repos:
                pushed = _parse_datetime(repo.get("pushed_at")) or _parse_datetime(repo.get("updated_at"))
                if pushed and pushed >= recent_cutoff:
                    recent_repos.append(repo)
                    maintained += 1
                language = (repo.get("language") or "").casefold()
                topic_text = " ".join(repo.get("topics") or []).casefold()
                repo_blob = f"{repo.get('name', '')} {repo.get('description', '') or ''} {language} {topic_text}".casefold()
                if "python" in repo_blob or any(term in repo_blob for term in ai_terms):
                    relevant += 1

            recent_activity_score = min(5, len(recent_repos))
            maintained_repo_score = min(5, (1 if maintained >= 1 else 0) + min(2, maintained // 3) + min(2, relevant // 2))
            score = recent_activity_score + maintained_repo_score
            evidence = [
                f"{len(recent_repos)} public repositories pushed/updated in the last {self.settings.github_recent_days} days.",
                f"{len(repos)} public repositories returned; {maintained} recently maintained repositories.",
            ]
            if relevant:
                evidence.append(f"{relevant} repositories appear relevant to Python/AI signals.")
            result = GithubResult(
                status="success",
                username=username,
                score=score,
                summary=f"Recent public repository activity: {len(recent_repos)} updated in the last {self.settings.github_recent_days} days; {relevant} appear Python/AI-relevant.",
                evidence=evidence,
                recent_activity_score=recent_activity_score,
                maintained_repo_score=maintained_repo_score,
            )
        except requests.RequestException as exc:
            result = GithubResult(status="api_error", username=username, summary=f"GitHub API unavailable: {exc.__class__.__name__}.")
        except Exception as exc:
            result = GithubResult(status="api_error", username=username, summary=f"GitHub enrichment failed: {exc.__class__.__name__}.")

        self._cache[username] = result
        return result


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _event_time(event: dict[str, Any]) -> datetime | None:
    return _parse_datetime(event.get("created_at"))
