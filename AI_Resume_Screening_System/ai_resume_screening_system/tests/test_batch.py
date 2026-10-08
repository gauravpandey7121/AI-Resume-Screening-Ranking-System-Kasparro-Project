from pathlib import Path

from src.resume_screening.config import Settings
from src.resume_screening.models import ResumeData
from src.resume_screening.pipeline import ScreeningPipeline


class FakeGithub:
    def enrich(self, url, candidate_ai_terms=None):
        from src.resume_screening.models import GithubResult
        return GithubResult(status="not_provided", summary="No GitHub profile provided.")


def test_batch_continues_after_bad_file(tmp_path: Path, monkeypatch):
    good = tmp_path / "good.txt"
    good.write_text("Alice\nSkills\nPython, LangChain\nProjects\nBuilt a RAG pipeline using embeddings.", encoding="utf-8")
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    unsupported = tmp_path / "notes.md"
    unsupported.write_text("ignore me", encoding="utf-8")

    from src.resume_screening import pipeline as pipeline_module
    original = pipeline_module.parse_resume

    def fake_parse(path):
        if path.suffix == ".pdf":
            raise ValueError("broken file")
        return original(path)

    monkeypatch.setattr(pipeline_module, "parse_resume", fake_parse)
    output = ScreeningPipeline(Settings(), github=FakeGithub()).run(tmp_path)
    assert output.batch_summary.total_resumes == 2
    assert output.batch_summary.successfully_parsed == 1
    assert output.batch_summary.failed_or_unreadable == 1
    assert output.batch_summary.eligible == 1



def test_exact_duplicate_file_is_skipped(tmp_path: Path):
    content = "Alice\nSkills\nPython, LangChain\nProjects\nBuilt a RAG pipeline using embeddings."
    (tmp_path / "a.txt").write_text(content, encoding="utf-8")
    (tmp_path / "b.txt").write_text(content, encoding="utf-8")
    output = ScreeningPipeline(Settings(), github=FakeGithub()).run(tmp_path)
    assert output.batch_summary.total_resumes == 2
    assert output.batch_summary.successfully_parsed == 1
    assert output.batch_summary.duplicate_files == 1


def test_github_rate_limit_is_non_fatal(monkeypatch):
    from src.resume_screening.github import GithubEnricher
    from src.resume_screening.config import Settings

    class Response:
        status_code = 429
        ok = False

        def json(self):
            return {}

        def raise_for_status(self):
            raise AssertionError("should not raise")

    gh = GithubEnricher(Settings(github_enabled=True))
    monkeypatch.setattr(gh._session, "get", lambda *args, **kwargs: Response())
    result = gh.enrich("https://github.com/example-user")
    assert result.status == "rate_limited"
    assert result.score == 0
