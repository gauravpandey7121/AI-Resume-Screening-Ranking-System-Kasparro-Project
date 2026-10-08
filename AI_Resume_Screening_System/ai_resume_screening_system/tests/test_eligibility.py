from src.resume_screening.eligibility import evaluate_eligibility
from src.resume_screening.models import Project, ResumeData


def make_resume(text: str, skills: list[str], projects: list[Project] | None = None) -> ResumeData:
    return ResumeData(
        file_name="test.pdf",
        file_hash="x",
        candidate_name="Test Candidate",
        skills=skills,
        projects=projects or [],
        sections={"skills": ", ".join(skills), "projects": text},
        raw_text=text,
    )


def test_python_and_rag_is_eligible():
    resume = make_resume(
        "Built a RAG pipeline using embeddings and retrieval with document chunking.",
        ["Python", "LangChain"],
        [Project(name="Doc Assistant", description="Built a RAG pipeline using embeddings and retrieval with document chunking.", evidence=["Built a RAG pipeline using embeddings and retrieval with document chunking."])],
    )
    result = evaluate_eligibility(resume)
    assert result.eligible is True


def test_java_react_only_is_rejected_for_python():
    resume = make_resume("Built a React app with Spring Boot REST APIs.", ["Java", "React", "Spring Boot"])
    result = evaluate_eligibility(resume)
    assert result.eligible is False
    assert "No evidence of Python stack" in result.rejection_reasons


def test_python_without_ai_is_rejected():
    resume = make_resume("Built a Django CRUD application using Python and PostgreSQL.", ["Python", "Django", "PostgreSQL"])
    result = evaluate_eligibility(resume)
    assert result.eligible is False
    assert "No AI/agentic project evidence" in result.rejection_reasons


def test_missing_both_reports_both_reasons():
    resume = make_resume("Built a JavaScript portfolio website.", ["JavaScript", "React"])
    result = evaluate_eligibility(resume)
    assert result.eligible is False
    assert set(result.rejection_reasons) == {"No evidence of Python stack", "No AI/agentic project evidence"}
