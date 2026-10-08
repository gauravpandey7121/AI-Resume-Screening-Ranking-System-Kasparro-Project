from src.resume_screening.models import Project, ResumeData, ResumeAnalysis
from src.resume_screening.scoring import score_resume


def resume_with_project(desc: str, skills: list[str]) -> ResumeData:
    return ResumeData(
        file_name="resume.pdf",
        file_hash="x",
        candidate_name="Test",
        skills=skills,
        projects=[Project(name="AI Project", description=desc, evidence=[desc], technologies=skills)],
        sections={"projects": desc, "experience": "", "skills": ", ".join(skills)},
        raw_text=desc,
    )


def test_strong_rag_agent_project_scores_above_shallow_wrapper():
    strong = resume_with_project(
        "Built a stateful multi-agent RAG workflow with LangGraph, embeddings, vector retrieval, tool calling, Redis memory, and an evaluation suite.",
        ["Python", "LangGraph", "FastAPI", "PostgreSQL", "Redis", "Docker"],
    )
    shallow = resume_with_project(
        "Built a chatbot using OpenAI API to answer user questions.",
        ["Python", "OpenAI API"],
    )
    s1 = score_resume(strong, ResumeAnalysis(candidate_name="Test", skills=strong.skills, projects=["AI Project"]))
    s2 = score_resume(shallow, ResumeAnalysis(candidate_name="Test", skills=shallow.skills, projects=["AI Project"]))
    assert s1.breakdown.ai_project_depth > s2.breakdown.ai_project_depth
    assert any(p.type == "shallow_ai_project" for p in s2.penalties)


def test_scores_are_within_baseline_caps():
    resume = resume_with_project(
        "Built a RAG pipeline with retrieval, agents, tool calling, evaluation, FastAPI, PostgreSQL, Redis, Docker and AWS deployment.",
        ["Python", "FastAPI", "PostgreSQL", "Redis", "Docker"],
    )
    result = score_resume(resume, ResumeAnalysis(candidate_name="Test", skills=resume.skills, projects=["AI Project"]), github_score=10)
    assert 0 <= result.breakdown.ai_project_depth <= 40
    assert 0 <= result.breakdown.python_backend <= 30
    assert 0 <= result.breakdown.cloud_fullstack <= 15
    assert 0 <= result.breakdown.github <= 10
    assert 0 <= result.breakdown.engineering_depth <= 5
    assert 0 <= result.breakdown.total <= 100
