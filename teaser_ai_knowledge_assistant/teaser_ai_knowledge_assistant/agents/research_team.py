from services.llm_service import generate_response


def planner_agent(
    topic: str,
) -> str:
    prompt = f"""
TOPIC:
{topic}

Create a focused research and writing plan.

Return:
1. The main objective
2. Five research questions
3. The report structure
4. Risks, assumptions, and limitations

Do not write the final report yet.
""".strip()

    return generate_response(
        prompt,
        system_prompt=(
            "You are the Planner Agent. Decompose topics "
            "into logical and useful plans."
        ),
        temperature=0.1,
    )


def researcher_agent(
    topic: str,
    plan: str,
    source_context: str = "",
) -> str:
    if source_context.strip():
        context = source_context
    else:
        context = (
            "No external sources were supplied. Use general "
            "model knowledge, identify uncertainty, and do not "
            "claim to have browsed or verified current facts."
        )

    prompt = f"""
TOPIC:
{topic}

PLANNER'S PLAN:
{plan}

AVAILABLE SOURCE CONTEXT:
{context}

Develop organized research notes.

Rules:
- Prefer supplied source context.
- Do not invent quotations or statistics.
- Do not claim that web research was performed.
- Mark uncertain or time-sensitive claims.
- Separate findings, examples, risks, and open questions.
""".strip()

    return generate_response(
        prompt,
        system_prompt=(
            "You are the Researcher Agent. Gather and organize "
            "reliable material for the Writer Agent."
        ),
        temperature=0.2,
    )


def writer_agent(
    topic: str,
    plan: str,
    research_notes: str,
) -> str:
    prompt = f"""
TOPIC:
{topic}

PLAN:
{plan}

RESEARCH NOTES:
{research_notes}

Write a professional report.

Include:
- Title
- Executive summary
- Key findings
- Practical examples
- Risks and limitations
- Recommendations
- Conclusion

Do not add facts that are absent from the notes.
""".strip()

    return generate_response(
        prompt,
        system_prompt=(
            "You are the Writer Agent. Turn research notes "
            "into a clear and useful report."
        ),
        temperature=0.3,
    )


def reviewer_agent(
    topic: str,
    draft: str,
) -> str:
    prompt = f"""
TOPIC:
{topic}

DRAFT REPORT:
{draft}

Review and revise the report.

Improve:
- Accuracy
- Internal consistency
- Organization
- Readability
- Practical usefulness
- Treatment of uncertainty
- Removal of unsupported claims

Return only the complete revised report.
""".strip()

    return generate_response(
        prompt,
        system_prompt=(
            "You are the Reviewer Agent. Produce the strongest "
            "and most responsible final version."
        ),
        temperature=0.1,
    )


def run_research_team(
    topic: str,
    source_context: str = "",
) -> dict[str, str]:
    plan = planner_agent(topic)

    research = researcher_agent(
        topic,
        plan,
        source_context,
    )

    draft = writer_agent(
        topic,
        plan,
        research,
    )

    final_report = reviewer_agent(
        topic,
        draft,
    )

    return {
        "plan": plan,
        "research": research,
        "draft": draft,
        "final": final_report,
    }