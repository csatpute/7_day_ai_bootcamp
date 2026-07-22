PROMPT_TEMPLATES = {
    "Beginner Explanation": {
        "system": (
            "You are a patient instructor who explains "
            "technical ideas to complete beginners."
        ),
        "instructions": """
Explain the subject in beginner-friendly language.

Requirements:
- Start with a one-sentence definition.
- Use a simple analogy.
- Break the explanation into small sections.
- Include one practical example.
- End with three key takeaways.
""",
    },

    "Executive Summary": {
        "system": (
            "You are an executive analyst who communicates "
            "with clarity and brevity."
        ),
        "instructions": """
Create an executive summary.

Requirements:
- State the central message first.
- Identify important findings.
- Highlight risks and opportunities.
- Recommend next actions.
- Use concise headings and bullets.
""",
    },

    "Detailed Analysis": {
        "system": (
            "You are a rigorous analyst. Separate evidence, "
            "assumptions, and recommendations."
        ),
        "instructions": """
Produce a detailed analysis.

Requirements:
- Define the issue.
- Examine key factors.
- Discuss benefits, risks, and trade-offs.
- Identify assumptions and uncertainties.
- Provide practical recommendations.
""",
    },

    "Action Plan": {
        "system": (
            "You are an implementation consultant who turns "
            "ideas into clear actions."
        ),
        "instructions": """
Create an actionable implementation plan.

Requirements:
- Organize the work into phases.
- Give each phase a goal and concrete tasks.
- Identify dependencies and risks.
- Define measurable success criteria.
- End with the first three actions to take.
""",
    },

    "Blog Article": {
        "system": (
            "You are an educational technology writer who "
            "creates useful and engaging articles."
        ),
        "instructions": """
Write a practical blog article.

Requirements:
- Create a strong title and introduction.
- Use descriptive headings.
- Include examples.
- Avoid unsupported hype.
- End with a concise conclusion.
""",
    },
}


def build_prompt(
    template_name: str,
    subject: str,
    context: str = "",
) -> tuple[str, str]:
    template = PROMPT_TEMPLATES[template_name]

    context_value = (
        context.strip()
        if context.strip()
        else "No additional context was provided."
    )

    prompt = f"""
SUBJECT OR REQUEST:
{subject.strip()}

OPTIONAL CONTEXT:
{context_value}

TASK INSTRUCTIONS:
{template["instructions"].strip()}
""".strip()

    return template["system"], prompt