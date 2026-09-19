from __future__ import annotations

from pydantic import BaseModel, Field


class NextTask(BaseModel):
    """Single next-step task assigned to the student."""

    task_description: str = Field(
        description="Concrete next task for the student. Exactly one task is allowed.",
    )
    template_guideline: str = Field(
        description="Template or action guideline the student should follow.",
    )
    acceptance_criteria: str = Field(
        description="Completion criteria used to evaluate if the task is done.",
    )


class ProjectCoachOutput(BaseModel):
    """Strict structured output contract for Project Coach."""

    project_stage: str = Field(
        description="Current stage of the project, such as ideation/prototype/validation.",
    )
    current_diagnosis: str = Field(
        description="Largest logical contradiction or gap based on rules and graph evidence.",
    )
    evidence_used: str = Field(
        description="Quoted student text or graph case evidence used for the diagnosis.",
    )
    impact_if_unfixed: str = Field(
        description="Major consequence if this issue is not fixed.",
    )
    next_task: NextTask = Field(
        description="Exactly one next task object. Lists are not allowed.",
    )


class AuditTrail(BaseModel):
    """Evidence chain bound to a single rubric scoring item."""

    quote: str = Field(default="", description="Exact student quote snippet.")
    h_rule_id: str = Field(default="", description="Triggered H-rule id, such as H1/H8.")
    kg_card_id: str = Field(default="", description="Retrieved KG knowledge-card id.")
    explanation: str = Field(default="", description="Short rationale grounded in evidence.")


class RubricScoreResult(BaseModel):
    """Single rubric item score with mandatory evidence binding."""

    item_id: str = Field(description="Rubric item id, such as R4.")
    score: float = Field(description="Normalized item score in 0-5 scale.")
    audit_trail: AuditTrail = Field(description="Bound evidence chain.")


__all__ = [
    "NextTask",
    "ProjectCoachOutput",
    "AuditTrail",
    "RubricScoreResult",
]
