"""Agent skill registry — modular capabilities for the orchestrator.

Each skill encapsulates a specific capability (retrieval, image understanding,
routing, evaluation) and exposes a uniform interface for the orchestrator.

Usage:
    from industry_agent.agent.skills import SKILL_REGISTRY

    retrieval = SKILL_REGISTRY["retrieval"]
    results = retrieval.execute(query="电钻指示灯闪烁")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SkillResult:
    """Standard result container returned by all skills."""

    success: bool = True
    data: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str = ""


class BaseSkill(ABC):
    """Abstract base class for agent skills."""

    name: str = "base"
    description: str = ""

    @abstractmethod
    def execute(self, **kwargs: Any) -> SkillResult:
        """Execute the skill with given parameters."""

    def is_available(self) -> bool:
        """Check if this skill is ready to use."""
        return True


# Lazy imports to avoid circular dependencies
def _get_retrieval_skill() -> type[BaseSkill]:
    from industry_agent.agent.skills.retrieval_skill import RetrievalSkill

    return RetrievalSkill


def _get_image_skill() -> type[BaseSkill]:
    from industry_agent.agent.skills.image_skill import ImageSkill

    return ImageSkill


def _get_routing_skill() -> type[BaseSkill]:
    from industry_agent.agent.skills.routing_skill import RoutingSkill

    return RoutingSkill


def _get_evaluation_skill() -> type[BaseSkill]:
    from industry_agent.agent.skills.evaluation_skill import EvaluationSkill

    return EvaluationSkill


SKILL_REGISTRY: dict[str, type[BaseSkill]] = {
    "retrieval": _get_retrieval_skill,
    "image": _get_image_skill,
    "routing": _get_routing_skill,
    "evaluation": _get_evaluation_skill,
}


def get_skill(name: str) -> BaseSkill:
    """Instantiate and return a skill by name."""
    factory = SKILL_REGISTRY.get(name)
    if factory is None:
        raise ValueError(f"Unknown skill: {name}. Available: {list(SKILL_REGISTRY.keys())}")
    skill_class = factory() if callable(factory) and not isinstance(factory, type) else factory
    return skill_class()
