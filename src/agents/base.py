from abc import ABC, abstractmethod
from typing import Dict, Any
from .types import Task, StepResult

class BaseAgent(ABC):
    name: str = "base"

    @abstractmethod
    def run(self, task: Task) -> StepResult:
        ...

    def ok(self, detail: str = "", **artifacts) -> StepResult:
        return StepResult(name=self.name, status="succeeded", detail=detail, artifacts=artifacts)

    def fail(self, detail: str = "", **artifacts) -> StepResult:
        return StepResult(name=self.name, status="failed", detail=detail, artifacts=artifacts)