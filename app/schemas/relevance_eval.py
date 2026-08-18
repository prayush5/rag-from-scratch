from pydantic import BaseModel


class RelevanceEvaluation(BaseModel):
    relevant: bool
    reason: str