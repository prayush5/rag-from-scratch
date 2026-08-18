from pydantic import BaseModel

class FaithfulnessEvaluation(BaseModel):
    faithful: bool
    unsupported_claims: list[str]
    reason: str