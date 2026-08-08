from typing import Literal
from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    source: str = Field(description="The source document filename, e.g. 'hr_policy_handbook.pdf'")
    page: int = Field(description="The page number within the source document")


class AnswerResponse(BaseModel):
    answer: str = Field(
        description="The answer to the user's question, or an explanation of what "
                     "information is missing if the context doesn't support an answer."
    )
    answer_found: bool = Field(
        description="True only if the provided context contains enough information "
                     "to confidently answer the question. False if the context is "
                     "insufficient, irrelevant, or only tangentially related."
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="'high' if the answer is explicitly and completely supported by "
                     "the context, 'medium' if partially supported, 'low' if significant "
                     "inference was required."
    )
    sources: list[SourceCitation] = Field(
        default_factory=list,
        description="The specific source chunks actually used to construct the answer. "
                     "Empty if answer_found is false.",
    )