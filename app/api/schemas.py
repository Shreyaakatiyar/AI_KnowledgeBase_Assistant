from pydantic import BaseModel, Field
from app.models.schemas import AnswerResponse, SourceCitation


class QuestionRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The user's question for the knowledge base assistant.",
        examples=["How many days of PTO do I get per year?"],
    )


class QuestionResponse(BaseModel):
    query: str
    answer: str
    answer_found: bool
    confidence: str
    sources: list[SourceCitation]
    latency_ms: float

    @classmethod
    def from_answer(cls, query: str, answer: AnswerResponse, latency_ms: float) -> "QuestionResponse":
        return cls(
            query=query,
            answer=answer.answer,
            answer_found=answer.answer_found,
            confidence=answer.confidence,
            sources=answer.sources,
            latency_ms=round(latency_ms, 1),
        )


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int