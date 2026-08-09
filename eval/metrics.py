from pydantic import BaseModel, Field

from app.services.generation import generate_structured
from app.services.vector_store import RetrievedChunk


def retrieval_hit_rate(chunks: list[RetrievedChunk], expected_sources: list[tuple[str, int]]) -> bool:
    if not expected_sources:
        return True
    retrieved_keys = {(c.source, c.page) for c in chunks}
    return any(key in retrieved_keys for key in expected_sources)


def reciprocal_rank(chunks: list[RetrievedChunk], expected_sources: list[tuple[str, int]]) -> float:
    if not expected_sources:
        return 1.0
    for rank, chunk in enumerate(chunks, start=1):
        if (chunk.source, chunk.page) in expected_sources:
            return 1.0 / rank
    return 0.0


class CorrectnessJudgment(BaseModel):
    is_correct: bool = Field(
        description="True if the generated answer conveys the same key information "
                     "as the reference answer"
    )
    explanation: str = Field(description="Brief justification for the judgment")


_CORRECTNESS_PROMPT = """You are grading an AI assistant's answer against a reference answer.
Judge whether the GENERATED ANSWER conveys the same key facts as the REFERENCE ANSWER.
Minor wording differences are fine - focus on factual correctness and completeness.

Reference answer: {reference}
Generated answer: {generated}"""


def judge_correctness(generated_answer: str, reference_answer: str) -> CorrectnessJudgment:
    return generate_structured(
        prompt=_CORRECTNESS_PROMPT.format(reference=reference_answer, generated=generated_answer),
        response_schema=CorrectnessJudgment,
    )


class FaithfulnessJudgment(BaseModel):
    is_faithful: bool = Field(
        description="True if every claim in the answer is directly supported by the provided context"
    )
    unsupported_claims: list[str] = Field(default_factory=list)


_FAITHFULNESS_PROMPT = """You are checking an AI assistant's answer for hallucination.
Given the CONTEXT it was allowed to use and the ANSWER it produced, identify any claim
in the answer that is NOT directly supported by the context - even if the claim happens
to be true in general. The answer must be grounded ONLY in the given context.

Context:
{context}

Answer:
{answer}"""


def judge_faithfulness(answer: str, context_chunks: list[RetrievedChunk]) -> FaithfulnessJudgment:
    context = "\n\n".join(c.text for c in context_chunks)
    return generate_structured(
        prompt=_FAITHFULNESS_PROMPT.format(context=context, answer=answer),
        response_schema=FaithfulnessJudgment,
    )