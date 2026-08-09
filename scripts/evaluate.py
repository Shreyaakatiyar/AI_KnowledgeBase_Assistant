import sys
import json
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.logging_config import setup_logging, get_logger
from app.services.retrieval import retrieve
from app.services.qa_pipeline import answer_question
from eval.golden_dataset import GOLDEN_DATASET
from eval.metrics import retrieval_hit_rate, reciprocal_rank, judge_correctness, judge_faithfulness

setup_logging()
logger = get_logger(__name__)


def run_evaluation() -> list[dict]:
    results = []
    for case in GOLDEN_DATASET:
        chunks = retrieve(case.question)
        answer = answer_question(case.question)

        hit = retrieval_hit_rate(chunks, case.expected_sources)
        rr = reciprocal_rank(chunks, case.expected_sources)
        answer_found_correct = answer.answer_found == case.expected_answer_found

        correctness = None
        faithfulness = None
        if case.reference_answer and answer.answer_found:
            correctness = judge_correctness(answer.answer, case.reference_answer)
        if answer.answer_found and answer.sources:
            faithfulness = judge_faithfulness(answer.answer, chunks)

        result = {
            "question": case.question,
            "category": case.category,
            "retrieval_hit": hit,
            "reciprocal_rank": round(rr, 3),
            "answer_found_correct": answer_found_correct,
            "answer_found": answer.answer_found,
            "confidence": answer.confidence,
            "is_correct": correctness.is_correct if correctness else None,
            "is_faithful": faithfulness.is_faithful if faithfulness else None,
            "unsupported_claims": faithfulness.unsupported_claims if faithfulness else [],
        }
        results.append(result)

        passed = (
            hit and answer_found_correct
            and (correctness is None or correctness.is_correct)
            and (faithfulness is None or faithfulness.is_faithful)
        )
        logger.info(f"[{'PASS' if passed else 'FAIL'}] ({case.category}) {case.question[:60]}")

    return results


def summarize(results: list[dict]) -> dict:
    n = len(results)
    hit_rate = sum(r["retrieval_hit"] for r in results) / n
    mrr = sum(r["reciprocal_rank"] for r in results) / n
    answer_found_accuracy = sum(r["answer_found_correct"] for r in results) / n

    judged_correctness = [r["is_correct"] for r in results if r["is_correct"] is not None]
    correctness_rate = sum(judged_correctness) / len(judged_correctness) if judged_correctness else None

    judged_faithfulness = [r["is_faithful"] for r in results if r["is_faithful"] is not None]
    faithfulness_rate = sum(judged_faithfulness) / len(judged_faithfulness) if judged_faithfulness else None

    return {
        "n_cases": n,
        "retrieval_hit_rate": round(hit_rate, 3),
        "mean_reciprocal_rank": round(mrr, 3),
        "answer_found_accuracy": round(answer_found_accuracy, 3),
        "answer_correctness_rate": round(correctness_rate, 3) if correctness_rate is not None else None,
        "faithfulness_rate": round(faithfulness_rate, 3) if faithfulness_rate is not None else None,
    }


def main():
    logger.info(f"Running evaluation on {len(GOLDEN_DATASET)} golden test cases...")
    results = run_evaluation()
    summary = summarize(results)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for key, value in summary.items():
        print(f"  {key:30s}: {value}")
    print("=" * 60)

    failing = [r for r in results if not r["retrieval_hit"] or not r["answer_found_correct"]]
    if failing:
        print(f"\n{len(failing)} case(s) need attention:")
        for r in failing:
            print(f"  - [{r['category']}] {r['question']}")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "results": results,
    }
    report_dir = Path(__file__).resolve().parent.parent / "eval" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info(f"Full report saved to {report_path}")


if __name__ == "__main__":
    main()