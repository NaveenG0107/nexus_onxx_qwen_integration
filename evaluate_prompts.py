import os
import sys
import json
import time
import re

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath("."))

from src.generation.service import get_model_manager


def normalize_sql(sql_str: str) -> str:
    """Normalize SQL by lowercasing, stripping extra whitespace, and trailing semicolons."""
    if not sql_str:
        return ""
    # Remove markdown code block fences if present
    cleaned = re.sub(r"```(?:sql)?|```", "", sql_str, flags=re.IGNORECASE)
    # Lowercase and normalize spaces
    cleaned = re.sub(r"\s+", " ", cleaned.strip().lower())
    # Strip semicolon at the very end for loose comparison
    return cleaned.rstrip(";")


def run_evaluation(
    dataset_path: str = "nl_to_sql_dbml_create_only.jsonl",
    output_report_path: str = "evaluation_results.json",
    use_custom_onnx: bool = False,
    max_tokens: int = 80,
):
    print("=" * 70)
    print("STARTING EVALUATION DIRECTLY VIA PYTHON")
    print(f"Dataset:    {dataset_path}")
    print(f"Model Mode: {'Custom SLM ONNX' if use_custom_onnx else 'Qwen2.5-0.5B ONNX (Default)'}")
    print(f"Max Tokens: {max_tokens}")
    print("=" * 70)

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        return

    # Load dataset
    records = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} test questions.\n")
    print("Initializing Model Manager...")
    manager = get_model_manager()

    results = []
    total_matched = 0
    total_time = 0.0
    model_name = "unknown"

    print("-" * 70)
    for idx, item in enumerate(records, 1):
        test_id = item.get("id", f"test_{idx:02d}")
        prompt = item.get("prompt", "")
        expected_sql = item.get("expected_sql", "")

        t_start = time.perf_counter()
        generated_resp, model_name = manager.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            use_custom_onnx=use_custom_onnx,
        )
        elapsed_sec = round(time.perf_counter() - t_start, 4)
        total_time += elapsed_sec

        # Comparison logic
        norm_expected = normalize_sql(expected_sql)
        norm_generated = normalize_sql(generated_resp)
        is_exact_match = (expected_sql.strip() == (generated_resp or "").strip())
        is_norm_match = (norm_expected == norm_generated)

        if is_norm_match:
            total_matched += 1

        status = "MATCH" if is_norm_match else "DIFF"

        print(f"[{idx}/{len(records)}] {test_id} | Status: {status} | Time: {elapsed_sec:.4f}s")
        print(f"  Prompt:   {prompt}")
        print(f"  Expected: {expected_sql}")
        print(f"  Received: {(generated_resp or '').strip()}")
        print("-" * 70)

        results.append({
            "id": test_id,
            "prompt": prompt,
            "expected_sql": expected_sql,
            "llm_response": generated_resp,
            "execution_time_seconds": elapsed_sec,
            "exact_match": is_exact_match,
            "normalized_match": is_norm_match,
        })

    # Summary
    accuracy = (total_matched / len(records)) * 100 if records else 0.0
    summary = {
        "model_used": model_name,
        "total_questions": len(records),
        "total_matched": total_matched,
        "accuracy_percentage": round(accuracy, 2),
        "total_execution_time_seconds": round(total_time, 4),
        "avg_execution_time_per_question_seconds": round(total_time / len(records), 4) if records else 0.0,
        "results": results,
    }

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Model Used:                {summary['model_used']}")
    print(f"Total Questions Evaluated: {summary['total_questions']}")
    print(f"Matches (Normalized):      {summary['total_matched']} / {summary['total_questions']}")
    print(f"Accuracy:                  {summary['accuracy_percentage']}%")
    print(f"Total Execution Time:      {summary['total_execution_time_seconds']}s")
    print(f"Avg Time Per Question:     {summary['avg_execution_time_per_question_seconds']}s")
    print("=" * 70)

    # Save to JSON file
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Full results saved to: {output_report_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate NL to SQL Prompts directly with Python (Defaults to Qwen ONNX)")
    parser.add_argument("--use-custom-slm", action="store_true", default=False, help="Use Custom SLM ONNX instead of Qwen ONNX")
    parser.add_argument("--max-tokens", type=int, default=80, help="Max tokens for generation")
    parser.add_argument("--dataset", type=str, default="nl_to_sql_dbml_create_only.jsonl", help="Dataset path")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output file")
    args = parser.parse_args()

    run_evaluation(
        dataset_path=args.dataset,
        output_report_path=args.output,
        use_custom_onnx=args.use_custom_slm,
        max_tokens=args.max_tokens,
    )

