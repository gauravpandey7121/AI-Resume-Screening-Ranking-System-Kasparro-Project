from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.resume_screening.config import settings
from src.resume_screening.pipeline import ScreeningPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Resume Screening & Ranking System")
    parser.add_argument("--input", required=True, help="Directory containing resumes")
    parser.add_argument("--output", required=True, help="Output JSON file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input)
    output_path = Path(args.output)
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    result = ScreeningPipeline(settings).run(input_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.model_dump(mode="json"), indent=2), encoding="utf-8")

    summary = result.batch_summary
    print(f"Processed: {summary.total_resumes}")
    print(f"Parsed: {summary.successfully_parsed}")
    print(f"Eligible: {summary.eligible}")
    print(f"Rejected: {summary.rejected}")
    print(f"Failed/unreadable: {summary.failed_or_unreadable}")
    print(f"Duplicates skipped: {summary.duplicate_files}")
    print(f"Results: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
