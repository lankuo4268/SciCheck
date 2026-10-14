"""CLI entry for SciCheck.

Usage:
  D:/anaconda/python.exe scicheck_cli.py <input.json> [--output result.json]
  D:/anaconda/python.exe scicheck_cli.py --compare examples/piecewise_const_0_3.json
      (runs plain quad AND quad-with-breakpoints side by side, records both runs)

The default path runs the full scicheck flow (first candidate + repair if needed).
"""
import argparse
import json
import sys
from pathlib import Path

from scicheck import (
    scicheck_integration,
    normalize_problem,
    run_plain_quad,
    run_quad_with_breakpoints,
    exact_ok,
)


def _result(res: dict) -> int:
    print(json.dumps(res, indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="SciCheck: verify a numeric integral against an independent exact Fraction reference."
    )
    parser.add_argument("input", help="path to input JSON")
    parser.add_argument("--output", help="optional path to write the result JSON")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="explicitly run plain quad AND quad-with-breakpoints, record both runs",
    )
    args = parser.parse_args(argv)

    in_path = Path(args.input)
    with in_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if args.compare:
        norm = normalize_problem(payload)
        r1 = run_plain_quad(norm, run_id=1)
        r2 = run_quad_with_breakpoints(norm, run_id=2)
        out = {
            "reference_fraction": str(norm.reference_exact()),
            "reference_float": norm.reference_float(),
            "allowed_error_float": norm.allowed_error_float(),
            "runs": [
                {
                    "run_id": r.run_id,
                    "action": r.action,
                    "method": r.method,
                    "candidate": r.value,
                    "quad_abserr_reported": r.quad_abserr_reported,
                    "exact_ok": r.exact_ok,
                    "exact_check": r.exact_check,
                    "resource": r.resource,
                }
                for r in [r1, r2]
            ],
            "note": "side-by-side plain vs breakpoint quad; verdict uses exact Fraction comparison",
        }
        if args.output:
            Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
            print(f"wrote {args.output}")
        print(json.dumps(out, indent=2))
        return 0

    res = scicheck_integration(payload)
    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"wrote {out_path}")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
