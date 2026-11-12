"""展示资料统计、饱和曲线轮廓与交叉检查，不改变原始文件。"""
import argparse
import json

from .curve import analyze, best_curves_only_bounds, brute_force_bounds
from .domain import summarize
from .io import load


def _emit(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["inspect", "profile", "crosscheck"])
    parser.add_argument("file")
    parser.add_argument("--candidate-n", type=float, default=None)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    value, digest = load(args.file)
    if args.command == "inspect":
        _emit({"source_sha256": digest, "summary": summarize(value)})
    elif args.command == "profile":
        report = analyze(value)
        report["source_sha256"] = digest
        _emit(report)
    else:
        candidates = (
            [args.candidate_n] if args.candidate_n is not None
            else value.get("candidate_n", [])
        )
        if not candidates:
            raise ValueError("资料中没有候选规模，请用 --candidate-n 指定")
        checks = []
        for candidate in candidates:
            checks.append({
                "candidate_n": candidate,
                "analytic": {
                    "lower": None,
                    "upper": None,
                },
                "brute_force": brute_force_bounds(
                    value, candidate, steps=args.steps),
                "best_curves_only": best_curves_only_bounds(value, candidate),
            })
            report = analyze(value, [candidate])
            checks[-1]["analytic"] = report["candidates"][0][
                "prediction_interval"]
        _emit({"source_sha256": digest, "crosscheck": checks})


if __name__ == "__main__":
    main()
