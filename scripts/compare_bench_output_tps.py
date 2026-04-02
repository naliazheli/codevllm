import argparse
import json
import os


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--optimized", required=True)
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    b = load(args.baseline)
    o = load(args.optimized)

    b_rows = b.get("summary_results") or b.get("results", [])
    o_rows = o.get("summary_results") or o.get("results", [])
    b_map = {r["concurrency"]: r for r in b_rows}
    o_map = {r["concurrency"]: r for r in o_rows}
    levels = sorted(set(b_map.keys()) & set(o_map.keys()))

    lines = []
    lines.append(f"baseline: {os.path.abspath(args.baseline)}")
    lines.append(f"optimized: {os.path.abspath(args.optimized)}")
    lines.append("")
    lines.append("| concurrency | baseline_output_tps | optimized_output_tps | delta | delta_pct |")
    lines.append("|---:|---:|---:|---:|---:|")

    for c in levels:
        b_tps = float(b_map[c]["output_tps"])
        o_tps = float(o_map[c]["output_tps"])
        d = o_tps - b_tps
        pct = (d / b_tps * 100.0) if b_tps else 0.0
        lines.append(f"| {c} | {b_tps:.3f} | {o_tps:.3f} | {d:.3f} | {pct:.2f}% |")

    report = "\n".join(lines)
    print(report)

    if args.output:
        out = os.path.abspath(args.output)
        out_dir = os.path.dirname(out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
