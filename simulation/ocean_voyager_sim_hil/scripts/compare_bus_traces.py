#!/usr/bin/env python3
from __future__ import annotations
import argparse
from ocean_voyager.simulation.replay import compare_bus_traces


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare simulator and real-hardware device-bus traces.")
    parser.add_argument("--sim", required=True, help="Simulator bus_trace.jsonl")
    parser.add_argument("--real", required=True, help="Real hardware bus_trace.jsonl")
    parser.add_argument("--output", default="bus_trace_comparison.json")
    args = parser.parse_args()
    report = compare_bus_traces(args.sim, args.real, args.output)
    print(report)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
