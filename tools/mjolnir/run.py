"""Mjolnir runner — orchestrates Lanes A, B, C and reports results.

Usage:
    python -m tools.mjolnir.run                  # all lanes
    python -m tools.mjolnir.run --lane a         # mock JS only
    python -m tools.mjolnir.run --lane b         # real-browser only
    python -m tools.mjolnir.run --lane c         # backend hammer only
    python -m tools.mjolnir.run --scenario uniform-fast   # one scenario
    python -m tools.mjolnir.run --browsers chromium       # Lane B subset

Writes JSON results to tools/mjolnir/output/run-<timestamp>.json (gitignored).
"""
import argparse
import asyncio
import datetime
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .lane_a import run_lane_a
from .lane_b import run_lane_b
from .lane_c import run_lane_c

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def _render_summary(results: list, console: Console):
    """Pretty-print results as a table grouped by lane."""
    by_lane = {}
    for r in results:
        lane = r["lane"]
        if lane == "b":
            key = f"b-{r.get('browser', '?')}"
        else:
            key = lane
        by_lane.setdefault(key, []).append(r)

    for lane, lane_results in sorted(by_lane.items()):
        table = Table(title=f"Lane {lane.upper()}", show_lines=False)
        table.add_column("Scenario", style="cyan", no_wrap=True)
        table.add_column("Status", justify="center")
        table.add_column("Detail", style="dim")
        for r in lane_results:
            scenario = r["scenario"]
            if r["ok"]:
                status = "[green]✓ PASS[/green]"
            else:
                status = "[red]✗ FAIL[/red]"
            if r.get("error"):
                detail = f"error: {r['error']}"
            else:
                bits = []
                if "instances" in r:
                    bits.append(f"inst={r['instances']}")
                if "max_concurrent" in r:
                    bits.append(f"max_conc={r['max_concurrent']}")
                if "aborts" in r:
                    bits.append(f"aborts={r['aborts']}")
                if "errored" in r:
                    bits.append(f"errored={r['errored']}")
                # Failed invariants
                bad = [k for k, v in r.get("invariants", {}).items() if not v["ok"]]
                if bad:
                    bits.append(f"failed: {', '.join(bad)}")
                detail = "  ".join(bits)
            table.add_row(scenario, status, detail)
        console.print(table)

    # Aggregate
    total = len(results)
    passed = sum(1 for r in results if r["ok"])
    color = "green" if passed == total else "yellow" if passed > 0 else "red"
    console.print(
        f"\n[bold {color}]Overall: {passed}/{total} passed.[/bold {color}]"
    )


def _save_json(results: list) -> Path:
    """Write results to output/ as timestamped JSON."""
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = OUTPUT_DIR / f"run-{ts}.json"
    # Strip non-serializable bits
    serializable = []
    for r in results:
        clean = {k: v for k, v in r.items()}
        if "sub_results" in clean:
            # Already JSON-safe but pass through json.dumps to catch issues
            try:
                json.dumps(clean["sub_results"])
            except TypeError:
                clean["sub_results"] = str(clean["sub_results"])
        serializable.append(clean)
    path.write_text(json.dumps(serializable, indent=2, default=str))
    return path


async def main():
    parser = argparse.ArgumentParser(description="Mjolnir — TTS streaming hammer")
    parser.add_argument("--lane", choices=["a", "b", "c", "all"], default="all",
                        help="which lane(s) to run")
    parser.add_argument("--scenario", default=None,
                        help="run only one scenario by name")
    parser.add_argument("--browsers", default=None,
                        help="comma-separated subset for Lane B "
                             "(chromium,firefox,brave). Default: all available.")
    parser.add_argument("--no-autoplay-bypass", action="store_true",
                        help="Lane B: launch browsers WITHOUT the autoplay bypass "
                             "flag. Tests real MEI / transient-activation rules. "
                             "Use to reproduce the autoplay bug class.")
    args = parser.parse_args()

    console = Console()
    console.rule("[bold]Mjolnir TTS Streaming Hammer[/bold]")

    all_results = []
    browsers = args.browsers.split(",") if args.browsers else None

    if args.lane in ("a", "all"):
        console.print("\n[bold cyan]── Lane A: mocked-Audio scenarios ──[/bold cyan]")
        all_results.extend(await run_lane_a(only=args.scenario))

    if args.lane in ("b", "all"):
        console.print("\n[bold cyan]── Lane B: real-browser scenarios ──[/bold cyan]")
        all_results.extend(await run_lane_b(
            only=args.scenario, browsers=browsers,
            no_autoplay_bypass=args.no_autoplay_bypass,
        ))

    if args.lane in ("c", "all"):
        console.print("\n[bold cyan]── Lane C: backend hammer ──[/bold cyan]")
        all_results.extend(await run_lane_c(only=args.scenario))

    console.print()
    console.rule("[bold]Results[/bold]")
    _render_summary(all_results, console)

    out_path = _save_json(all_results)
    console.print(f"\n[dim]Results saved: {out_path}[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
