"""Metered LLM signal-extraction test (CLAUDE.md Phase-0 homework).

Two modes:

  * OFFLINE SELF-TEST (always runs, $0): proves the validation layer accepts a
    well-formed signal and REJECTS out-of-range / malformed model output. This is
    the "never present an unvalidated AI number as fact" guard.

  * LIVE METERED TEST (only with --live AND ANTHROPIC_API_KEY set): runs a few
    short transcripts through Claude N times each, and reports (a) output
    CONSISTENCY across re-runs (std of each numeric field) and (b) TOKEN COST.
    Deliberately tiny by default (3 transcripts x 3 runs = 9 calls) so you can
    measure cost/consistency before ever running it at universe scale.

Usage:
    python examples/llm_signal_test.py                 # offline self-test only
    set ANTHROPIC_API_KEY=...                           # (PowerShell: $env:ANTHROPIC_API_KEY="...")
    python examples/llm_signal_test.py --live --runs 3
"""
from __future__ import annotations

import argparse
import os
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pydantic import ValidationError  # noqa: E402
from research.signals_llm import (  # noqa: E402
    PROMPT_TEMPLATE, parse_signal, validate_signal,
)

# Approximate Sonnet pricing (USD per token) — adjust to current rate card.
PRICE_IN = 3.0 / 1_000_000
PRICE_OUT = 15.0 / 1_000_000

SAMPLES = [
    ("clearly_positive",
     "CEO: We are raising full-year revenue guidance to $4.2-4.3B from $3.9B. "
     "Bookings accelerated 28% and we are seeing the strongest demand in company history. "
     "Margins expanded 200bps and we are confident heading into next year."),
    ("clearly_negative",
     "CFO: We are withdrawing prior guidance given deteriorating macro conditions. "
     "Orders fell 19% sequentially, we are restructuring, and visibility is extremely limited. "
     "We expect margin pressure to persist through the year."),
    ("mixed",
     "Management: Revenue was in line with expectations. We reaffirm full-year guidance. "
     "Some end-markets softened while others held up; we remain cautiously optimistic and are "
     "monitoring the environment closely."),
]


def offline_self_test() -> bool:
    print("=== Offline validation self-test ($0, no API) ===")
    ok = True

    good = '{"guidance_change": 0.4, "tone": 0.6, "reason": "raised FY guidance on strong bookings"}'
    sig = parse_signal(good)
    print(f"  accept well-formed:        OK  -> {sig.guidance_change:+.2f} / {sig.tone:+.2f}")

    cases = {
        "reject out-of-range [-1,1]": ({"guidance_change": 2.5, "tone": 0.1, "reason": "x"}, ValidationError),
        "reject missing field":       ({"tone": 0.1, "reason": "x"}, ValidationError),
        "reject empty reason":        ({"guidance_change": 0.1, "tone": 0.1, "reason": ""}, ValidationError),
    }
    for label, (payload, exc) in cases.items():
        try:
            validate_signal(payload)
            print(f"  {label:<27} FAIL (was accepted!)")
            ok = False
        except exc:
            print(f"  {label:<27} OK  (rejected)")

    try:
        parse_signal("the stock looks great, strong buy")
        print("  reject non-JSON output:    FAIL (was accepted!)")
        ok = False
    except ValueError:
        print("  reject non-JSON output:    OK  (rejected)")

    print(f"  -> validation guard {'PASSED' if ok else 'FAILED'}\n")
    return ok


def live_metered_test(runs: int) -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("=== Live metered test SKIPPED: ANTHROPIC_API_KEY not set ===")
        print("  PowerShell:  $env:ANTHROPIC_API_KEY = \"sk-ant-...\"")
        print("  then:        python examples/llm_signal_test.py --live\n")
        return

    import anthropic
    client = anthropic.Anthropic()
    model = "claude-sonnet-4-6"
    tot_in = tot_out = 0

    print(f"=== Live metered test: {len(SAMPLES)} transcripts x {runs} runs ({model}) ===")
    for name, text in SAMPLES:
        gcs, tones = [], []
        for _ in range(runs):
            msg = client.messages.create(
                model=model, max_tokens=300,
                messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(text=text)}],
            )
            tot_in += msg.usage.input_tokens
            tot_out += msg.usage.output_tokens
            sig = parse_signal(msg.content[0].text)  # validated or it raises
            gcs.append(sig.guidance_change)
            tones.append(sig.tone)
        gc_sd = statistics.pstdev(gcs) if len(gcs) > 1 else 0.0
        tn_sd = statistics.pstdev(tones) if len(tones) > 1 else 0.0
        print(f"  {name:<17} guidance={statistics.fmean(gcs):+.2f} (sd {gc_sd:.2f})  "
              f"tone={statistics.fmean(tones):+.2f} (sd {tn_sd:.2f})  consistency="
              f"{'GOOD' if max(gc_sd, tn_sd) < 0.15 else 'NOISY'}")

    cost = tot_in * PRICE_IN + tot_out * PRICE_OUT
    print(f"\n  tokens: {tot_in:,} in / {tot_out:,} out over {len(SAMPLES) * runs} calls")
    print(f"  approx cost: ${cost:.4f}  (~${cost / (len(SAMPLES) * runs):.4f}/call)")
    print("  -> cache repeated transcripts in production for ~90% input-cost savings.\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="make real (paid) API calls")
    ap.add_argument("--runs", type=int, default=3, help="re-runs per transcript (consistency)")
    args = ap.parse_args()

    offline_self_test()
    if args.live:
        live_metered_test(args.runs)
    else:
        print("(omit --live to stay offline; pass --live with ANTHROPIC_API_KEY for the metered test)")


if __name__ == "__main__":
    main()
