#!/usr/bin/env python3
"""Side-by-side quality check: does a cheaper model give the same answers?

Runs the same sample prompts through the current model and a candidate model
and writes an HTML page showing both answers next to each other, the
exact-match rate when an expected answer is given, and the cost of each run.

Meant to be run by the client with their own key, or by Pantheon on a
scrubbed sample the client chose (no personal data).

samples.jsonl, one object per line:
  {"id": "t1", "system": "You tag support tickets...", "input": "My card was charged twice", "expected": "billing"}
"system" and "expected" are optional.

Usage:
  ANTHROPIC_API_KEY=... python3 quality_check.py samples.jsonl \
      --current claude-sonnet-4-6 --candidate claude-haiku-4-5 --out compare.html
  python3 quality_check.py sample/samples.jsonl --current a --candidate b --dry-run
"""
from __future__ import annotations

import argparse
import html
import json
import os
import time
import urllib.request
from pathlib import Path

from audit import load_pricing, normalize_model


def call(model: str, provider: str, system: str, user: str, max_tokens: int) -> tuple[str, int, int]:
    if provider == "anthropic":
        body = {"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": user}]}
        if system:
            body["system"] = system
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", json.dumps(body).encode(), {
            "x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
            "content-type": "application/json"})
        data = json.load(urllib.request.urlopen(req, timeout=120))
        text = "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
        return text, data["usage"]["input_tokens"], data["usage"]["output_tokens"]
    body = {"model": model, "input": user, "max_output_tokens": max_tokens}
    if system:
        body["instructions"] = system
    req = urllib.request.Request("https://api.openai.com/v1/responses", json.dumps(body).encode(), {
        "authorization": f"Bearer {os.environ['OPENAI_API_KEY']}", "content-type": "application/json"})
    data = json.load(urllib.request.urlopen(req, timeout=120))
    text = "".join(c.get("text", "") for o in data.get("output", []) if o.get("type") == "message"
                   for c in o.get("content", []))
    return text, data["usage"]["input_tokens"], data["usage"]["output_tokens"]


def norm(s: str) -> str:
    return " ".join(s.strip().strip(".").lower().split())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("samples", type=Path)
    ap.add_argument("--current", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("quality_check.html"))
    ap.add_argument("--dry-run", action="store_true", help="fake answers; tests the pipeline without a key")
    a = ap.parse_args(argv)

    models = load_pricing()["models"]
    samples = [json.loads(l) for l in a.samples.read_text().splitlines() if l.strip()][: a.limit]
    runs = {}
    for label, name in (("current", a.current), ("candidate", a.candidate)):
        key = normalize_model(name, models)
        provider = models[key]["provider"] if key else ("anthropic" if "claude" in name else "openai")
        out, tin, tout = [], 0, 0
        for s in samples:
            if a.dry_run:
                ans = s.get("expected") or f"[{label} answer to {s['id']}]"
                ans, i, o = (ans if hash((label, s["id"])) % 7 else ans + " (differs)"), 200, 20
            else:
                for attempt in range(4):
                    try:
                        ans, i, o = call(name, provider, s.get("system", ""), s["input"], a.max_tokens)
                        break
                    except Exception as e:  # rate limits, timeouts
                        if attempt == 3:
                            ans, i, o = f"[error: {e}]", 0, 0
                        time.sleep(2 ** attempt)
            out.append(ans)
            tin, tout = tin + i, tout + o
        price = models.get(key or "", {})
        usd = (tin * price.get("input", 0) + tout * price.get("output", 0)) / 1e6
        runs[label] = {"model": name, "answers": out, "usd": usd}

    cur, cand = runs["current"]["answers"], runs["candidate"]["answers"]
    agree = sum(norm(x) == norm(y) for x, y in zip(cur, cand))
    with_exp = [(s, x, y) for s, x, y in zip(samples, cur, cand) if s.get("expected")]
    acc = lambda i: sum(norm(t[i]) == norm(t[0]["expected"]) for t in with_exp)
    rows = "".join(
        f"<tr class='{'same' if norm(x) == norm(y) else 'diff'}'><td>{html.escape(s['id'])}</td>"
        f"<td><pre>{html.escape(s['input'][:600])}</pre></td>"
        f"<td><pre>{html.escape(x)}</pre></td><td><pre>{html.escape(y)}</pre></td>"
        f"<td>{html.escape(str(s.get('expected', '')))}</td></tr>"
        for s, x, y in zip(samples, cur, cand))
    n = len(samples)
    summary = [f"{agree}/{n} answers identical after normalising case and spacing"]
    if with_exp:
        summary.append(f"accuracy vs expected: {a.current} {acc(1)}/{len(with_exp)}, "
                       f"{a.candidate} {acc(2)}/{len(with_exp)}")
    summary.append(f"cost of this run: {a.current} ${runs['current']['usd']:.4f}, "
                   f"{a.candidate} ${runs['candidate']['usd']:.4f}")
    if a.dry_run:
        summary.append("DRY RUN: answers are fake")
    a.out.write_text(f"""<!doctype html><meta charset=utf-8><title>Quality check</title>
<style>body{{font:14px/1.45 system-ui,sans-serif;margin:24px;color:#101828}} table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #e4e7ec;padding:8px;vertical-align:top;text-align:left}} pre{{white-space:pre-wrap;margin:0;font:13px/1.4 ui-monospace,monospace}}
tr.diff td{{background:#fffaeb}} .muted{{color:#667085}}</style>
<h1>Quality check: {html.escape(a.current)} vs {html.escape(a.candidate)}</h1>
<ul>{''.join(f'<li>{html.escape(x)}</li>' for x in summary)}</ul>
<p class=muted>Rows where the answers differ are highlighted. Different wording isn't necessarily worse; read them.</p>
<table><tr><th>id</th><th>input</th><th>{html.escape(a.current)}</th><th>{html.escape(a.candidate)}</th><th>expected</th></tr>{rows}</table>""",
                     encoding="utf-8")
    print(f"Wrote {a.out}: " + "; ".join(summary))


if __name__ == "__main__":
    main()
