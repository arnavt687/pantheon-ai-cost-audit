#!/usr/bin/env python3
"""Pantheon AI Cost Audit.

Turns a startup's LLM API usage into a one-page report of where the bill can
shrink and by roughly how much. Plain Python 3.9+, no dependencies.

Two kinds of input are accepted:

1. A workload sheet (preferred): one row per product feature.
   feature, model, requests_per_month, avg_input_tokens, avg_output_tokens,
   task_type, latency_sensitive, static_prefix_tokens, cached_input_share,
   monthly_cost
   Only feature, model, requests_per_month, avg_input_tokens and
   avg_output_tokens are required. See sample/workload.csv.

2. A raw usage export from the OpenAI or Anthropic console (CSV). Columns are
   detected by name. Usage is grouped by model and scaled to one month. Since
   an export says nothing about what each call does, only the safest levers
   are estimated from it.

Usage:
  python3 audit.py sample/workload.csv --company "Acme AI" --out acme_report.html

Nothing leaves the machine: no network calls, no API keys.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
TIER_RANK = {"small": 0, "mid": 1, "large": 2, "frontier": 3}

# What class of model a task usually needs. Anything cheaper than this is not
# suggested; anything at or above it is suggested only with a quality test.
TASK_TIER = {
    **dict.fromkeys(["classification", "classify", "extraction", "extract", "routing", "router",
                     "moderation", "tagging", "formatting", "parsing", "labeling", "intent"], "small"),
    **dict.fromkeys(["summarization", "summary", "chat", "support", "writing", "rag", "qa",
                     "translation", "rewrite", "email", "content", "transcript"], "mid"),
    **dict.fromkeys(["reasoning", "code", "coding", "agent", "planning", "research", "analysis",
                     "math", "legal", "medical"], "large"),
}

MIN_CACHE_PREFIX = 1024      # both providers only cache prefixes at least this long
HIT_RATE = {"low": 0.70, "high": 0.90}
BATCH_DISCOUNT = 0.5


def load_pricing(path: Path | None = None) -> dict:
    return json.loads((path or HERE / "pricing.json").read_text())


def normalize_model(raw: str, models: dict) -> str | None:
    """Map 'claude-sonnet-4-6-20260115' or 'gpt-4o-2024-08-06' to a pricing key."""
    s = (raw or "").strip().lower().replace("_", "-")
    if s in models:
        return s
    s = re.sub(r"[-@](\d{8}|\d{4}-\d{2}-\d{2})$", "", s)   # dated snapshot suffix
    s = re.sub(r"-(latest|preview)$", "", s)
    if s in models:
        return s
    # longest pricing key that prefixes the name, e.g. 'gpt-4o-mini-audio' -> 'gpt-4o-mini'
    hits = [k for k in models if s.startswith(k)]
    return max(hits, key=len) if hits else None


def num(v, default=0.0) -> float:
    if v is None:
        return default
    s = str(v).replace(",", "").replace("$", "").strip()
    if s == "":
        return default
    try:
        return float(s)
    except ValueError:
        return default


def yes(v) -> bool | None:
    s = str(v or "").strip().lower()
    if s in {"y", "yes", "true", "1"}:
        return True
    if s in {"n", "no", "false", "0"}:
        return False
    return None


@dataclass
class Workload:
    feature: str
    model_raw: str
    model: str | None
    requests: float                 # per month
    avg_in: float                   # all input tokens per request, cached or not
    avg_out: float
    task_type: str = ""
    latency_sensitive: bool | None = None
    static_prefix: float | None = None
    cached_share: float = 0.0       # share of input already served from cache today
    cache_write_share: float = 0.0  # Anthropic: share of input written to cache today
    reported_cost: float | None = None
    from_export: bool = False
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- reading input

def _find(headers, *patterns):
    for p in patterns:
        for h in headers:
            if re.fullmatch(p, h.strip().lower()):
                return h
    return None


def read_input(path: Path, models: dict) -> tuple[list[Workload], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} has no rows")
    headers = list(rows[0].keys())
    if _find(headers, r"feature") and _find(headers, r"requests?_per_month"):
        return _read_workload(rows, headers, models), []
    return _read_export(rows, headers, models)


def _read_workload(rows, headers, models) -> list[Workload]:
    out = []
    for r in rows:
        if not (r.get("feature") or "").strip():
            continue
        prefix = r.get("static_prefix_tokens")
        w = Workload(
            feature=r["feature"].strip(),
            model_raw=r.get("model", ""),
            model=normalize_model(r.get("model", ""), models),
            requests=num(r.get("requests_per_month")),
            avg_in=num(r.get("avg_input_tokens")),
            avg_out=num(r.get("avg_output_tokens")),
            task_type=(r.get("task_type") or "").strip().lower(),
            latency_sensitive=yes(r.get("latency_sensitive")),
            static_prefix=num(prefix) if str(prefix or "").strip() else None,
            cached_share=min(max(num(r.get("cached_input_share")), 0.0), 1.0),
            reported_cost=num(r.get("monthly_cost")) if str(r.get("monthly_cost") or "").strip() else None,
        )
        out.append(w)
    return out


def _read_export(rows, headers, models):
    h_model = _find(headers, r"model", r"model_?name", r"model_?id")
    h_in = _find(headers, r"(uncached_)?input_tokens", r"n_context_tokens_total", r"prompt_tokens",
                 r"input_tokens_uncached", r"usage_input_tokens_no_cache")
    h_cached = _find(headers, r"input_cached_tokens", r"cached_(input_)?tokens",
                     r"cache_read_input_tokens", r"usage_input_tokens_cache_read")
    h_write = _find(headers, r"cache_creation_input_tokens", r"cache_write_tokens",
                    r"usage_input_tokens_cache_write(_5m|_1h)?")
    h_out = _find(headers, r"output_tokens", r"n_generated_tokens_total", r"completion_tokens",
                  r"usage_output_tokens")
    h_req = _find(headers, r"num_model_requests", r"n_requests", r"requests?", r"request_count")
    h_cost = _find(headers, r"cost", r"cost_usd", r"amount", r"amount_usd", r"amount_value")
    h_date = _find(headers, r"start_time(_iso)?", r"date", r"usage_date(_utc)?", r"day", r"timestamp")
    h_batch = _find(headers, r"batch", r"is_batch", r"service_tier")
    if not (h_model and h_in and h_out):
        raise SystemExit("Couldn't recognise this file. Expected a workload sheet (see sample/workload.csv) "
                         f"or a usage export with model/input/output token columns. Headers: {headers}")

    warnings, agg, days = [], {}, set()
    for r in rows:
        name = (r.get(h_model) or "").strip()
        if not name:
            continue
        a = agg.setdefault(name, {"in": 0.0, "cached": 0.0, "write": 0.0, "out": 0.0, "req": 0.0,
                                  "cost": 0.0, "batch_in": 0.0})
        tin = num(r.get(h_in))
        a["in"] += tin
        a["cached"] += num(r.get(h_cached)) if h_cached else 0
        a["write"] += num(r.get(h_write)) if h_write else 0
        a["out"] += num(r.get(h_out))
        a["req"] += num(r.get(h_req)) if h_req else 0
        a["cost"] += num(r.get(h_cost)) if h_cost else 0
        if h_batch and str(r.get(h_batch)).strip().lower() in {"true", "1", "yes", "batch"}:
            a["batch_in"] += tin
        if h_date and r.get(h_date):
            d = _parse_day(r[h_date])
            if d:
                days.add(d)

    span_days = (max(days) - min(days)).days + 1 if days else 30
    scale = 30.4 / span_days
    if not days:
        warnings.append("No date column found; assumed the export covers one month.")
    elif span_days < 7:
        warnings.append(f"Export covers only {span_days} day(s); monthly figures are a rough extrapolation.")

    out = []
    for name, a in agg.items():
        # OpenAI exports count cached tokens inside input_tokens; Anthropic lists them separately.
        anth = "claude" in name.lower()
        total_in = a["in"] + (a["cached"] + a["write"] if anth else 0)
        req = a["req"] or None
        w = Workload(
            feature=f"All traffic on {name}",
            model_raw=name,
            model=normalize_model(name, models),
            requests=(req or 1) * scale,
            avg_in=total_in / (req or 1),
            avg_out=a["out"] / (req or 1),
            cached_share=(a["cached"] / total_in) if total_in else 0.0,
            cache_write_share=(a["write"] / total_in) if total_in else 0.0,
            reported_cost=a["cost"] * scale if h_cost else None,
            from_export=True,
        )
        if not req:
            w.notes.append("No request counts in export; per-request figures are totals.")
        if a["batch_in"]:
            w.notes.append("Some of this traffic already uses Batch.")
            w.latency_sensitive = True  # don't double-count the batch lever
        out.append(w)
    return out, warnings


def _parse_day(s):
    s = str(s).strip()
    if re.fullmatch(r"\d{9,11}(\.\d+)?", s):
        return datetime.utcfromtimestamp(float(s)).date()
    if re.match(r"\d{4}-\d{2}-\d{2}", s):
        return date.fromisoformat(s[:10])
    try:
        return datetime.strptime(s.split()[0], "%m/%d/%Y").date()
    except ValueError:
        return None


# ---------------------------------------------------------------- cost math

def cost(w: Workload, model_key: str, models: dict, *, cached_share=None, write_share=None,
         batch=False) -> float:
    """Monthly USD for workload w priced on model_key."""
    p = models[model_key]
    cs = w.cached_share if cached_share is None else cached_share
    ws = w.cache_write_share if write_share is None else write_share
    if p["provider"] != "anthropic":
        ws = 0.0
    fresh = max(1.0 - cs - ws, 0.0)
    per_req_in = w.avg_in * (fresh * p["input"] + cs * p["cached"] + ws * p.get("cache_write", p["input"]))
    per_req = (per_req_in + w.avg_out * p["output"]) / 1e6
    total = per_req * w.requests
    return total * (BATCH_DISCOUNT if batch else 1.0)


def required_tier(w: Workload, models: dict) -> str | None:
    t = w.task_type
    for key, tier in TASK_TIER.items():
        if t and (t == key or t.startswith(key)):
            return tier
    return None


@dataclass
class Lever:
    kind: str           # model | cache | batch | output
    title: str
    detail: str
    low: float          # monthly USD saved, conservative
    high: float         # monthly USD saved, optimistic
    needs_test: bool = False


@dataclass
class Finding:
    w: Workload
    current: float
    levers: list[Lever]
    target_model: str | None = None

    @property
    def low(self):
        return min(sum(l.low for l in self.levers), self.current)

    @property
    def high(self):
        return min(sum(l.high for l in self.levers), self.current)


def analyse(w: Workload, models: dict) -> Finding:
    if not w.model:
        w.notes.append(f"No price on file for '{w.model_raw}'; add it to pricing.json.")
        return Finding(w, w.reported_cost or 0.0, [])
    cur_key = w.model
    cur = models[cur_key]
    current = cost(w, cur_key, models)
    if w.reported_cost and current and abs(w.reported_cost - current) / current > 0.15:
        w.notes.append(f"Token math gives ${current:,.0f}/mo but the bill says ${w.reported_cost:,.0f}; "
                       "savings are scaled from token math.")
    levers: list[Lever] = []

    # 1. Model choice ------------------------------------------------------
    need = required_tier(w, models)
    floor = need or cur["tier"]
    candidates = [k for k, m in models.items()
                  if m["provider"] == cur["provider"] and m["current"] and m["verified"]
                  and TIER_RANK[m["tier"]] >= TIER_RANK[floor]
                  and TIER_RANK[m["tier"]] <= TIER_RANK[cur["tier"]]]
    by_cost = lambda k: cost(w, k, models)
    best = min(candidates, key=by_cost, default=None)
    same = min((k for k in candidates if models[k]["tier"] == cur["tier"]), key=by_cost, default=None)
    after_model_low = after_model_high = cur_key
    save_same = current - by_cost(same) if same else 0.0
    save_best = current - by_cost(best) if best else 0.0
    if save_same > 0.5:
        after_model_low = same
    if save_best > 0.5:
        after_model_high = best
    if save_best > 0.5 and best == same:
        levers.append(Lever("model", f"Move {cur_key} → {best}",
                            "Newer model in the same class at a lower price. Run your own eval before "
                            "switching; behaviour can change between versions.",
                            save_same, save_same))
    elif save_best > 0.5:
        first = (f"Move {cur_key} → {same} now (same class, lower price). " if save_same > 0.5 else "")
        levers.append(Lever("model", f"Try {best} instead of {cur_key}",
                            first + f"'{w.task_type}' work is often handled well by a {models[best]['tier']} "
                            f"model; {best} is counted only in the optimistic estimate until it passes a "
                            "quality test on your own examples.",
                            max(save_same, 0.0), save_best, needs_test=True))

    # 2. Prompt caching ----------------------------------------------------
    def cache_saving(model_key, scenario):
        p = models[model_key]
        fresh_share = max(1 - w.cached_share - w.cache_write_share, 0)
        fresh_tokens = w.avg_in * fresh_share
        prefix = w.static_prefix
        if prefix is None:
            prefix = 0.4 * w.avg_in if (scenario == "high" and w.avg_in >= 2 * MIN_CACHE_PREFIX) else 0
        prefix = min(prefix, fresh_tokens)
        if prefix < MIN_CACHE_PREFIX:
            return 0.0
        hit = HIT_RATE[scenario]
        write_premium = (p.get("cache_write", p["input"]) - p["input"]) if p["provider"] == "anthropic" else 0
        per_req = prefix * (hit * (p["input"] - p["cached"]) - (1 - hit) * write_premium) / 1e6
        return max(per_req * w.requests, 0.0)

    c_low, c_high = cache_saving(after_model_low, "low"), cache_saving(after_model_high, "high")
    if c_high > 0.5:
        known = w.static_prefix is not None
        levers.append(Lever("cache", "Cache the repeated prompt prefix",
                            ("Put the fixed part (system prompt, tool definitions, reference docs) first and "
                             "mark it cacheable. " if cur["provider"] == "anthropic" else
                             "OpenAI caches automatically once the fixed part comes first and is ≥1,024 tokens; "
                             "move anything that changes per request (dates, user names) after it. ")
                            + ("" if known else "Estimate assumes ~40% of input is fixed; a sample prompt "
                               "would firm this up."),
                            c_low, c_high))

    # 3. Batch -------------------------------------------------------------
    if w.latency_sensitive is False:
        remaining_low = current - sum(l.low for l in levers)
        remaining_high = current - sum(l.high for l in levers)
        levers.append(Lever("batch", "Send through the Batch API",
                            "Nobody is waiting on these calls, so they can run within 24 hours at half price.",
                            remaining_low * BATCH_DISCOUNT, remaining_high * BATCH_DISCOUNT))
    elif w.latency_sensitive is None and not w.from_export:
        w.notes.append("Tell us if any of this can wait a few hours: Batch would halve its cost.")

    # 4. Output length (flag only) ----------------------------------------
    if need == "small" and w.avg_out > 400:
        levers.append(Lever("output", "Shorten outputs",
                            f"{w.avg_out:,.0f} output tokens per call is long for {w.task_type}. Ask for JSON "
                            "with only the needed fields and set max_tokens; output costs 5× input.",
                            0.0, 0.0))

    scale = (w.reported_cost / current) if (w.reported_cost and current) else 1.0
    if scale != 1.0:
        for l in levers:
            l.low *= scale
            l.high *= scale
        current = w.reported_cost
    return Finding(w, current, levers, best if best != cur_key else None)


# ---------------------------------------------------------------- report

def money(x: float) -> str:
    return f"${x:,.0f}"


def _lever_amount(l: Lever) -> str:
    a, b = sorted((l.low, l.high))
    if b < 0.5:
        return ""
    text = money(b) if round(a) == round(b) else f"{money(a)}–{money(b)}"
    return f"<span class=amt>{text}/mo</span>"


def render(company: str, findings: list[Finding], warnings: list[str], pricing: dict, prepared: date) -> str:
    total = sum(f.current for f in findings)
    low = sum(f.low for f in findings)
    high = sum(f.high for f in findings)
    pct = lambda x: f"{(x / total * 100):.0f}%" if total else "—"
    findings = sorted(findings, key=lambda f: -f.high)

    rows = []
    for f in findings:
        lever_html = "".join(
            f'<li><b>{html.escape(l.title)}</b>'
            f'{" <span class=tag>needs quality test</span>" if l.needs_test else ""}'
            f'<br><span class=muted>{html.escape(l.detail)}</span>'
            + _lever_amount(l)
            + "</li>" for l in f.levers) or '<li class=muted>Nothing obvious from the numbers alone.</li>'
        notes = "".join(f"<li>{html.escape(n)}</li>" for n in f.w.notes)
        rows.append(f"""
      <tr>
        <td><b>{html.escape(f.w.feature)}</b><br><span class=muted>{html.escape(f.w.model_raw)}
          · {f.w.requests:,.0f} calls/mo · {f.w.avg_in:,.0f} in / {f.w.avg_out:,.0f} out tokens</span>
          {f'<ul class="muted small">{notes}</ul>' if notes else ''}</td>
        <td class=num>{money(f.current)}</td>
        <td><ul class=levers>{lever_html}</ul></td>
        <td class=num><b>{money(f.low)}–{money(f.high)}</b></td>
      </tr>""")

    warn_html = "".join(f"<li>{html.escape(w)}</li>" for w in warnings)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Cost Audit · {html.escape(company)}</title>
<style>
:root{{--ink:#101828;--muted:#667085;--line:#e4e7ec;--brand:#3b2f8f;--brand-bg:#f4f3ff;--good:#067647;--good-bg:#ecfdf3;--warn:#b54708;--warn-bg:#fffaeb}}
*{{box-sizing:border-box}} body{{margin:0;color:var(--ink);font:14px/1.5 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;background:#fff}}
main{{max-width:1040px;margin:0 auto;padding:32px 16px 48px}}
.brand{{display:flex;align-items:center;gap:10px;color:var(--brand);font-weight:700;letter-spacing:.02em}}
.mark{{width:26px;height:26px;border-radius:7px;background:var(--brand);color:#fff;display:grid;place-items:center;font-size:14px}}
h1{{font-size:26px;margin:14px 0 4px}} h2{{font-size:16px;margin:32px 0 10px}}
.sub,.muted{{color:var(--muted)}} .small{{font-size:12px;margin:6px 0 0;padding-left:16px}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-top:20px}}
.tile{{border:1px solid var(--line);border-radius:12px;padding:16px}}
.tile b{{display:block;font-size:28px;line-height:1.15;font-variant-numeric:tabular-nums}}
.tile.good{{background:var(--good-bg);border-color:#abefc6}} .tile.good b{{color:var(--good)}}
.table-wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:12px}}
table{{border-collapse:collapse;width:100%;min-width:820px}}
th,td{{padding:12px;border-bottom:1px solid var(--line);vertical-align:top;text-align:left}}
th{{font-size:12px;text-transform:uppercase;letter-spacing:.03em;color:var(--muted);background:#f9fafb}}
tr:last-child td{{border-bottom:none}} .num{{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}}
ul.levers{{margin:0;padding-left:16px}} ul.levers li{{margin-bottom:8px}}
.amt{{display:inline-block;margin-left:6px;color:var(--good);font-weight:600;font-size:12px}}
.tag{{display:inline-block;padding:1px 7px;border-radius:99px;background:var(--warn-bg);color:var(--warn);font-size:11px;font-weight:600}}
.box{{border:1px solid var(--line);border-radius:12px;padding:14px 18px;background:#fcfcfd}}
footer{{margin-top:36px;padding-top:14px;border-top:1px solid var(--line);color:var(--muted);font-size:12px}}
@media print{{main{{padding:0}} .table-wrap{{border:none}} tr{{break-inside:avoid}}}}
</style></head><body><main>
<div class="brand"><span class="mark">P</span>PANTHEON SOLUTIONS</div>
<h1>AI API Cost Audit</h1>
<p class="sub">{html.escape(company)} · prepared {prepared.strftime("%B %-d, %Y")} · prices as of {pricing["as_of"]}</p>

<div class="tiles">
  <div class="tile"><b>{money(total)}</b><span class="muted">estimated monthly LLM spend</span></div>
  <div class="tile good"><b>{money(low)}–{money(high)}</b><span class="muted">estimated monthly savings ({pct(low)}–{pct(high)})</span></div>
  <div class="tile"><b>{money(low * 12)}–{money(high * 12)}</b><span class="muted">per year at today's volume</span></div>
</div>

<h2>Where the money goes, and what to change</h2>
<div class="table-wrap"><table>
<thead><tr><th>Workload</th><th class=num>Now / mo</th><th>Changes</th><th class=num>Saves / mo</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>

<h2>How to read the range</h2>
<div class="box"><ul>
<li><b>Low end</b> counts only the safer changes: newer models of the same class at a lower price (still worth a quick check), caching of prompt parts you've told us are fixed, and Batch for work nobody waits on.</li>
<li><b>High end</b> adds moves to a smaller model and caching we estimated without seeing your prompts. Those are marked <span class=tag>needs quality test</span>: we run 50–200 of your real examples through both models and you compare the answers before anything changes.</li>
<li>Costs are recomputed from token counts at list prices. Discounts, credits and committed-use deals aren't included.</li>
</ul></div>
{f'<h2>Data notes</h2><ul class=muted>{warn_html}</ul>' if warnings else ''}

<h2>Next step</h2>
<div class="box">Pick the one or two biggest rows above. We'll run the quality test on a sample you choose (with personal data removed) and send a side-by-side comparison. You decide what ships; nothing touches your production code or keys.</div>

<footer>Pantheon Solutions · AI cost audit. We never ask for API keys or customer data. Usage files are used only to prepare this report and deleted afterwards.</footer>
</main></body></html>"""


def write_csv(path: Path, findings: list[Finding]):
    with path.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["feature", "model", "current_monthly", "lever", "title", "save_low", "save_high", "needs_test"])
        for fd in findings:
            for l in fd.levers or [Lever("none", "", "", 0, 0)]:
                wr.writerow([fd.w.feature, fd.w.model_raw, round(fd.current, 2), l.kind, l.title,
                             round(l.low, 2), round(l.high, 2), l.needs_test])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("usage_csv", type=Path)
    ap.add_argument("--company", default="Your company")
    ap.add_argument("--out", type=Path, default=Path("ai_cost_audit.html"))
    ap.add_argument("--pricing", type=Path)
    a = ap.parse_args(argv)

    pricing = load_pricing(a.pricing)
    models = pricing["models"]
    workloads, warnings = read_input(a.usage_csv, models)
    findings = [analyse(w, models) for w in workloads]
    if any(not models.get(w.model or "", {}).get("verified", True) for w in workloads):
        warnings.append("Some legacy OpenAI prices come from a third-party tracker; check them against your bill.")
    a.out.write_text(render(a.company, findings, warnings, pricing, date.today()), encoding="utf-8")
    csv_path = a.out.with_suffix(".csv")
    write_csv(csv_path, findings)
    total = sum(f.current for f in findings)
    print(f"Wrote {a.out} and {csv_path}: spend ${total:,.0f}/mo, savings "
          f"${sum(f.low for f in findings):,.0f}–${sum(f.high for f in findings):,.0f}/mo")
    for w in warnings:
        print("note:", w)


if __name__ == "__main__":
    main()
