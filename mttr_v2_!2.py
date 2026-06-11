"""
mttr_v2.py  —  V3 pipeline (deterministic-first)
======================================================
Why this rewrite:

The old V2 pipeline relied on a 3B LLM to "present the data exactly as
shown, follow this exact format, never hallucinate, never contradict
yourself". Small local models cannot reliably do all of that at once —
which produced contradictory output ("Record 1: ... " followed by
"No matching records found.") and hallucinated "engineering knowledge"
even when 1+ real records existed.

V3 principle: **the LLM never formats database rows**. Formatting
(tables, problem lists, solution lists, default listings, aggregates)
is 100% deterministic Python. The LLM is used ONLY for:
  - troubleshooting / fix recommendations (semantic & hybrid modes)
  - short prose summaries (fed pre-computed stats, not raw rows)
  - general knowledge questions
  - a short, tightly-scoped "engineering note" when truly zero records
    exist (and only when the question is fault/troubleshoot related)

Other fixes:
  - SQL retry ladder is now generic: it can relax ANY condition in the
    WHERE clause (not just specific machine/model patterns), so a
    filter typo or overly-specific model name no longer means "0 rows
    forever".
  - Regex safety nets catch "...in a table", "all problems", "only
    solutions", "summary" etc. even if the planner LLM misclassifies
    output_format.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from datetime import date
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
OLLAMA_URL           = "http://127.0.0.1:11434"
OLLAMA_MODEL_PLANNER = "qwen2.5-coder:7b"
OLLAMA_MODEL_SYNTH   = "llama3.2:3b"
SQLITE_PATH          = "./mttr_records.db"
PAGE_SIZE            = 8
MAX_SYNTH_TOKENS     = 900     # for troubleshoot prose
MAX_SUMMARY_TOKENS   = 350     # for summary prose
MAX_NODATA_TOKENS    = 180     # for the "no records" engineering note
_SYSTEM_COLS         = {"id", "image_b64", "image_name", "image_mime", "chroma_id"}

# ─────────────────────────────────────────────────────────────────────────────
# SAFE SQL RUNNER
# ─────────────────────────────────────────────────────────────────────────────
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|"
    r"pragma|replace|vacuum|reindex|begin|commit|rollback)\b",
    re.IGNORECASE,
)
_ALLOWED_TABLES = re.compile(r"\b(mttr_records|tsg_records)\b", re.IGNORECASE)


def safe_sql(sql: str, limit: int = 300) -> list[dict]:
    """Execute a read-only SELECT. Adds LIMIT only for non-aggregate queries."""
    s = sql.strip().rstrip(";").strip()
    if ";" in s:
        raise ValueError("Multiple statements not allowed.")
    if not re.match(r"^\s*select\b", s, re.IGNORECASE):
        raise ValueError("Only SELECT allowed.")
    if _FORBIDDEN.search(s):
        raise ValueError("Forbidden keyword.")
    if not _ALLOWED_TABLES.search(s):
        raise ValueError("Must query mttr_records or tsg_records.")
    is_aggregate = bool(
        re.search(r"\b(count\s*\(|sum\s*\(|avg\s*\(|group\s+by)\b", s, re.IGNORECASE)
    )
    if not is_aggregate and not re.search(r"\blimit\b", s, re.IGNORECASE):
        s += f" LIMIT {limit}"
    conn = sqlite3.connect(f"file:{SQLITE_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(s).fetchall()]
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# LIVE SCHEMA (what the planner sees — real values, grouped by column)
# ─────────────────────────────────────────────────────────────────────────────
def build_live_schema() -> str:
    lines: list[str] = []
    try:
        conn = sqlite3.connect(f"file:{SQLITE_PATH}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        _SKIP = {"id", "image_b64", "image_mime", "chroma_id", "image_name"}

        for table in ("mttr_records", "tsg_records"):
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                lines.append(f"\nTable: {table}  ({count} rows total)")
                cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
                for col in cols:
                    cname = col[1]
                    if cname in _SKIP:
                        continue
                    try:
                        rows = conn.execute(
                            f"SELECT {cname}, COUNT(*) c FROM {table} "
                            f"WHERE {cname} IS NOT NULL AND CAST({cname} AS TEXT) != '' "
                            f"GROUP BY {cname} ORDER BY c DESC LIMIT 20"
                        ).fetchall()
                        vals = [str(r[0])[:30] for r in rows if r[0] is not None]
                        val_str = (
                            f"  =>  e.g. {', '.join(repr(v) for v in vals[:12])}"
                            + (f"  (+{len(vals)-12} more)" if len(vals) > 12 else "")
                        ) if vals else ""
                    except Exception:
                        val_str = ""
                    lines.append(f"  {cname}{val_str}")
            except Exception:
                continue

        try:
            r = conn.execute(
                "SELECT MIN(iso_date), MAX(iso_date) FROM mttr_records "
                "WHERE iso_date IS NOT NULL AND iso_date != ''"
            ).fetchone()
            if r and r[0]:
                lines.append(f"\nDate range in DB: {r[0]}  to  {r[1]}")
        except Exception:
            pass

        conn.close()
    except Exception as e:
        lines.append(f"(Schema unavailable: {e})")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA CLIENT
# ─────────────────────────────────────────────────────────────────────────────
_http = httpx.AsyncClient(timeout=240.0)


async def call_ollama(prompt: str, model: str, max_tokens: int = 600) -> str:
    try:
        r = await _http.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "30m",
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.05,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                    "num_ctx": 8192,
                },
            },
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except httpx.ConnectError:
        raise HTTPException(503, "Ollama not running. Start with: ollama serve")


# ─────────────────────────────────────────────────────────────────────────────
# PLAN PARSER
# ─────────────────────────────────────────────────────────────────────────────
_PLAN_DEFAULTS: dict = {
    "mode": "semantic",
    "sql": "",
    "semantic_query": "",
    "output_format": "default",
    "wants_chart": False,
    "chart_sql": "",
    "chart_type": "",
    "answer_focus": "",
}


def parse_plan(raw: str) -> dict:
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    m = re.search(r"\{[\s\S]+\}", cleaned)
    if not m:
        return dict(_PLAN_DEFAULTS)
    for attempt in (m.group(0), re.sub(r",\s*([}\]])", r"\1", m.group(0))):
        try:
            plan = json.loads(attempt)
            return {**_PLAN_DEFAULTS, **plan}
        except Exception:
            continue
    return dict(_PLAN_DEFAULTS)


# ─────────────────────────────────────────────────────────────────────────────
# PLANNER
# ─────────────────────────────────────────────────────────────────────────────
async def make_plan(query: str) -> dict:
    schema = build_live_schema()

    prompt = f"""You are an expert SQLite query writer for an industrial maintenance database.
The REAL column values from the database are shown below. Use them for exact spellings in LIKE filters.

{schema}

Today: {date.today().isoformat()}

═══ SQLITE RULES — follow every rule exactly ═══

RULE 1 — ALWAYS SELECT *
  Never write SELECT problem or SELECT work_done_by or any partial column list.
  The ONLY exceptions: explicit "only problems" → SELECT problem, or COUNT/SUM/GROUP BY aggregates.
  For every other query: SELECT * FROM ...
  Even for "only problems" / "only solutions" requests, PREFER SELECT * — the
  presentation layer will extract just the field that's needed. Only use a
  partial column list for COUNT/SUM/GROUP BY aggregates.

RULE 2 — TEXT MATCHING (flexible LIKE, not exact equals)
  LOWER(col) LIKE LOWER('%value%')
  Multi-word: LOWER(machine) LIKE '%pick%place%' or LOWER(machine) LIKE '%wave%solder%'
  Model names: LOWER(model_name) LIKE '%omron%' or LOWER(model_name) LIKE '%omron%vt%'
  Worker names: LOWER(work_done_by) LIKE '%surname%'

RULE 3 — DATES: ALWAYS use iso_date column (format YYYY-MM-DD)
  Year filter: SUBSTR(iso_date,1,4) = '2026'
  Month filter: SUBSTR(iso_date,1,7) = '2026-01'
  Most recent: ORDER BY iso_date DESC
  Oldest: ORDER BY iso_date ASC
  NEVER filter on the 'date' column — always iso_date.

RULE 4 — loss_time is REAL minutes, -1 = unknown. Never exclude -1 rows unless user asks.

RULE 5 — LINE RANGE: line_no BETWEEN 1 AND 10

RULE 6 — LIMITS
  User says "all": LIMIT 300
  Default: LIMIT 100
  NEVER use LIMIT 8 — that truncates results.

RULE 7 — ORDER BY
  "latest/newest/recent/last" → ORDER BY iso_date DESC
  "oldest/earliest/first"     → ORDER BY iso_date ASC
  "sorted by downtime"        → ORDER BY loss_time DESC
  No explicit order stated    → ORDER BY iso_date DESC  (always show recent first)

RULE 8 — MODEL NAME MATCHING
  Users often say partial model names like "Omron VT", "Panasonic NPM", "Heller 1809".
  Match them with LIKE: LOWER(model_name) LIKE '%omron%' or LOWER(model_name) LIKE '%omron%vt%'
  PREFER the broader single-token match (e.g. '%omron%') unless the user gave
  multiple distinguishing words AND the schema sample values above show that
  exact combination occurring — overly specific LIKE patterns return 0 rows.
  Use BOTH machine AND model filters when user provides both.

RULE 9 — CHARTS
  wants_chart=true ONLY when user says chart/graph/pie/bar/trend.
  chart_sql must return EXACTLY two columns: col AS label, COUNT(*) AS value
  Example: SELECT shift AS label, COUNT(*) AS value FROM mttr_records GROUP BY shift ORDER BY value DESC

RULE 10 — tsg_records ONLY when user says "troubleshooting guide" or "TSG"

═══ ROUTING RULES ═══

mode="sql"      → ANY structured filter present: machine name, model, line, worker, date/year/month, shift
mode="semantic" → ONLY pure fault symptoms, zero structured filters ("motor overheating how to fix")
mode="hybrid"   → fault symptom PLUS at least one structured filter
mode="tsg"      → user mentions troubleshooting guide / TSG
mode="general"  → definition/concept question, no database needed
mode="chat"     → greeting or smalltalk

output_format rules:
  "table"          → user says table / tabular / in a table
  "problems_only"  → user says "only problems" / "list of problems" / "all problems"
  "solutions_only" → user says "only solutions" / "all solutions"
  "summary"        → user says summary / brief / overview
  "default"        → everything else

═══ OUTPUT: raw JSON only, no markdown, no text before or after ═══
{{
  "mode": "sql|semantic|hybrid|tsg|general|chat",
  "sql": "complete SELECT * ... statement or empty string",
  "semantic_query": "symptom text for vector search or empty",
  "output_format": "default|table|problems_only|solutions_only|summary",
  "wants_chart": false,
  "chart_sql": "SELECT col AS label, COUNT(*) AS value ... or empty",
  "chart_type": "bar|pie|line|hbar|empty",
  "answer_focus": "one sentence: exactly what the user wants"
}}

═══ EXAMPLES (study carefully) ═══

Q: "give me a list of all problems in aoi machine omron vt model"
{{"mode":"sql","sql":"SELECT * FROM mttr_records WHERE LOWER(machine) LIKE '%aoi%' AND LOWER(model_name) LIKE '%omron%' ORDER BY iso_date DESC","semantic_query":"","output_format":"problems_only","wants_chart":false,"chart_sql":"","chart_type":"","answer_focus":"all problems for AOI machine Omron VT model"}}

Q: "give me all the problems of aoi machine in a tabular form"
{{"mode":"sql","sql":"SELECT * FROM mttr_records WHERE LOWER(machine) LIKE '%aoi%' ORDER BY iso_date DESC","semantic_query":"","output_format":"table","wants_chart":false,"chart_sql":"","chart_type":"","answer_focus":"all AOI machine problems in table"}}

Q: "give me all the work done by all workers in january 2026"
{{"mode":"sql","sql":"SELECT * FROM mttr_records WHERE SUBSTR(iso_date,1,7)='2026-01' ORDER BY work_done_by, iso_date","semantic_query":"","output_format":"default","wants_chart":false,"chart_sql":"","chart_type":"","answer_focus":"all jobs done in January 2026 by worker"}}

Q: "arrange in latest to oldest order of all the work done on wave soldering machine"
{{"mode":"sql","sql":"SELECT * FROM mttr_records WHERE LOWER(machine) LIKE '%wave%solder%' ORDER BY iso_date DESC","semantic_query":"","output_format":"default","wants_chart":false,"chart_sql":"","chart_type":"","answer_focus":"wave soldering records ordered latest to oldest"}}

Q: "show all problems of pick and place machine from line 1 to 7 in a table"
{{"mode":"sql","sql":"SELECT * FROM mttr_records WHERE LOWER(machine) LIKE '%pick%place%' AND line_no BETWEEN 1 AND 7 ORDER BY line_no, iso_date DESC","semantic_query":"","output_format":"table","wants_chart":false,"chart_sql":"","chart_type":"","answer_focus":"pick and place problems lines 1-7 in table"}}

Q: "motor vibration issue give solution"
{{"mode":"semantic","sql":"","semantic_query":"motor vibration solution fix","output_format":"default","wants_chart":false,"chart_sql":"","chart_type":"","answer_focus":"solutions for motor vibration"}}

Q: "pie chart of failures by shift in 2026"
{{"mode":"sql","sql":"","semantic_query":"","output_format":"default","wants_chart":true,"chart_sql":"SELECT shift AS label, COUNT(*) AS value FROM mttr_records WHERE SUBSTR(iso_date,1,4)='2026' AND shift IS NOT NULL AND shift != '' GROUP BY shift ORDER BY value DESC","chart_type":"pie","answer_focus":"shift-wise failure distribution 2026"}}

Q: "who did the most work in 2025"
{{"mode":"sql","sql":"SELECT work_done_by AS label, COUNT(*) AS value FROM mttr_records WHERE SUBSTR(iso_date,1,4)='2025' AND work_done_by IS NOT NULL AND work_done_by != '' GROUP BY work_done_by ORDER BY value DESC","semantic_query":"","output_format":"default","wants_chart":false,"chart_sql":"","chart_type":"","answer_focus":"top worker in 2025"}}

Q: "what is wave soldering"
{{"mode":"general","sql":"","semantic_query":"","output_format":"default","wants_chart":false,"chart_sql":"","chart_type":"","answer_focus":"definition and explanation of wave soldering"}}

Now output JSON for:
USER QUESTION: {query}

JSON:"""

    _fallback = {**_PLAN_DEFAULTS, "semantic_query": query, "answer_focus": query}
    try:
        raw  = await call_ollama(prompt, model=OLLAMA_MODEL_PLANNER, max_tokens=500)
        plan = parse_plan(raw)
        if plan["mode"] == "semantic" and not plan.get("semantic_query"):
            plan["semantic_query"] = query
        print(f"[Planner] mode={plan['mode']}  fmt={plan['output_format']}  "
              f"sql={plan['sql'][:120] if plan['sql'] else '(none)'}")
        return plan
    except Exception as e:
        print(f"[Planner] error: {e}")
        return _fallback


# ─────────────────────────────────────────────────────────────────────────────
# SQL RETRY LADDER (generic — relaxes ANY condition in WHERE, in priority order)
# ─────────────────────────────────────────────────────────────────────────────
def _split_top_level_and(where_body: str) -> list[str]:
    """
    Split a WHERE body on top-level ' AND ' — i.e. NOT inside parentheses.
    Good enough for the simple, generated SQL this planner produces
    (no nested subqueries).
    """
    parts: list[str] = []
    depth = 0
    buf = []
    tokens = re.split(r"(\(|\)|\bAND\b)", where_body, flags=re.IGNORECASE)
    i = 0
    cur = ""
    for tok in tokens:
        if tok == "(":
            depth += 1
            cur += tok
        elif tok == ")":
            depth -= 1
            cur += tok
        elif tok.upper() == "AND" and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += tok
    if cur.strip():
        parts.append(cur.strip())
    return [p for p in parts if p]


def _condition_priority(cond: str) -> int:
    """
    Lower number = relax FIRST (most likely to be over-specific / wrong).
    model_name (often a typo'd / overly specific brand string) goes first,
    then machine, then everything else, then date/line/shift last (those
    are usually exactly what the user asked for and shouldn't be dropped
    unless nothing else helped).
    """
    c = cond.lower()
    if "model_name" in c:
        return 0
    if "spare_parts" in c:
        return 1
    if "work_done_by" in c:
        return 2
    if "machine" in c:
        return 3
    if "iso_date" in c or "line_no" in c or "shift" in c:
        return 5
    return 4


def _relax_sql(sql: str) -> list[str]:
    """
    Returns a list of progressively relaxed SQL variants to try after 0 rows.

    Strategy (generic, works for any WHERE shape produced by the planner):
      1. For the highest-priority condition (see _condition_priority), if it's
         a `LOWER(col) LIKE LOWER('%a%b%c%')` style filter, first try
         broadening it to just the first token (`%a%`).
      2. Then try DROPPING that condition entirely.
      3. Repeat for the next-highest-priority condition.
      4. Finally, a last-resort variant with ALL conditions dropped except
         date/line/shift (if any).
    """
    variants: list[str] = []
    s = sql.strip().rstrip(";")

    m = re.search(r"^(.*?\bWHERE\b)(.*?)(\bORDER\s+BY\b.*|\bLIMIT\b.*|$)", s,
                   re.IGNORECASE | re.DOTALL)
    if not m:
        return variants

    prefix, where_body, suffix = m.group(1), m.group(2).strip(), m.group(3).strip()
    conditions = _split_top_level_and(where_body)
    if len(conditions) < 1:
        return variants

    def rebuild(conds: list[str]) -> str:
        if not conds:
            base = re.sub(r"\bWHERE\b\s*$", "", prefix, flags=re.IGNORECASE).strip()
            return f"{base} {suffix}".strip()
        return f"{prefix} {' AND '.join(conds)} {suffix}".strip()

    ordered = sorted(range(len(conditions)), key=lambda i: _condition_priority(conditions[i]))

    for idx in ordered:
        cond = conditions[idx]

        # 1. Broaden a multi-token LIKE filter to its first significant token
        like_m = re.search(
            r"LOWER\((\w+)\)\s+LIKE\s+LOWER\('%([^']+)%'\)", cond, re.IGNORECASE
        )
        if like_m:
            col = like_m.group(1)
            pattern_inner = like_m.group(2)
            tokens = [t for t in re.findall(r"[a-z0-9]+", pattern_inner.lower()) if len(t) >= 3]
            if len(tokens) > 1:
                broader_cond = f"LOWER({col}) LIKE LOWER('%{tokens[0]}%')"
                new_conds = conditions.copy()
                new_conds[idx] = broader_cond
                variants.append(rebuild(new_conds))

        # 2. Drop this condition entirely
        if _condition_priority(cond) <= 3:   # don't casually drop date/line/shift here
            new_conds = conditions[:idx] + conditions[idx + 1:]
            variants.append(rebuild(new_conds))

    # 3. Last resort: keep ONLY date/line/shift conditions (if any), drop the rest
    keep = [c for c in conditions if _condition_priority(c) >= 5]
    if 0 < len(keep) < len(conditions):
        variants.append(rebuild(keep))

    # dedupe, drop empties / identical-to-original
    out, seen = [], set()
    for v in variants:
        v2 = re.sub(r"\s{2,}", " ", v).strip()
        if v2 and v2 != s and v2 not in seen:
            seen.add(v2)
            out.append(v2)
    return out


async def run_sql_with_retry(sql: str) -> tuple[list[dict], str]:
    """
    1. Try the original SQL.
    2. On 0 rows, try relaxed variants (broadest-first per _relax_sql ordering).
    3. On any error, attempt a single LLM repair.
    Returns (rows, sql_actually_used).
    """
    if not sql or not sql.strip():
        return [], ""

    try:
        rows = safe_sql(sql)
        if rows:
            return rows, sql
    except Exception as e1:
        print(f"[SQL] Primary failed: {e1}\n  SQL: {sql[:150]}")
        repaired = await _repair_sql(sql, str(e1))
        if repaired:
            try:
                rows = safe_sql(repaired)
                if rows:
                    return rows, repaired
            except Exception:
                pass

    for relaxed in _relax_sql(sql):
        print(f"[SQL] Retrying relaxed: {relaxed[:140]}")
        try:
            rows = safe_sql(relaxed)
            if rows:
                print(f"[SQL] Relaxed query returned {len(rows)} rows.")
                return rows, relaxed
        except Exception as re2:
            print(f"[SQL] Relaxed attempt failed: {re2}")
            continue

    return [], sql


async def _repair_sql(sql: str, error: str) -> Optional[str]:
    """Ask the planner model to fix a broken SQL. Returns corrected SQL or None."""
    schema = build_live_schema()
    fix_prompt = (
        f"This SQLite SELECT failed:\n{sql}\n\nError: {error}\n\n"
        f"Schema:\n{schema}\n\n"
        f"Write ONLY the corrected SELECT statement. No markdown, no explanation."
    )
    try:
        raw = await call_ollama(fix_prompt, model=OLLAMA_MODEL_PLANNER, max_tokens=300)
        raw = re.sub(r"```(?:sql)?|```", "", raw).strip()
        m = re.search(r"(?i)select[\s\S]+", raw)
        if m:
            fixed = m.group(0).strip()
            print(f"[SQL] Repaired: {fixed[:120]}")
            return fixed
    except Exception as e:
        print(f"[SQL] Repair call failed: {e}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDER (deterministic — no LLM)
# ─────────────────────────────────────────────────────────────────────────────
_VIZ_MAP = {
    "bar":  "bar_chart",
    "pie":  "pie_chart",
    "line": "line_chart",
    "hbar": "horizontal_bar_chart",
}


def build_chart(rows: list[dict], chart_type: str, title: str) -> Optional[dict]:
    data = []
    for r in rows:
        label = r.get("label")
        value = r.get("value")
        if label is None:
            vals = list(r.values())
            if len(vals) >= 2:
                label, value = vals[0], vals[1]
        if label is None or value is None:
            continue
        try:
            fv = float(value)
        except (TypeError, ValueError):
            continue
        data.append({
            "label": str(label)[:35],
            "value": int(fv) if fv == int(fv) else round(fv, 1),
        })
    if not data:
        return None
    total = sum(d["value"] for d in data) or 1
    for d in data:
        d["percent"] = round(d["value"] / total * 100, 1)
    return {
        "responseType": _VIZ_MAP.get(chart_type, "bar_chart"),
        "title": title or "Chart",
        "xAxis": "Category",
        "yAxis": "Value",
        "data": data[:15],
        "summary": "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# SHARED ROW HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _loss_str(r: dict) -> str:
    """Convert loss_time (minutes) to human label. -1 or None = Unknown."""
    lt = r.get("loss_time", -1)
    try:
        lt = float(lt)
    except (TypeError, ValueError):
        return "Unknown"
    if lt < 0:
        return "Unknown"
    return f"{lt:.0f} min" if lt < 60 else f"{lt/60:.1f} hr ({lt:.0f} min)"


def _date_val(r: dict) -> str:
    """Always return a date string — prefers iso_date, falls back to date column."""
    return str(r.get("iso_date") or r.get("date") or "").strip()


def _is_tsg_rows(rows: list[dict]) -> bool:
    return bool(rows) and ("issue" in rows[0] or "corrective" in rows[0])


def rows_to_text(rows: list[dict]) -> str:
    """
    Human-readable block format — used as CONTEXT for LLM prompts (troubleshoot
    / summary), never as the final user-facing output for plain listings.
    """
    if not rows:
        return "(no records)"

    is_tsg = _is_tsg_rows(rows)
    blocks = []

    for i, r in enumerate(rows, 1):
        if is_tsg:
            blocks.append(
                f"[{i}] Line={r.get('line_no','')} | Machine={r.get('machine','')} | "
                f"Issue={r.get('issue','')} | Cause={r.get('cause','')} | "
                f"Action={r.get('corrective','')}"
            )
        else:
            def f(key: str, fallback: str = "Unknown") -> str:
                v = r.get(key)
                if v in (None, "", -1, -1.0):
                    return fallback
                return str(v).strip() or fallback

            loss   = _loss_str(r)
            dt     = _date_val(r) or "Unknown"
            model  = f("model_name", "")
            line   = r.get("line_no")
            spare  = f("spare_parts", "")
            parts  = [
                f"[{i}]",
                f"Worker={f('work_done_by')}",
                f"Date={dt}",
                f"Shift={f('shift')}",
                f"Machine={f('machine')}",
                f"Model={model}" if model else None,
                f"Line={line}"   if line not in (None, "", -1) else None,
                f"Problem={f('problem')}",
                f"Solution={f('solution')}",
                f"Downtime={loss}",
                f"SpareParts={spare}" if spare else None,
            ]
            blocks.append(" | ".join(p for p in parts if p is not None))

    return "\n".join(blocks)


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATE FORMATTER (deterministic — for COUNT/SUM/GROUP BY results)
# ─────────────────────────────────────────────────────────────────────────────
def format_aggregate(rows: list[dict], sql: str) -> Optional[str]:
    if not rows:
        return None
    first = rows[0]
    keys  = list(first.keys())

    if len(rows) == 1 and len(keys) == 1:
        k   = str(keys[0]).lower()
        val = list(first.values())[0]
        if "cnt" in k or "count" in k:
            return f"Found **{val}** matching record{'s' if val != 1 else ''}."
        if "total" in k or "sum" in k:
            if val is None or float(val) <= 0:
                return "No downtime data found."
            v = float(val)
            return (
                f"Total downtime: **{v:.0f} minutes**"
                if v < 60
                else f"Total downtime: **{v/60:.1f} hours** ({v:.0f} min)"
            )

    if "group by" in sql.lower() and len(keys) >= 2:
        name_col  = keys[0]
        count_col = keys[1]
        lines = []
        for i, row in enumerate(rows[:25], 1):
            name  = str(row.get(name_col) or "Unknown")
            count = row.get(count_col, "")
            extra = ""
            if len(keys) > 2:
                k3 = keys[2]
                v3 = row.get(k3)
                if v3 is not None:
                    try:
                        extra = f"  (avg downtime: {float(v3):.0f} min)"
                    except Exception:
                        pass
            lines.append(f"{i}. **{name}**: {count}{extra}")
        return "\n".join(lines)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# ★★★ DETERMINISTIC RECORD FORMATTERS — no LLM, never skip/contradict ★★★
# ─────────────────────────────────────────────────────────────────────────────
def _esc(v) -> str:
    if v in (None, "", -1, -1.0):
        return "—"
    s = str(v).replace("|", "/").replace("\n", " ").strip()
    return s if s else "—"


def fmt_table(rows: list[dict]) -> str:
    if not rows:
        return "(no records)"
    is_tsg = _is_tsg_rows(rows)
    lines: list[str] = []
    if is_tsg:
        lines.append("| # | Line | Machine | Issue | Cause | Corrective Action |")
        lines.append("|---|------|---------|-------|-------|-------------------|")
        for i, r in enumerate(rows, 1):
            lines.append(
                f"| {i} | {_esc(r.get('line_no'))} | {_esc(r.get('machine'))} | "
                f"{_esc(r.get('issue'))} | {_esc(r.get('cause'))} | {_esc(r.get('corrective'))} |"
            )
    else:
        lines.append("| # | Date | Worker | Shift | Machine | Model | Line | Problem | Solution | Downtime |")
        lines.append("|---|------|--------|-------|---------|-------|------|---------|----------|----------|")
        for i, r in enumerate(rows, 1):
            line_no = r.get("line_no")
            line_val = line_no if line_no not in (None, "", -1) else None
            lines.append(
                f"| {i} | {_esc(_date_val(r))} | {_esc(r.get('work_done_by'))} | "
                f"{_esc(r.get('shift'))} | {_esc(r.get('machine'))} | {_esc(r.get('model_name'))} | "
                f"{_esc(line_val)} | {_esc(r.get('problem'))} | {_esc(r.get('solution'))} | "
                f"{_esc(_loss_str(r))} |"
            )
    lines.append("")
    lines.append(f"Total: {len(rows)} record{'s' if len(rows) != 1 else ''}.")
    return "\n".join(lines)


def fmt_field_list(rows: list[dict], field: str, tsg_field: str) -> str:
    """
    Numbered list of a single field (problems_only / solutions_only).
    field    = column name for normal mttr_records ('problem' / 'solution')
    tsg_field= column name for tsg_records          ('issue'   / 'corrective')
    """
    if not rows:
        return "(no records)"
    is_tsg = _is_tsg_rows(rows)
    use_field = tsg_field if is_tsg else field
    items = []
    for r in rows:
        v = r.get(use_field)
        if v in (None, ""):
            continue
        v = str(v).strip()
        if v:
            items.append(v)
    if not items:
        return f"None of the matching records have a recorded {field}."
    lines = [f"{i}. {v}" for i, v in enumerate(items, 1)]
    lines.append("")
    lines.append(f"Total: {len(items)} {field}{'s' if len(items) != 1 else ''} found.")
    return "\n".join(lines)


def fmt_default_list(rows: list[dict]) -> str:
    if not rows:
        return "(no records)"
    is_tsg = _is_tsg_rows(rows)
    lines = []
    for i, r in enumerate(rows, 1):
        if is_tsg:
            lines.append(
                f"{i}. Line {_esc(r.get('line_no'))} | {_esc(r.get('machine'))} | "
                f"Issue: {_esc(r.get('issue'))} | Cause: {_esc(r.get('cause'))} | "
                f"Fix: {_esc(r.get('corrective'))}"
            )
        else:
            dt      = _date_val(r) or "Unknown date"
            worker  = r.get("work_done_by") or "Unknown"
            machine = r.get("machine") or "Unknown machine"
            model   = r.get("model_name")
            line    = r.get("line_no")
            scope   = machine
            if model not in (None, ""):
                scope += f" {model}"
            if line not in (None, "", -1):
                scope += f" Line {line}"
            problem  = _esc(r.get("problem"))
            solution = _esc(r.get("solution"))
            loss     = _loss_str(r)
            lines.append(
                f"{i}. [{dt}] {worker} | {scope} | Problem: {problem} | "
                f"Solution: {solution} | Downtime: {loss}"
            )
    lines.append("")
    lines.append(f"Total: {len(rows)} record{'s' if len(rows) != 1 else ''}.")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# ★★★ LLM PROMPTS — only for genuine reasoning / prose ★★★
# ─────────────────────────────────────────────────────────────────────────────
def _lang_note(hinglish: bool) -> str:
    if not hinglish:
        return ""
    return (
        "\nLANGUAGE: Reply in Hinglish (Hindi in Roman script + English technical "
        "terms). Keep machine names, model names, and numbers in English.\n"
    )


def build_troubleshoot_prompt(query: str, rows: list[dict], answer_focus: str,
                               hinglish: bool) -> str:
    if rows:
        records_block = rows_to_text(rows)
        db_section = f"=== RELATED PAST RECORDS ({len(rows)}) ===\n{records_block}\n=== END ==="
    else:
        db_section = "=== NO RELATED PAST RECORDS FOUND ==="

    return (
        f"You are a senior SMD/industrial maintenance engineer helping fix a live fault.\n\n"
        f"TECHNICIAN REPORTS: {query}\n"
        f"FOCUS: {answer_focus or query}\n\n"
        f"{db_section}\n\n"
        f"Respond in EXACTLY this format:\n"
        f"MOST LIKELY CAUSE:\n[1-2 sentences]\n\n"
        f"RECOMMENDED FIX:\n1. [step]\n2. [step]\n3. [step]\n\n"
        f"WHY THIS HAPPENS:\n[1-2 sentences]\n\n"
        f"SAFETY NOTE:\n[one sentence]\n\n"
        f"RULES:\n"
        f"- If past records above are relevant, reference what worked before.\n"
        f"- Do NOT invent record data not shown above.\n"
        f"- Do NOT add any other sections, headers, or preamble.\n"
        f"{_lang_note(hinglish)}"
    )


def _summary_stats(rows: list[dict]) -> dict:
    machines: Counter = Counter()
    workers: Counter  = Counter()
    problems: list[str] = []
    total_loss = 0.0
    loss_n = 0
    for r in rows:
        m = (r.get("machine") or "").strip()
        if m:
            machines[m] += 1
        w = (r.get("work_done_by") or "").strip()
        if w:
            workers[w] += 1
        p = (r.get("problem") or r.get("issue") or "").strip()
        if p:
            problems.append(p)
        try:
            lt = float(r.get("loss_time", -1))
            if lt >= 0:
                total_loss += lt
                loss_n += 1
        except (TypeError, ValueError):
            pass
    return {
        "total": len(rows),
        "top_machines": machines.most_common(3),
        "top_workers": workers.most_common(3),
        "sample_problems": problems[:6],
        "avg_loss": (total_loss / loss_n) if loss_n else None,
    }


def build_summary_prompt(query: str, rows: list[dict], answer_focus: str,
                          hinglish: bool) -> str:
    stats = _summary_stats(rows)
    machines_str = ", ".join(f"{m} ({c})" for m, c in stats["top_machines"]) or "various machines"
    workers_str  = ", ".join(f"{w} ({c})" for w, c in stats["top_workers"]) or "various technicians"
    problems_str = "; ".join(stats["sample_problems"]) or "no specific problems recorded"
    avg_loss     = f"{stats['avg_loss']:.0f} minutes" if stats["avg_loss"] is not None else "not recorded"

    return (
        f"Write a 4-6 sentence plain-prose summary for a maintenance report.\n\n"
        f"FACTS (use ONLY these — do not invent other numbers, names, or dates):\n"
        f"- Total matching records: {stats['total']}\n"
        f"- Machines involved (with record counts): {machines_str}\n"
        f"- Technicians involved (with record counts): {workers_str}\n"
        f"- Example problems recorded: {problems_str}\n"
        f"- Average downtime where known: {avg_loss}\n\n"
        f"USER ASKED: {query}\n"
        f"FOCUS: {answer_focus or query}\n\n"
        f"Write ONLY the summary paragraph(s). No headers, no bullet points, "
        f"no numbered lists, no preamble.\n"
        f"{_lang_note(hinglish)}"
    )


async def build_no_data_response(query: str, answer_focus: str, mode: str,
                                  hinglish: bool, _FAULT_KW_RE: re.Pattern) -> str:
    """
    Deterministic 'no records' message. Only adds an LLM-written engineering
    note when the question is actually fault/troubleshoot-related — and even
    then, with a short, tightly-scoped prompt that can't ramble about
    "the database" or produce headers/bullets.
    """
    if answer_focus:
        base = f"No matching records found in the database for: {answer_focus}."
    else:
        base = "No matching records found in the database for this query."

    wants_engineering_note = mode in ("semantic", "hybrid", "tsg") or bool(_FAULT_KW_RE.search(query))
    if not wants_engineering_note:
        return base + " Try checking the spelling or broadening your search."

    prompt = (
        f"A technician asked this maintenance question, but no matching records "
        f"exist in the database: \"{query}\"\n\n"
        f"Write 2-3 short sentences of general engineering guidance on this topic, "
        f"based on standard industrial/SMD maintenance practice.\n"
        f"Do NOT mention databases, records, search results, or whether anything "
        f"was found — just give the practical guidance itself.\n"
        f"Do NOT use headers, bullet points, or numbered lists.\n"
        f"{_lang_note(hinglish)}"
    )
    try:
        eng = await call_ollama(prompt, model=OLLAMA_MODEL_SYNTH, max_tokens=MAX_NODATA_TOKENS)
        eng = eng.strip()
        # Strip accidental headers/bullets the small model sometimes adds anyway
        eng = re.sub(r"(?im)^\s*(no\s+(matching\s+)?records?.*|from\s+engineering\s+knowledge:?)\s*$", "", eng).strip()
        eng = re.sub(r"(?m)^\s*[-•]\s*", "", eng).strip()
    except Exception:
        eng = ""

    if eng:
        return f"{base}\n\nEngineering knowledge: {eng}"
    return base


# ─────────────────────────────────────────────────────────────────────────────
# LANGUAGE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
_HINDI_MARKERS = frozenset({
    "hai", "hain", "tha", "thi", "ho", "kar", "karo", "kya", "kyun",
    "kaise", "nahi", "aur", "ya", "mein", "pe", "se", "ko", "ka", "ke",
    "hum", "aap", "tum", "theek", "sahi", "kharab", "dikkat", "samasya",
    "kaam", "kitne", "kitna", "batao", "chalu", "band",
})

_HINDI_FILLER = frozenset({
    "hai", "hain", "karo", "kaise", "theek", "sahi", "kar", "nahi",
    "kya", "mein", "aur", "ya", "se", "ko", "ka", "ke", "pe",
    "solve", "fix", "repair", "check", "thik", "hua", "hui",
})


def is_hinglish(text: str) -> bool:
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    if not tokens:
        return False
    hits = sum(1 for t in tokens if t in _HINDI_MARKERS)
    return hits >= 2 or (hits >= 1 and hits / len(tokens) >= 0.12)


def strip_hinglish_filler(query: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+", query)
    kept   = [t for t in tokens if t.lower() not in _HINDI_FILLER]
    cleaned = " ".join(kept).strip()
    return cleaned if len(cleaned) >= 3 else query


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT FORMAT REGEX SAFETY NETS
# (catch what the planner LLM might misclassify)
# ─────────────────────────────────────────────────────────────────────────────
_TABLE_RE          = re.compile(r"\b(table|tabular|in a table|spreadsheet|grid)\b", re.IGNORECASE)
_PROBLEMS_ONLY_RE  = re.compile(
    r"\b(only\s+problems?|all\s+problems?|list\s+of\s+(all\s+)?problems?|problems?\s+only)\b",
    re.IGNORECASE,
)
_SOLUTIONS_ONLY_RE = re.compile(
    r"\b(only\s+solutions?|all\s+solutions?|list\s+of\s+(all\s+)?solutions?|solutions?\s+only)\b",
    re.IGNORECASE,
)
_SUMMARY_RE        = re.compile(r"\b(summary|summarize|summarise|brief\s+overview|overview)\b", re.IGNORECASE)


def resolve_output_format(plan: dict, query: str) -> str:
    """Planner's output_format, with regex overrides for common phrasings."""
    fmt = plan.get("output_format", "default")
    if _TABLE_RE.search(query):
        return "table"
    if _PROBLEMS_ONLY_RE.search(query):
        return "problems_only"
    if _SOLUTIONS_ONLY_RE.search(query):
        return "solutions_only"
    if _SUMMARY_RE.search(query):
        return "summary"
    return fmt


# ─────────────────────────────────────────────────────────────────────────────
# SEMANTIC SEARCH BRIDGE
# ─────────────────────────────────────────────────────────────────────────────
_semantic_fn = None
_tsg_fn      = None


def set_search_fns(sem, tsg):
    global _semantic_fn, _tsg_fn
    _semantic_fn = sem
    _tsg_fn      = tsg


async def do_semantic(query: str) -> list[dict]:
    if _semantic_fn is None:
        return []
    q = strip_hinglish_filler(query) if is_hinglish(query) else query
    if len(q) < 3:
        q = query
    results, _ = await _semantic_fn(q, use_multi_query=False)
    return results


async def do_tsg(query: str) -> list[dict]:
    if _tsg_fn is None:
        return []
    return await _tsg_fn(query)


# ─────────────────────────────────────────────────────────────────────────────
# RECORD CARD BUILDERS (frontend shape)
# ─────────────────────────────────────────────────────────────────────────────
def to_record_cards(rows: list[dict]) -> list[dict]:
    cards = []
    for r in rows:
        if "machine" not in r and "problem" not in r:
            continue
        cards.append({
            "machine":      r.get("machine", ""),
            "model_name":   r.get("model_name", ""),
            "problem":      r.get("problem", ""),
            "solution":     r.get("solution", ""),
            "loss_time":    _loss_str(r),
            "work_done_by": r.get("work_done_by", ""),
            "date":         _date_val(r),
            "shift":        r.get("shift", ""),
            "spare_parts":  r.get("spare_parts", ""),
            "image_b64":    r.get("image_b64", ""),
            "image_name":   r.get("image_name", ""),
            "image_mime":   r.get("image_mime", ""),
        })
    return cards


def to_tsg_cards(rows: list[dict]) -> list[dict]:
    return [
        {
            "line_no":    r.get("line_no", ""),
            "machine":    r.get("machine", ""),
            "issue":      r.get("issue", ""),
            "cause":      r.get("cause", ""),
            "corrective": r.get("corrective", ""),
        }
        for r in rows
    ]


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────────────────────────────────────
class V2Request(BaseModel):
    query: str
    machine_filter: Optional[str] = None
    model_filter: Optional[str] = None
    last_ai_response: Optional[str] = None
    selected_text: Optional[str] = None
    offset: int = 0
    tsg_followup_query: Optional[str] = None
    output_format: Optional[str] = None
    date_filter: Optional[dict] = None


class V2Response(BaseModel):
    ai_suggestion: str
    intent: str
    db_records_used: int
    db_records_summary: list
    diagram_data: Optional[dict] = None
    visualization: Optional[dict] = None
    retrieval_confidence: Optional[float] = None
    corrected_query: Optional[str] = None
    detected_model: Optional[str] = None
    total_records_found: int = 0
    has_more: bool = False
    current_offset: int = 0
    suggest_tsg: bool = False
    tsg_records_used: int = 0
    tsg_records_summary: list = []
    output_format: Optional[str] = None
    date_filter_label: Optional[str] = None
    date_filtered_count: Optional[int] = None
    parsed_filters: Optional[dict] = None
    execution_path: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN HANDLER
# ─────────────────────────────────────────────────────────────────────────────

_FAULT_KW = re.compile(
    r"\b(jam|fault|error|alarm|broken|overheat|vibrat|noise|stuck|"
    r"not\s+working|not\s+picking|repair|kharab|dikkat)\b",
    re.IGNORECASE,
)


async def _handle(req: V2Request) -> V2Response:
    query  = req.query.strip()
    hingl  = is_hinglish(query)
    source = "maintenance database"

    # ── 1. PLAN ───────────────────────────────────────────────────────────────
    plan = await make_plan(query)
    mode = plan["mode"]
    synth_mode = mode  # may be overridden below (e.g. SQL→semantic fallback)

    # Inject active UI model filter into SQL when not already present
    if req.model_filter and plan.get("sql"):
        s = plan["sql"]
        clause = f"LOWER(model_name) LIKE LOWER('%{req.model_filter}%')"
        if "model_name" not in s.lower():
            if re.search(r"\bwhere\b", s, re.IGNORECASE):
                plan["sql"] = re.sub(
                    r"(?i)\bwhere\b", f"WHERE {clause} AND ", s, count=1
                )
            else:
                m = re.search(r"(?i)\b(order\s+by|limit)\b", s)
                plan["sql"] = (
                    s[: m.start()] + f" WHERE {clause} " + s[m.start():]
                    if m else s + f" WHERE {clause}"
                )

    # ── 2. EXECUTE ────────────────────────────────────────────────────────────
    rows:       list[dict] = []
    confidence: float      = 1.0
    used_sql:   str        = ""

    if mode == "chat":
        return V2Response(
            ai_suggestion="Hello! How can I help you with maintenance data?",
            intent="conversational",
            db_records_used=0,
            db_records_summary=[],
        )

    if mode == "general":
        ai = await call_ollama(
            f"You are an expert SMD maintenance engineer.\n"
            f"Question: {query}\n"
            f"Answer clearly and concisely in numbered points.",
            model=OLLAMA_MODEL_SYNTH,
            max_tokens=600,
        )
        return V2Response(
            ai_suggestion=ai,
            intent="general",
            db_records_used=0,
            db_records_summary=[],
            execution_path="general",
        )

    # ── SQL execution with retry ladder ──────────────────────────────────────
    if mode in ("sql", "hybrid", "tsg") and plan.get("sql"):
        rows, used_sql = await run_sql_with_retry(plan["sql"])
        if mode == "tsg":
            source = "Troubleshooting Guide"

    # ── Hybrid: semantic re-rank within SQL gate ──────────────────────────────
    if mode == "hybrid":
        sem_q = plan.get("semantic_query") or query
        if hingl:
            sem_q = strip_hinglish_filler(sem_q)
        sem_rows = await do_semantic(sem_q)
        if rows and sem_rows:
            gate   = {r.get("chroma_id", "") for r in rows if r.get("chroma_id")}
            ranked = [r for r in sem_rows if r.get("chroma_id", "") in gate]
            seen   = {r.get("chroma_id", "") for r in ranked}
            ranked += [r for r in rows if r.get("chroma_id", "") not in seen]
            rows = ranked
            confidence = 0.9
        elif sem_rows and not rows:
            rows = sem_rows
        source = "maintenance database (hybrid)"

    # ── Pure semantic ─────────────────────────────────────────────────────────
    if mode == "semantic":
        sem_q = plan.get("semantic_query") or query
        if hingl:
            sem_q = strip_hinglish_filler(sem_q)
        rows = await do_semantic(sem_q)
        confidence = 0.8
        source = "maintenance database (semantic)"

    # ── TSG semantic fallback ─────────────────────────────────────────────────
    if mode == "tsg" and not rows:
        rows = await do_tsg(plan.get("semantic_query") or query)
        source = "Troubleshooting Guide"

    # ── SQL→0 rows + fault keyword → semantic fallback ────────────────────────
    if mode == "sql" and used_sql and not rows and _FAULT_KW.search(query):
        rows = await do_semantic(query)
        confidence = 0.7
        source = "maintenance database (semantic fallback)"
        synth_mode = "hybrid"   # treat results as troubleshoot-grounding, not a plain listing
        print("[V2] SQL=0, fell back to semantic")

    # ── 2b. CHART (deterministic) ─────────────────────────────────────────────
    visualization = None
    if plan.get("wants_chart") and plan.get("chart_sql"):
        chart_rows, _ = await run_sql_with_retry(plan["chart_sql"])
        if chart_rows:
            visualization = build_chart(
                chart_rows,
                plan.get("chart_type", "bar"),
                plan.get("answer_focus", "Chart"),
            )

    # ── 3. SYNTHESIZE (deterministic-first) ───────────────────────────────────
    total    = len(rows)
    page     = rows[:PAGE_SIZE]
    is_tsg_q = (mode == "tsg") or _is_tsg_rows(rows)
    has_data = total > 0

    fmt = resolve_output_format(plan, query)
    answer_focus = plan.get("answer_focus", "")

    if visualization and not rows:
        ai_response = visualization.get("summary") or "Here is the requested chart."

    elif not has_data:
        ai_response = await build_no_data_response(query, answer_focus, synth_mode, hingl, _FAULT_KW)

    else:
        is_aggregate_only = (
            synth_mode in ("sql", "hybrid")
            and used_sql
            and not any("problem" in r or "solution" in r or "issue" in r for r in rows[:3])
            and bool(re.search(
                r"\b(count\s*\(|sum\s*\(|avg\s*\(|group\s+by)\b",
                used_sql, re.IGNORECASE,
            ))
        )
        agg = format_aggregate(rows, used_sql) if is_aggregate_only else None

        if agg:
            ai_response = agg
        elif fmt == "table":
            ai_response = fmt_table(rows)
        elif fmt == "problems_only":
            ai_response = fmt_field_list(rows, "problem", "issue")
        elif fmt == "solutions_only":
            ai_response = fmt_field_list(rows, "solution", "corrective")
        elif fmt == "summary":
            ai_response = await call_ollama(
                build_summary_prompt(query, rows, answer_focus, hingl),
                model=OLLAMA_MODEL_SYNTH,
                max_tokens=MAX_SUMMARY_TOKENS,
            )
        elif synth_mode in ("semantic", "hybrid"):
            ai_response = await call_ollama(
                build_troubleshoot_prompt(query, rows, answer_focus, hingl),
                model=OLLAMA_MODEL_SYNTH,
                max_tokens=MAX_SYNTH_TOKENS,
            )
        else:
            ai_response = fmt_default_list(rows)

    # ── 4. BUILD RESPONSE ─────────────────────────────────────────────────────
    tsg_cards = to_tsg_cards(page) if is_tsg_q else []
    rec_cards = to_record_cards(page) if not is_tsg_q else []

    intent_out = (
        "tsg_lookup"
        if is_tsg_q
        else "troubleshoot"
        if synth_mode in ("semantic", "hybrid")
        else "db_lookup"
    )

    return V2Response(
        ai_suggestion        = ai_response,
        intent               = intent_out,
        db_records_used      = len(rec_cards),
        db_records_summary   = rec_cards,
        visualization        = visualization,
        retrieval_confidence = round(float(confidence), 4),
        total_records_found  = total,
        has_more             = total > PAGE_SIZE,
        current_offset       = 0,
        suggest_tsg          = False,
        tsg_records_used     = len(tsg_cards),
        tsg_records_summary  = tsg_cards,
        output_format        = fmt if fmt != "default" else None,
        parsed_filters       = {
            "mode":  mode,
            "focus": answer_focus,
            "sql":   (used_sql or plan.get("sql") or "")[:200],
            "fmt":   fmt,
        },
        execution_path = mode,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ROUTE REGISTRATION
# ─────────────────────────────────────────────────────────────────────────────
def register_v2_routes(app: FastAPI):
    import app as _app

    set_search_fns(
        getattr(_app, "semantic_search", None),
        getattr(_app, "tsg_retrieve", None),
    )

    @app.post("/query2", response_model=V2Response)
    async def query_v2(req: V2Request):
        from app import (
            detect_intent, query_records, _EXPLAIN_FOLLOWUP, QueryRequest,
        )
        q = req.query.strip()
        intent0 = detect_intent(q)

        # Delegate diagram handling to proven V1 pipeline
        if intent0 in ("diagram", "diagram_context"):
            return await query_records(QueryRequest(**req.dict()))

        # Delegate explain / selected-text follow-ups
        if bool(_EXPLAIN_FOLLOWUP.search(q)) and (
            bool(req.last_ai_response) or bool(req.selected_text)
        ):
            return await query_records(QueryRequest(**req.dict()))

        return await _handle(req)

    @app.get("/v2-debug")
    async def v2_debug():
        """Shows exactly what the planner sees — use to verify schema loaded."""
        return {
            "schema":        build_live_schema(),
            "planner_model": OLLAMA_MODEL_PLANNER,
            "synth_model":   OLLAMA_MODEL_SYNTH,
        }

    print("[V2] Registered: POST /query2  |  GET /v2-debug")
