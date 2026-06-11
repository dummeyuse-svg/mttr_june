import json
import re
import base64
import io
import sqlite3
import hashlib
import numpy as np

import math
import asyncio
from collections import defaultdict
from pathlib import Path
from typing import Optional, Any
from datetime import datetime, date

import chromadb
import httpx
from chromadb.utils import embedding_functions
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles

from mttr_v2 import register_v2_routes

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
COLLECTION_NAME     = "mttr_records"
TSG_COLLECTION_NAME = "tsg_records"
DB_PATH             = "./chroma_db"
# EMBED_MODEL_PATH    = "./local_model"
# clean_excel.py AND app.py — must be identical
EMBED_MODEL_PATH = "./local_model_bhasha"
print(f"[Embed] Using model at: {EMBED_MODEL_PATH}")
SQLITE_PATH         = "./mttr_records.db"

OLLAMA_URL   = "http://127.0.0.1:11434"
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_MODEL_FAST  = "qwen2.5:0.5b"   # structured extraction only
OLLAMA_MODEL_PLANNER = "qwen2.5-coder:7b"   # planner / text-to-SQL ONLY
PLANNER_MAX_ROWS     = 60                   # hard cap on rows fed to synthesis

TOP_K          = 30
PAGE_SIZE      = 8
MAX_TOKENS     = 800
CONFIDENCE_THR = 0.38

try:
    import easyocr
    import cv2
    from PIL import Image, ImageEnhance, ImageFilter
    OCR_AVAILABLE = True
    _ocr_reader = None
except ImportError:
    OCR_AVAILABLE = False
    _ocr_reader = None


# ─────────────────────────────────────────────────────────────────────────────
# ★★★ SCHEMA REGISTRY — The heart of the dynamic architecture ★★★
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_COLS = {"id", "image_b64", "image_name", "image_mime", "chroma_id"}

_KNOWN_COL_HINTS: dict[str, dict] = {
    "machine":      {"label": "Machine name",          "examples": ["Panasonic", "Fuji", "JUKI"], "match_type": "partial"},
    "model_name":   {"label": "Machine model",         "examples": ["NPM-W2", "NXT III"],         "match_type": "partial"},
    "smd_line":     {"label": "SMD line number",       "examples": ["1", "2", "22"],               "match_type": "partial"},
    "problem":      {"label": "Fault / problem description", "examples": [],                       "match_type": "semantic"},
    "solution":     {"label": "Solution / action taken",     "examples": [],                       "match_type": "semantic"},
    "loss_time":    {"label": "Downtime in minutes",   "examples": ["45", "90"],                   "match_type": "numeric"},
    "work_done_by": {"label": "Technician / person who did the work", "examples": ["Rahul", "Anil"], "match_type": "partial"},
    "date":         {"label": "Date of the record",    "examples": ["2026-01-15"],                 "match_type": "date"},
    "shift":        {"label": "Work shift",            "examples": ["night", "general", "morning"],"match_type": "exact"},
    "spare_parts":  {"label": "Spare parts used",      "examples": ["belt", "motor"],              "match_type": "partial"},
    "line_no":      {"label": "SMD line number (exact)", "examples": ["3", "10", "42"], "match_type": "numeric"},
}


class SchemaRegistry:
    def __init__(self):
        self.columns: dict[str, dict] = {}
        self.text_columns: list[str]  = []
        self.numeric_columns: list[str] = []
        self.date_columns: list[str]  = []
        self.column_values: dict[str, list[str]] = {}
        self._ready = False

    def load(self, conn: sqlite3.Connection):
        rows = conn.execute("PRAGMA table_info(mttr_records)").fetchall()
        self.columns = {}
        for row in rows:
            col_name  = row[1]
            col_type  = row[2].upper()
            if col_name in _SYSTEM_COLS:
                continue
            hint = _KNOWN_COL_HINTS.get(col_name, {
                "label": col_name.replace("_", " ").title(),
                "examples": [],
                "match_type": "partial",
            })
            col_type_category = "text"
            if any(t in col_type for t in ("INT", "REAL", "NUMERIC", "FLOAT", "DOUBLE")):
                col_type_category = "numeric"
            elif col_name in ("date",) or "date" in col_name.lower():
                col_type_category = "date"

            self.columns[col_name] = {
                **hint,
                "col_type": col_type_category,
                "sql_col":  col_name,
            }

        self.text_columns    = [c for c, m in self.columns.items() if m["col_type"] == "text" and m.get("match_type") not in ("semantic", "numeric")]
        self.numeric_columns = [c for c, m in self.columns.items() if m["col_type"] == "numeric"]
        self.date_columns    = [c for c, m in self.columns.items() if m["col_type"] == "date"]
        self._ready = True

    def refresh_values(self, conn: sqlite3.Connection):
        self.column_values = {}
        for col in self.text_columns:
            if self.columns[col].get("match_type") in ("semantic",):
                continue
            try:
                rows = conn.execute(
                    f"SELECT DISTINCT {col} FROM mttr_records WHERE {col} IS NOT NULL AND {col} != '' LIMIT 5000"
                ).fetchall()
                self.column_values[col] = [r[0] for r in rows if r[0]]
            except Exception:
                self.column_values[col] = []

    def schema_description_for_llm(self) -> str:
        lines = []
        for col, meta in self.columns.items():
            examples_str = ""
            known_vals   = self.column_values.get(col, [])
            show_vals    = known_vals[:8] if known_vals else meta.get("examples", [])
            if show_vals:
                examples_str = f" (e.g. {', '.join(str(v) for v in show_vals[:6])})"
            lines.append(f'  "{col}": {meta["label"]}{examples_str}')
        return "\n".join(lines)


_schema = SchemaRegistry()


# ─────────────────────────────────────────────────────────────────────────────
# SQLITE ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def get_sqlite_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_sqlite():
    conn = get_sqlite_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mttr_records (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            chroma_id   TEXT,
            smd_line    TEXT,
            line_no     INTEGER,                    -- ← ADD
            machine     TEXT COLLATE NOCASE,
            model_name  TEXT COLLATE NOCASE,
            problem     TEXT,
            solution    TEXT,
            loss_time   REAL DEFAULT -1,
            work_done_by TEXT COLLATE NOCASE,
            date        TEXT,
            iso_date    TEXT,                       -- ← ADD
            shift       TEXT COLLATE NOCASE,
            spare_parts TEXT,
            image_b64   TEXT,
            image_name  TEXT,
            image_mime  TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_machine      ON mttr_records(machine)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_model        ON mttr_records(model_name)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_work_done_by ON mttr_records(work_done_by)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_date         ON mttr_records(date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_iso_date     ON mttr_records(iso_date)")   # ← ADD
    conn.execute("CREATE INDEX IF NOT EXISTS idx_shift        ON mttr_records(shift)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_smd_line     ON mttr_records(smd_line)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_line_no      ON mttr_records(line_no)")    # ← ADD
    conn.commit()
    conn.close()

def sqlite_rows_to_meta(rows) -> list[dict]:
    result = []
    for row in rows:
        d = dict(row)
        d.setdefault("machine",      "")
        d.setdefault("model_name",   "")
        d.setdefault("problem",      "")
        d.setdefault("solution",     "")
        d.setdefault("loss_time",    -1)
        d.setdefault("work_done_by", "")
        d.setdefault("date",         "")
        d.setdefault("shift",        "")
        d.setdefault("spare_parts",  "")
        d.setdefault("image_b64",    "")
        d.setdefault("image_name",   "")
        d.setdefault("image_mime",   "")
        d.setdefault("smd_line",     "")
        d.setdefault("chroma_id",    "")
        result.append(d)
    return result


def sqlite_is_ready() -> bool:
    try:
        conn = get_sqlite_conn()
        row  = conn.execute("SELECT COUNT(*) as cnt FROM mttr_records").fetchone()
        conn.close()
        return (row["cnt"] if row else 0) > 0
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# FUZZY VALUE MATCHER
# ─────────────────────────────────────────────────────────────────────────────

def _edit_distance(a: str, b: str) -> int:
    if abs(len(a) - len(b)) > 3:
        return 4
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
        if min(dp) > 3:
            return 4
    return dp[n]


def fuzzy_match_value(candidate: str, known_values: list[str], threshold: int = 2) -> Optional[str]:
    if not candidate or not known_values:
        return None
    c_lower = candidate.lower()
    for v in known_values:
        if v.lower() == c_lower:
            return v
    for v in known_values:
        if c_lower in v.lower() or v.lower() in c_lower:
            return v
    best_val, best_dist = None, threshold + 1
    for v in known_values:
        d = _edit_distance(c_lower, v.lower())
        if d < best_dist:
            best_dist = d
            best_val  = v
    return best_val if best_dist <= threshold else None


def _value_matches_substr(value: str, known_values: list[str]) -> bool:
    """True if value overlaps any known value (exact / substring either direction)."""
    if not value or not known_values:
        return False
    v = value.strip().lower()
    for kv in known_values:
        k = str(kv).lower()
        if v == k or v in k or k in v:
            return True
    return False


def _reconcile_filter_columns(filters: dict) -> dict:
    """
    Fix mis-assigned partial-match filters. For each value, check it against the
    KNOWN values of its assigned column:
      - matches own column        → keep (raw, so LIKE stays broad)
      - typo of an own-column val  → snap to the correct value
      - belongs to another column  → move it (if that column is free)
      - matches nothing known      → drop the bogus filter
    This stops 'Panasonic' (a model brand) from being treated as a machine name.
    """
    if not _schema._ready:
        return filters

    checkable = [c for c in ("machine", "model_name", "work_done_by", "spare_parts")
                 if c in _schema.columns]

    for col in list(filters.keys()):
        if col.startswith("_") or col in _SPECIAL_FILTER_COLS or col not in checkable:
            continue
        val = str(filters[col]).strip()
        if not val:
            continue

        own_vals = _schema.column_values.get(col, [])
        if _value_matches_substr(val, own_vals):
            continue                                    # raw LIKE will match — keep as typed

        snapped = fuzzy_match_value(val, own_vals, threshold=2)
        if snapped:
            filters[col] = snapped                      # typo correction
            continue

        # Value doesn't belong to this column — try to relocate it
        for other in checkable:
            if other == col or filters.get(other):
                continue
            other_vals = _schema.column_values.get(other, [])
            if _value_matches_substr(val, other_vals) or fuzzy_match_value(val, other_vals, 2):
                filters[other] = val
                break

        del filters[col]                                # drop the wrong assignment

    return filters


def _scan_known_values(query: str, filters: dict) -> dict:
    """
    Deterministic safety net for multi-filter queries.

    The small LLM often drops the machine/model/worker when the query also
    contains a line number, year, shift, etc. (e.g. "optima unload on line 4"
    → LLM returns only the line). So we scan the RAW query for any KNOWN value
    of these columns and lock it in, overriding whatever the LLM did.

    - Only scans distinctive columns (machine, model_name, work_done_by) whose
      values won't appear by accident in fault text. spare_parts is intentionally
      excluded — values like "Motor"/"Belt" routinely appear in problem text and
      would cause false matches; those stay with the LLM extractor.
    - Picks the LONGEST matching value per column so "Optima Unload" wins over a
      bare "Optima", and "Panasonic NPM" wins over "Panasonic".
    """
    if not _schema._ready:
        return filters

    q = query.lower()
    scan_cols = [c for c in ("machine", "model_name", "work_done_by")
                 if c in _schema.columns]

    for col in scan_cols:
        best = None
        for kv in _schema.column_values.get(col, []):
            k = str(kv).strip()
            if len(k) < 2:
                continue
            if k.lower() in q:                       # verbatim substring = high confidence
                if best is None or len(k) > len(best):
                    best = k
        if best:
            filters[col] = best                      # override — beats unreliable LLM output

    return filters

# ─────────────────────────────────────────────────────────────────────────────
# ★★★ CORE: LLM-BASED FILTER EXTRACTOR ★★★
# ─────────────────────────────────────────────────────────────────────────────

_FAST_REGEX_OVERRIDES = {
    "year": [
        r"\bin\s+(20\d{2})\b",
        r"\bfor\s+(20\d{2})\b",
        r"\byear\s+(20\d{2})\b",
        r"\b(20\d{2})\s+me\b",
        r"\b(20\d{2})\b",
    ],
    # "shift": [
    #     (r"\b(night\s*shift|night)\b",   "night"),
    #     (r"\b(general\s*shift|general)\b","general"),
    #     (r"\b(morning\s*shift|morning)\b","morning"),
    #     (r"\b(day\s*shift)\b",           "day"),
    # ],
    "line_no": [
        r"\bline\s*(?:no\.?|number|#)?\s*(\d+)\b",
        r"\bsmd\s*line\s*(\d+)\b",
        r"\bl\.?\s*(\d+)\b",
        r"\bline-(\d+)\b",
    ],
}

def _detect_shift(query: str) -> Optional[str]:
    """
    Fully data-driven shift detection. Shift names come ENTIRELY from the DB
    (_schema.column_values['shift']) — nothing is hardcoded. Works for any
    deployment: {night, day, evening} or {A, B, C, gen day, gen night} alike.
    Returns the EXACT stored value, which is required because the shift column
    is match_type 'exact' (LOWER(shift) = LOWER(?)).
    """
    if not _schema._ready:
        return None
    known = [str(v).strip() for v in _schema.column_values.get("shift", []) if str(v).strip()]
    if not known:
        return None

    q = query.lower()

    # Multi-char names ("gen night", "gen day", "evening", "night"): match as a
    # phrase. Longest-first so "gen night" wins over a bare "night".
    safe = sorted([v for v in known if len(v) >= 3], key=len, reverse=True)
    for v in safe:
        if re.search(rf"\b{re.escape(v.lower())}\b", q):
            return v

    # Short codes ("A", "B", "C"): only when tied to the word "shift", so a
    # stray letter never false-matches.
    short = [v for v in known if len(v) <= 2]
    for v in short:
        code = re.escape(v.lower())
        if re.search(rf"\b{code}\s+shift\b", q) or re.search(rf"\bshift\s*[-:]?\s*{code}\b", q):
            return v

    return None

# _RESPONSE_TYPE_PATTERNS = [
#     (r"\b(how\s+many|count|total\s+number|kitne|ginti)\b",                    "count"),
#     (r"\b(total\s+downtime|total\s+loss\s+time|total\s+hours?\s+lost|total\s+loss)\b", "sum_downtime"),
#     (r"\b(which\s+shift.*most|most.*shift|shift.*most)\b",                    "analytics_shift"),
#     (r"\b(which\s+machine.*most|most.*machine|machine.*most\s+problem)\b",    "analytics_machine"),
#     (r"\b(who.*most\s+work|most\s+work.*who|top\s+worker|most\s+repair)\b",  "analytics_worker"),
#     (r"\b(most\s+common\s+problem|frequent\s+fault|top\s+issue)\b",           "analytics_problem"),
#     (
#         r"\b(not\s+working|not\s+picking|not\s+moving|not\s+running|broken|failing"
#         r"|alarm|how\s+(?:do\s+i|to)\s+fix|help\s+me\s+fix|how\s+to\s+resolve"
#         # ── Hinglish "how to fix / solve" ──
#         r"|kaise\s+(?:theek|thik|sahi|solve|fix|repair|badlu|hataye?|nikalu)"
#         r"|(?:theek|thik|sahi|solve|fix)\s+kar(?:u|o|na|e|en)?)\b",
#         "troubleshoot",
#     ),
# ]

_RESPONSE_TYPE_PATTERNS = [
    (r"\b(how\s+many|count|total\s+number|kitne|ginti)\b",                    "count"),
    (r"\b(total\s+downtime|total\s+loss\s+time|total\s+hours?\s+lost|total\s+loss)\b", "sum_downtime"),

    # ── Most-frequent PROBLEM (ranked). MUST come before machine/shift, else
    #    "most ... machine" steals "most frequent problem in <machine> machine". ──
    (
        r"(most\s+(?:common|frequent|frequently|occurring|occurred|recurring|repeated)\b.*?"
        r"\b(?:problem|issue|fault|failure|defect|breakdown)"
        r"|(?:problem|issue|fault|failure|defect|breakdown)s?\b.*?\bmost\s+(?:common|frequent|frequently)"
        r"|top\s+(?:problem|issue|fault|failure)s?\b"
        r"|frequent\s+fault\b|top\s+issue\b"
        # ── Hinglish ──
        r"|sabse\s+(?:zyada|jyada)\b.*?\b(?:problem|dikkat|samasya|fault))",
        "analytics_problem",
    ),

    (r"\b(which\s+shift.*most|most.*shift|shift.*most)\b",                    "analytics_shift"),

    # Machine analytics only when machine is the GROUPING dimension, not a named scope.
    (r"\b(which\s+machine.*most|most.*machine.*problem|machine.*most\s+problem)\b", "analytics_machine"),

    (r"\b(who.*most\s+work|most\s+work.*who|top\s+worker|most\s+repair)\b",  "analytics_worker"),
    (
        r"\b(not\s+working|not\s+picking|not\s+moving|not\s+running|broken|failing"
        r"|alarm|how\s+(?:do\s+i|to)\s+fix|help\s+me\s+fix|how\s+to\s+resolve"
        r"|kaise\s+(?:theek|thik|sahi|solve|fix|repair|badlu|hataye?|nikalu)"
        r"|(?:theek|thik|sahi|solve|fix)\s+kar(?:u|o|na|e|en)?)\b",
        "troubleshoot",
    ),
]

_HAS_FAULT_KW = re.compile(
    r"\b(jam|fault|error|alarm|failure|broken|clog|overheat|vibrat|"
    r"noise|leak|burn|short|stuck|damage|issue|problem|not\s+working|"
    r"not\s+picking|not\s+moving|misalign|warp|melt|solder|paste|nozzle|"
    r"feeder|conveyor|motor|sensor|valve|pump|belt|bearing|spindle|"
    r"stoppage|stop|break|breakdown|halt|delay|"
    # ── Hinglish fault terms ──
    r"kharab|kharaab|garam|garmi|ruk|atak|phans|awaaz|aawaz|dikkat|"
    r"samasya|toot|tut|wear|ghis|ghisa|chal\s*nahi|kaam\s*nahi|theek\s*nahi)\b",
    re.IGNORECASE,
)

# Generic fault/part words that must NEVER be treated as a named entity filter.
_FAULT_WORDS = {
    "motor", "belt", "bearing", "sensor", "valve", "pump", "nozzle", "feeder",
    "conveyor", "spindle", "vibration", "vibrat", "noise", "jam", "leak",
    "problem", "issue", "fault", "error", "alarm", "solution",
}

async def extract_filters_dynamic(query: str) -> dict:
    filters: dict[str, Any] = {}
    q_lower = query.lower()

    # ── 1. FAST REGEX OVERRIDES ──────────────────────────────────────────────
    for pat in _FAST_REGEX_OVERRIDES["year"]:
        m = re.search(pat, query, re.IGNORECASE)
        if m:
            filters["year"] = m.group(1)
            break

    # for pat, value in _FAST_REGEX_OVERRIDES["shift"]:
    #     if re.search(pat, q_lower):
    #         filters["shift"] = value
    #         break
    shift_val = _detect_shift(query)
    if shift_val:
        filters["shift"] = shift_val

    for pat in _FAST_REGEX_OVERRIDES["line_no"]:
        m = re.search(pat, q_lower)
        if m:
            filters["line_no"] = int(m.group(1))
            break

    date_filter = parse_date_filter(query)
    if date_filter:
        mode = date_filter.get("mode")
        if mode == "year" and "year" not in filters:
            filters["year"] = str(date_filter["year"])
        elif mode == "after":
            filters["date_from"] = str(date_filter["date"])
        elif mode == "before":
            filters["date_to"]   = str(date_filter["date"])
        elif mode == "range":
            filters["date_from"] = str(date_filter["start"])
            filters["date_to"]   = str(date_filter["end"])
            if "year" in filters:
                del filters["year"]

    response_type = "list"
    for pat, rtype in _RESPONSE_TYPE_PATTERNS:
        if re.search(pat, q_lower):
            response_type = rtype
            break
    filters["_response_type"] = response_type
    filters["_has_fault_keyword"] = bool(_HAS_FAULT_KW.search(query))

    # ── 2. SCHEMA-DRIVEN EXTRACTION (LLM only when needed) ───────────────────
    if _schema._ready:
        # Free verbatim scan first — pins down correctly-spelled machine/model/worker.
        pre = _scan_known_values(query, dict(filters))
        verbatim_entity = any(pre.get(k) for k in ("machine", "model_name", "work_done_by"))
        if verbatim_entity:
            # A known entity was matched exactly — adopt it, skip the LLM round-trip.
            for k, v in pre.items():
                if not k.startswith("_") and v and k not in filters:
                    filters[k] = v
        else:
            # Nothing matched verbatim → the query may contain a typo'd name or a
            # spare-part filter the scan can't catch. Run the FULL-quality LLM extractor.
            llm_filters = await _llm_extract_filters(query, filters)
            for k, v in llm_filters.items():
                if k.startswith("_"):
                    continue
                if k in ("year", "shift", "smd_line", "date_from", "date_to"):
                    if k not in filters:
                        filters[k] = v
                else:
                    if v:
                        filters[k] = v

    requested_fields = _infer_requested_fields(query, filters)
    filters["_requested_fields"] = requested_fields

    # Normalize any SMD-line filter to an exact integer (line_no),
    # but only if the query actually contains a number (kills LLM hallucinations).
    if "smd_line" in filters and "line_no" not in filters:
        m = re.search(r"\d+", str(filters["smd_line"]))
        if m and re.search(r"\d", query):
            filters["line_no"] = int(m.group())
        del filters["smd_line"]

    return filters


def _llm_value_grounded(value: str, query: str) -> bool:
    """
    True only if the LLM-extracted filter value is actually present in the query.
    Guards against the small model hallucinating worker/machine/model names
    (e.g. inventing 'Amit Singh' / '2026' for a 'probe pin wear' question).
    """
    if not value:
        return False
    q = query.lower()
    val = str(value).lower().strip()
    if val and val in q:
        return True
    # token-level check: any significant word of the value must appear in the query
    for tok in re.findall(r"[a-z0-9]+", val):
        if len(tok) < 3:
            continue
        if tok in q:
            return True
        for qtok in re.findall(r"[a-z0-9]+", q):          # allow 1-char typo tolerance
            if len(qtok) >= 3 and _edit_distance(tok, qtok) <= 1:
                return True
    return False

async def _llm_extract_filters(query: str, existing_filters: dict) -> dict:
    schema_desc = _schema.schema_description_for_llm()

    value_hints = ""
    if _schema.column_values:
        lines = []
        for col, vals in _schema.column_values.items():
            if vals and col not in ("problem", "solution"):
                lines.append(f'  "{col}": [{", ".join(repr(v) for v in vals[:10])}]')
        if lines:
            value_hints = "\nKnown values in database:\n" + "\n".join(lines)

    prompt = f"""You are a database query filter extractor for an industrial maintenance system.

DATABASE SCHEMA (SQLite table: mttr_records):
{schema_desc}
{value_hints}

USER QUERY: {query}

Your job: extract ONLY explicit filters mentioned in the query.
Map each filter to the correct column name from the schema above.

Rules:
- Only extract filters that are EXPLICITLY stated.  Do NOT infer or guess.
- For "work done by X" or "by X" → use column "work_done_by"
- For machine names → use column "machine"
- For model names → use column "model_name"
- For line/SMD line numbers → skip (handled separately)
- "machine" is the MACHINE TYPE (e.g. Pick and Place, AOI Machine, Conveyor System).
  Brand/series names like "Panasonic NPM", "Heller 1809", "Optima Unload" are "model_name", NOT machine.
- For date/year → skip (handled separately)
- For shift (night/general/morning) → skip (handled separately)
- For problem/solution keywords → put them in "semantic_query" not filters
- Correct obvious spelling errors using known values above
- If nothing matches a column, return empty dict

Return ONLY a raw JSON object, no explanation, no markdown:
{{
  "filters": {{"column_name": "value", ...}},
  "semantic_query": "the fault/technical part of the query if any, else empty string"
}}

Examples:
Query: "give me all work done by Rahul in night shift"
Output: {{"filters": {{"work_done_by": "Rahul"}}, "semantic_query": ""}}

Query: "Panasonic NPM board jam issues"
Output: {{"filters": {{"model_name": "Panasonic NPM"}}, "semantic_query": "board jam"}}

Query: "show problems where belt was replaced as spare part"
Output: {{"filters": {{"spare_parts": "belt"}}, "semantic_query": "belt replaced"}}

Query: "how many records for Heller 1809 reflow oven"
Output: {{"filters": {{"machine": "Reflow Oven", "model_name": "Heller 1809"}}, "semantic_query": ""}}

Now extract from:
Query: {query}
Output:"""

    try:
        raw = await ask_ollama(prompt, max_tokens=200, model=OLLAMA_MODEL_FAST)
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return {}
        parsed = json.loads(m.group(0))
        result = {}
        raw_filters = parsed.get("filters", {})
        semantic_q  = parsed.get("semantic_query", "").strip()

        for col, val in raw_filters.items():
            if col in ("smd_line", "line_no"):          # line numbers come from regex only
                continue
            if col in _schema.columns and val and str(val).strip():
                clean_val = str(val).strip()
                if not _llm_value_grounded(clean_val, query):
                    continue
                known_vals = _schema.column_values.get(col, [])
                if known_vals and _schema.columns[col].get("match_type") not in ("semantic",):
                    corrected = fuzzy_match_value(clean_val, known_vals)
                    if corrected:
                        clean_val = corrected
                result[col] = clean_val

        if semantic_q:
            result["_semantic_query"] = semantic_q

        return result
    except Exception as e:
        print(f"[LLM Filter] Extraction failed: {e}")
        return {}

def _infer_requested_fields(query: str, filters: dict) -> list[str]:
    q = query.lower()
    if filters.get("work_done_by"):
        return ["work_done_by", "machine", "problem", "solution", "date", "shift", "loss_time"]
    if re.search(r"\bonly\s+problem", q):
        return ["problem"]
    if re.search(r"\bonly\s+solution", q):
        return ["solution"]
    if re.search(r"\ball\s+detail", q):
        all_cols = list(_schema.columns.keys()) if _schema._ready else [
            "problem", "solution", "date", "work_done_by", "machine",
            "loss_time", "shift", "spare_parts"
        ]
        return [c for c in all_cols if c not in _SYSTEM_COLS]
    return ["problem", "solution", "loss_time", "date"]

# ─────────────────────────────────────────────────────────────────────────────
# ★★★ GENERIC SQL BUILDER — STRICT FILTERING ★★★
# ─────────────────────────────────────────────────────────────────────────────

_SPECIAL_FILTER_COLS = {"date_from", "date_to", "year", "line_no", "smd_line", "_semantic_query"}
_META_KEYS = {"_response_type", "_has_fault_keyword", "_requested_fields", "_semantic_query"}


def _build_where(filters: dict, semantic_ids: Optional[list[str]] = None) -> tuple[str, list]:
    """
    Build a strict WHERE clause. All conditions are AND-ed together (hard gate).

    Match-type semantics:
      - exact   → LOWER(col) = LOWER(?)        (e.g. shift)
      - numeric → col = ?                       (e.g. loss_time, line_no)
      - partial → LOWER(col) LIKE LOWER(%?%)    (e.g. machine, model, spares)
      - semantic/date → handled elsewhere (ChromaDB / dedicated date logic)

    Special columns (line_no, year, date_from/to) are handled explicitly below,
    using the always-normalized iso_date column for correct comparisons.
    """
    wheres: list[str] = []
    params: list = []

    for key, value in filters.items():
        if value in (None, "", []) or key in _META_KEYS or key in _SPECIAL_FILTER_COLS:
            continue
        col_meta = _schema.columns.get(key) if _schema._ready else None
        if not col_meta:
            continue

        match_type = col_meta.get("match_type", "partial")
        sql_col    = col_meta["sql_col"]

        if match_type == "exact":                       # STRICT equality (e.g. shift)
            wheres.append(f"LOWER({sql_col}) = LOWER(?)")
            params.append(str(value).strip())

        elif match_type == "numeric":
            try:
                wheres.append(f"{sql_col} = ?")
                params.append(float(value))
            except (ValueError, TypeError):
                pass

        elif match_type in ("partial", "text"):         # machine, model, worker, spares
            if sql_col == "work_done_by":
                sub = []
                for p in str(value).strip().split():
                    sub.append(f"LOWER({sql_col}) LIKE LOWER(?)")
                    params.append(f"%{p}%")
                if sub:
                    wheres.append("(" + " AND ".join(sub) + ")")
            else:
                wheres.append(f"LOWER({sql_col}) LIKE LOWER(?)")
                params.append(f"%{value}%")
        # "semantic" / "date" match types are intentionally skipped here

    # ── Exact line number (integer column) ──
    line_no = filters.get("line_no")
    if line_no not in (None, ""):
        try:
            wheres.append("line_no = ?")
            params.append(int(line_no))
        except (ValueError, TypeError):
            pass

    # ── Year via iso_date (always YYYY-MM-DD, so this is exact) ──
    year = filters.get("year")
    if year and not filters.get("date_from") and not filters.get("date_to"):
        wheres.append("SUBSTR(iso_date,1,4) = ?")
        params.append(str(year))

    # ── Date range via iso_date (lexical compare is correct for ISO dates) ──
    date_from = filters.get("date_from")
    date_to   = filters.get("date_to")
    if date_from:
        wheres.append("(iso_date != '' AND iso_date >= ?)")
        params.append(str(date_from))
    if date_to:
        wheres.append("(iso_date != '' AND iso_date <= ?)")
        params.append(str(date_to))

    # ── Restrict to a set of semantically-matched ids (used by hybrid path) ──
    if semantic_ids is not None:
        if not semantic_ids:
            wheres.append("1 = 0")                       # no semantic match → empty result
        else:
            placeholders = ",".join("?" * len(semantic_ids))
            wheres.append(f"chroma_id IN ({placeholders})")
            params.extend(semantic_ids)

    where_clause = ("WHERE " + " AND ".join(wheres)) if wheres else ""
    return where_clause, params


def build_sql_from_filters(
    filters: dict,
    offset: int = 0,
    limit: int = PAGE_SIZE,
    count_only: bool = False,
    semantic_ids: Optional[list[str]] = None,
) -> tuple[str, list]:
    where_clause, params = _build_where(filters, semantic_ids)
    response_type = filters.get("_response_type", "list")

    if count_only:
        return f"SELECT COUNT(*) as cnt FROM mttr_records {where_clause}", params

    if response_type == "sum_downtime":
        return f"SELECT SUM(loss_time) as total_loss FROM mttr_records {where_clause}", params

    if response_type == "analytics_shift":
        return (
            f"SELECT shift, COUNT(*) as cnt, AVG(loss_time) as avg_loss "
            f"FROM mttr_records {where_clause} GROUP BY shift ORDER BY cnt DESC"
        ), params

    if response_type == "analytics_machine":
        return (
            f"SELECT machine, COUNT(*) as cnt, AVG(loss_time) as avg_loss "
            f"FROM mttr_records {where_clause} GROUP BY machine ORDER BY cnt DESC"
        ), params

    if response_type == "analytics_worker":
        return (
            f"SELECT work_done_by, COUNT(*) as cnt "
            f"FROM mttr_records {where_clause} GROUP BY work_done_by ORDER BY cnt DESC"
        ), params

    if response_type == "analytics_problem":
        return (
            f"SELECT problem, COUNT(*) as cnt "
            f"FROM mttr_records {where_clause} GROUP BY problem ORDER BY cnt DESC LIMIT 10"
        ), params

    order = "ORDER BY CASE WHEN loss_time < 0 THEN 1 ELSE 0 END, loss_time ASC"
    sql = f"SELECT * FROM mttr_records {where_clause} {order} LIMIT ? OFFSET ?"
    return sql, params + [limit, offset]


def fetch_candidate_ids(filters: dict) -> list[str]:
    """ALL chroma_ids matching the structured filters — the hard gate, no limit."""
    where_clause, params = _build_where(filters)
    conn = get_sqlite_conn()
    try:
        rows = conn.execute(
            f"SELECT chroma_id FROM mttr_records {where_clause}", params
        ).fetchall()
        return [r["chroma_id"] for r in rows if r["chroma_id"]]
    finally:
        conn.close()


def fetch_rows_by_chroma_ids(ids: list[str]) -> list[dict]:
    """Fetch full rows for a set of chroma_ids (used to backfill the hybrid path)."""
    if not ids:
        return []
    conn = get_sqlite_conn()
    try:
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT * FROM mttr_records WHERE chroma_id IN ({placeholders})", ids
        ).fetchall()
        return sqlite_rows_to_meta(rows)
    finally:
        conn.close()


def sql_count(filters: dict, semantic_ids: Optional[list[str]] = None) -> int:
    sql, params = build_sql_from_filters(filters, count_only=True, semantic_ids=semantic_ids)
    conn = get_sqlite_conn()
    try:
        row = conn.execute(sql, params).fetchone()
        return row["cnt"] if row else 0
    finally:
        conn.close()


def sql_fetch(filters: dict, offset: int = 0, limit: int = PAGE_SIZE,
              semantic_ids: Optional[list[str]] = None) -> list[dict]:
    sql, params = build_sql_from_filters(filters, offset=offset, limit=limit, semantic_ids=semantic_ids)
    conn = get_sqlite_conn()
    try:
        rows = conn.execute(sql, params).fetchall()
        return sqlite_rows_to_meta(rows)
    finally:
        conn.close()


def sql_analytics(filters: dict):
    sql, params = build_sql_from_filters(filters)
    conn = get_sqlite_conn()
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# QUERY PLANNER
# ─────────────────────────────────────────────────────────────────────────────

def plan_execution(filters: dict, has_semantic_query: bool, sqlite_ready: bool) -> str:
    response_type = filters.get("_response_type", "list")
    if response_type in ("count", "sum_downtime", "analytics_shift",
                          "analytics_machine", "analytics_worker", "analytics_problem"):
        return "analytics"

    if _schema._ready:
        struct_cols = [c for c in _schema.columns
                       if _schema.columns[c].get("match_type") not in ("semantic",)]
    else:
        struct_cols = ["machine", "model_name", "work_done_by", "shift", "spare_parts"]

    has_struct = any(filters.get(k) for k in struct_cols) or \
                 any(filters.get(k) for k in ("line_no", "year", "date_from", "date_to"))
    wants_semantic = bool(filters.get("_has_fault_keyword")) or has_semantic_query

    if not sqlite_ready:
        return "semantic_filter" if has_struct else "semantic_only"
    if has_struct and wants_semantic:
        return "hybrid"
    if has_struct:
        return "sql_only"          # pure structured filter → strict + complete
    return "semantic_only"

# ─────────────────────────────────────────────────────────────────────────────
# QUERY CACHE
# ─────────────────────────────────────────────────────────────────────────────
_query_cache: dict[str, dict] = {}


def cache_key(query: str, filters: dict) -> str:
    raw = json.dumps({"q": query.lower().strip(), "f": filters}, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()


def cache_get(key: str) -> Optional[dict]:
    return _query_cache.get(key)


def cache_set(key: str, value: dict):
    if len(_query_cache) >= 500:
        oldest = next(iter(_query_cache))
        del _query_cache[oldest]
    _query_cache[key] = value


# ─────────────────────────────────────────────────────────────────────────────
# SYNONYM MAP & QUERY EXPANSION
# ─────────────────────────────────────────────────────────────────────────────
SYNONYMS: dict[str, list[str]] = {
    "grab card":       ["grab card", "grabcard", "camera board"],
    "board jam":       ["board jam", "pcb jam", "conveyor jam", "pcb stuck"],
    "overheating":     ["overheating", "overheat", "high temperature", "thermal fault"],
    "nozzle clog":     ["nozzle clog", "nozzle blocked", "nozzle jam"],
    "feeder error":    ["feeder error", "feeder fault", "feeder jam"],
    "solder bridge":   ["solder bridge", "bridging", "solder short"],
    "pick error":      ["pick error", "pick fault", "no pick"],
    "vision error":    ["vision error", "camera error", "vision fault"],
    "conveyor":        ["conveyor", "transport", "belt", "track"],
    "reflow":          ["reflow oven", "reflow", "oven", "soldering oven"],
    "wave":            ["wave solder", "wave machine", "wave soldering"],
    "stencil printer": ["stencil printer", "spi", "screen printer", "paste printer"],
    "pick and place":  ["pick and place", "pnp", "pick & place", "mounter"],
}

# ─────────────────────────────────────────────────────────────────────────────
# LANGUAGE DETECTION  (English ↔ Hinglish)
# ─────────────────────────────────────────────────────────────────────────────

# Common Hindi function words / verbs in Roman script.
# These almost never appear in a purely English technical sentence.
_HINGLISH_MARKERS: frozenset = frozenset({
    # verbs & auxiliaries
    "hai", "hain", "tha", "thi", "ho", "hoga", "hogi", "hoge",
    "kar", "karo", "karna", "karta", "karti", "karte", "kiya", "kiye",
    "raha", "rahi", "rahe", "rukna", "ruk", "aana", "aaya", "aayi", "aaye",
    "jao", "jata", "jati", "jate", "lena", "lelo", "dena", "dedo",
    "dekhna", "dekho", "dekh", "batao", "batana",
    "chalana", "chalu", "band", "nikalo",
    "sakta", "sakti", "sakte", "chahiye", "lagta", "lagti", "lagte",
    # question words
    "kya", "kyun", "kyunki", "kyu", "kaise", "kab", "kahan", "kaun",
    "kitna", "kitne", "kitni",
    # negation
    "nahi", "nhi", "na",
    # conjunctions / discourse
    "aur", "ya", "lekin", "magar", "toh", "tou", "phir", "fir",
    "abhi", "pehle", "baad", "baar",
    # postpositions
    "mein", "pe", "upar", "neeche", "se", "ne", "ko", "ka", "ke",
    # pronouns
    "hum", "aap", "tum", "woh", "yeh", "isko", "usko", "inhe", "unhe",
    "mujhe", "tumhe", "humhe",
    # quantity / degree
    "bahut", "thoda", "zyada", "kam", "bilkul", "sab", "kuch",
    # adjectives
    "theek", "sahi", "galat", "accha", "bura", "purana", "naya",
})

# Extra Hinglish verb/aux/contraction forms used for RETRIEVAL cleanup only.
# These dilute keyword_score if left in (e.g. "aarha", "karu" drop kw below the
# admission threshold), so they must be stripped before semantic search.
_HINGLISH_EXTRA_FILLER: frozenset = frozenset({
    "aaya", "aayi", "aaye", "aarha", "aaraha", "arha", "araha", "aa",
    "rha", "rhe", "rhi",
    "karu", "karun", "karoon", "kru", "krun", "krna", "krke", "krdo",
    "kardo", "kardu", "krdu", "kare", "karein", "karenge", "karoge",
    "hatau", "hataun", "hataye", "hatana", "hatado",
    "nikalu", "nikalun", "nikalna",
    "badlu", "badlun", "badalna", "badal",
    "solve", "fix", "repair", "check", "thik", "theek", "sahi", "kaise",
    "kaisa", "kaisi", "hua", "hui", "huye", "issue", "problem", "dikkat",
})

_HINGLISH_FILLER_ALL = _HINGLISH_MARKERS | _HINGLISH_EXTRA_FILLER

def _strip_hinglish_filler(query: str) -> str:
    """
    Drop Hindi filler + generic action words so only the technical fault terms
    remain for embedding + BM25 retrieval. This keeps keyword_score high enough
    to clear the admission threshold.
    'probe pin wear issue aarha kaise sahi karu' -> 'probe pin wear'
    """
    tokens = re.findall(r"[A-Za-z0-9]+", query)
    kept   = [t for t in tokens if t.lower() not in _HINGLISH_FILLER_ALL]
    cleaned = " ".join(kept).strip()
    return cleaned if len(cleaned) >= 3 else query


def detect_language(text: str) -> str:
    """
    Returns 'hinglish' if the text appears to contain Hindi in Roman script,
    'english' otherwise.

    Decision rule: ≥2 Hinglish marker tokens, OR marker-to-total-token ratio ≥ 12 %.
    This avoids false-positives on purely English sentences that happen to
    contain a word like "par" or "ka".
    """
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    if not tokens:
        return "english"
    hits = sum(1 for t in tokens if t in _HINGLISH_MARKERS)
    if hits >= 2 or (hits >= 1 and hits / len(tokens) >= 0.12):
        return "hinglish"
    return "english"


def _lang_instruction(lang: str) -> str:
    """Prompt suffix that instructs the model on output language."""
    if lang == "hinglish":
        return (
            "\n\nLANGUAGE INSTRUCTION (MANDATORY — follow exactly):\n"
            "Reply in natural Hinglish: Hindi words in Roman/English script mixed with English technical terms.\n"
            "• Keep ALL technical terms in English: sensor, motor, PLC, conveyor, alarm, fault, bearing, "
            "belt, machine, maintenance, downtime, spare part, reflow, solder, feeder, nozzle, PCB, etc.\n"
            "• Write Hindi words in Roman script — NO Devanagari script at all.\n"
            "• Style example: 'Belt misalignment ho sakta hai. Pehle tension check karo, "
            "phir sensor alignment verify karo.'\n"
            "• Do NOT translate technical acronyms or machine model numbers.\n"
            "• Keep the tone practical and easy for a shop-floor technician to understand.\n"
        )
    return "\n\nLANGUAGE INSTRUCTION: Respond in clear, professional English.\n"


async def _hinglish_analytics(english_text: str) -> str:
    """
    Lightly convert a programmatically-built analytics string to Hinglish.
    Falls back to the original if the model call fails.
    """
    # prompt = (
    #     "Convert this maintenance analytics response to natural Hinglish "
    #     "(Hindi in Roman script, English technical terms kept as-is).\n"
    #     "Keep all numbers, machine names, percentages, and durations in English.\n"
    #     "Output ONLY the converted text — no explanation, no markdown fences:\n\n"
    #     f"{english_text}\n\nHINGLISH VERSION:"
    # )
    prompt = (
    "Convert this maintenance analytics response to natural Hinglish.\n"
    "STRICT: Use ONLY Roman/Latin script. NO Devanagari characters at all.\n"
    "Keep all numbers in Western digits (11, 27), and keep machine names, "
    "percentages, and durations in English.\n"
    "Output ONLY the converted text — no explanation, no markdown:\n\n"
    f"{english_text}\n\nHINGLISH (Roman script only):"
   )
    try:
        return await ask_ollama(prompt, max_tokens=500)
    except Exception:
        return english_text

_corpus_vocab: set[str] = set()


def expand_query(query: str) -> str:
    q = query.lower()
    extras = set()
    for canonical, variants in SYNONYMS.items():
        for v in variants:
            if v in q:
                extras.update(variants)
                extras.add(canonical)
    return (query + " " + " ".join(extras)) if extras else query


def normalize_query(query: str) -> str:
    if not _corpus_vocab:
        return query
    tokens = re.findall(r"[a-zA-Z0-9]+", query)
    corrected_tokens = []
    changed = False
    for tok in tokens:
        t = tok.lower()
        if len(t) >= 5 and t not in _corpus_vocab:
            best_word, best_dist = None, 2
            for word in _corpus_vocab:
                if abs(len(word) - len(t)) > 2:
                    continue
                d = _edit_distance(t, word)
                if d < best_dist:
                    best_dist = d
                    best_word = word
            if best_word:
                corrected_tokens.append(best_word)
                changed = True
                continue
        corrected_tokens.append(t)
    if not changed:
        return query
    result = query
    for original, corrected in zip(tokens, corrected_tokens):
        if original.lower() != corrected:
            result = re.sub(re.escape(original), corrected, result, count=1, flags=re.IGNORECASE)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# DATE FILTER PARSING
# ─────────────────────────────────────────────────────────────────────────────
_MONTH_MAP = {
    "jan": 1, "january": 1, "feb": 2, "february": 2,
    "mar": 3, "march": 3, "apr": 4, "april": 4, "may": 5,
    "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def _parse_date_from_parts(day_str, month_str, year_str) -> Optional[date]:
    try:
        year  = int(year_str)  if year_str  else datetime.now().year
        month = int(month_str) if month_str and month_str.isdigit() else _MONTH_MAP.get((month_str or "").lower())
        day   = int(day_str)   if day_str   else 1
        if not month:
            return None
        return date(year, month, day)
    except Exception:
        return None


def parse_date_filter(query: str) -> Optional[dict]:
    q = query
    range_pat = re.compile(
        r"between\s+(\d{1,2})?\s*([a-zA-Z]+)\s*(\d{4})?\s+and\s+(\d{1,2})?\s*([a-zA-Z]+)\s*(\d{4})?",
        re.IGNORECASE,
    )
    m = range_pat.search(q)
    if m:
        d1 = _parse_date_from_parts(m.group(1), m.group(2), m.group(3))
        d2 = _parse_date_from_parts(m.group(4), m.group(5), m.group(6))
        if d1 and d2:
            return {"mode": "range", "start": d1, "end": d2,
                    "label": f"between {d1.strftime('%d %b %Y')} and {d2.strftime('%d %b %Y')}"}

    after_pat = re.compile(r"(?:after|since|from)\s+(\d{1,2})?\s*([a-zA-Z]+)\s*(\d{4})?", re.IGNORECASE)
    m = after_pat.search(q)
    if m:
        d = _parse_date_from_parts(m.group(1), m.group(2), m.group(3))
        if d:
            return {"mode": "after", "date": d, "label": f"after {d.strftime('%d %b %Y')}"}

    before_pat = re.compile(r"(?:before|until|up to|till)\s+(\d{1,2})?\s*([a-zA-Z]+)\s*(\d{4})?", re.IGNORECASE)
    m = before_pat.search(q)
    if m:
        d = _parse_date_from_parts(m.group(1), m.group(2), m.group(3))
        if d:
            return {"mode": "before", "date": d, "label": f"before {d.strftime('%d %b %Y')}"}

    in_month_pat = re.compile(r"\bin\s+([a-zA-Z]+)\s+(\d{4})\b", re.IGNORECASE)
    m = in_month_pat.search(q)
    if m:
        month_name = m.group(1).lower()
        year       = int(m.group(2))
        month_num  = _MONTH_MAP.get(month_name)
        if month_num:
            start = date(year, month_num, 1)
            end   = date(year + 1, 1, 1) if month_num == 12 else date(year, month_num + 1, 1)
            return {"mode": "range", "start": start, "end": end,
                    "label": f"in {m.group(1).capitalize()} {year}"}
        
    bare_month_pat = re.compile(
        r"\b(?:for|in|during)\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
        r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
        r"nov(?:ember)?|dec(?:ember)?)\b(?!\s+20\d{2})", re.IGNORECASE)
    m = bare_month_pat.search(q)
    if m:
        month_num = _MONTH_MAP.get(m.group(1).lower()[:3])
        if month_num:
            year = date.today().year
            start = date(year, month_num, 1)
            end   = date(year + 1, 1, 1) if month_num == 12 else date(year, month_num + 1, 1)
            return {"mode": "range", "start": start, "end": end,
                    "label": f"in {m.group(1).capitalize()} {year}"}

    in_year_pat = re.compile(r"\bin\s+(20\d{2})\b", re.IGNORECASE)
    m = in_year_pat.search(q)
    if m:
        year = int(m.group(1))
        return {"mode": "year", "year": year, "label": f"in {year}"}

    last_pat = re.compile(r"last\s+(\d+)?\s*(day|days|week|weeks|month|months)", re.IGNORECASE)
    m = last_pat.search(q)
    if m:
        from datetime import timedelta
        num   = int(m.group(1)) if m.group(1) else 1
        unit  = m.group(2).lower().rstrip("s")
        today = date.today()
        delta = timedelta(days=num) if unit == "day" else timedelta(weeks=num) if unit == "week" else timedelta(days=30 * num)
        start = today - delta
        return {"mode": "after", "date": start, "label": f"last {m.group(0).split('last ', 1)[-1]}"}

    specific_pat = re.compile(
        r"(\d{1,2})\s*(?:st|nd|rd|th)?\s+([a-zA-Z]+)\s+(\d{4})"
        r"|([a-zA-Z]+)\s+(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(\d{4})"
        r"|(\d{4})[/-](\d{1,2})[/-](\d{1,2})"
        r"|(\d{1,2})[/-](\d{1,2})[/-](\d{4})",
        re.IGNORECASE,
    )
    m = specific_pat.search(q)
    if m:
        if m.group(1):
            d = _parse_date_from_parts(m.group(1), m.group(2), m.group(3))
        elif m.group(4):
            d = _parse_date_from_parts(m.group(5), m.group(4), m.group(6))
        elif m.group(7):
            d = _parse_date_from_parts(m.group(9), m.group(8), m.group(7))
        elif m.group(10):
            d = _parse_date_from_parts(m.group(10), m.group(11), m.group(12))
        else:
            d = None
        if d:
            before_any = re.search(r"\b(before|until|prior to|up to)\b", q, re.IGNORECASE)
            after_any  = re.search(r"\b(after|since|from|post)\b",       q, re.IGNORECASE)
            if before_any:
                return {"mode": "before", "date": d, "label": f"before {d.strftime('%d %b %Y')}"}
            if after_any:
                return {"mode": "after",  "date": d, "label": f"after {d.strftime('%d %b %Y')}"}
            return {"mode": "range", "start": d, "end": d, "label": f"on {d.strftime('%d %b %Y')}"}
    return None


def _parse_record_date(date_str: str) -> Optional[date]:
    if not date_str:
        return None
    date_str = str(date_str).strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y",
                "%d %b %Y", "%d %B %Y", "%b %d %Y", "%B %d %Y",
                "%Y/%m/%d", "%d-%b-%Y", "%d-%B-%Y"):
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            pass
    m = re.match(r"(\d{4})", date_str)
    if m:
        try:
            return date(int(m.group(1)), 1, 1)
        except Exception:
            pass
    return None


def apply_date_filter(records: list[dict], date_filter: Optional[dict]) -> list[dict]:
    if not date_filter:
        return records
    mode = date_filter.get("mode")
    filtered = []
    for meta in records:
        rec_date = _parse_record_date(str(meta.get("date", "")))
        if rec_date is None:
            continue
        if mode == "after"  and rec_date >= date_filter["date"]:  filtered.append(meta)
        elif mode == "before" and rec_date <= date_filter["date"]:  filtered.append(meta)
        elif mode == "range"  and date_filter["start"] <= rec_date <= date_filter["end"]: filtered.append(meta)
        elif mode == "year"   and rec_date.year == date_filter["year"]: filtered.append(meta)
        else: filtered.append(meta) if mode not in ("after","before","range","year") else None
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT FORMAT DETECTION
# ─────────────────────────────────────────────────────────────────────────────
_FORMAT_PHRASE_RE = re.compile(
    r"\b(?:"
    r"in\s+(?:a\s+)?tabular\s+form(?:at)?|in\s+(?:a\s+)?table\s+form(?:at)?|"
    r"as\s+(?:a\s+)?table|in\s+table|"
    r"(?:as|in)\s+(?:a\s+)?spreadsheet|(?:as|in)\s+(?:a\s+)?grid|"
    r"(?:as|in|into)\s+(?:a\s+)?(?:bar\s+|pie\s+)?chart|"
    r"(?:as|in|into)\s+(?:a\s+)?graph|"
    r"(?:as|in|into)\s+(?:a\s+)?tree(?:\s+diagram)?|"
    r"(?:as|in|into)\s+(?:a\s+)?mind\s*map|"
    r"(?:as|in|into)\s+(?:a\s+)?flow\s*chart|"
    r"(?:as|in|into)\s+(?:a\s+)?flow\s*diagram|"
    r"(?:as|in|into)\s+(?:a\s+)?diagram|"
    r"in\s+(?:a\s+)?bullet\s*points?|as\s+bullets?|"
    r"in\s+(?:a\s+)?numbered\s+list|in\s+point\s+form|in\s+points?|"
    r"(?:as|in)\s+(?:a\s+)?summary|"
    r"give\s+(?:it\s+|me\s+)?in\s+\w+\s+form(?:at)?"
    r")\b",
    re.IGNORECASE,
)


def _strip_format_phrases(query: str) -> str:
    """Remove presentation-format phrases so they don't contaminate retrieval."""
    if not query:
        return query
    cleaned = _FORMAT_PHRASE_RE.sub(" ", query)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned if len(cleaned) >= 3 else query


def detect_output_format(query: str) -> str:
    q = query.lower()
    if re.search(r"\b(tree|mind\s*map|hierarchy|hierarchical)\b", q):          return "tree"
    if re.search(r"\b(graph)\b", q):                                          return "graph"
    if re.search(r"\b(table|tabular|spreadsheet|grid)\b", q):                 return "table"
    if re.search(r"\b(bullet|bullets|bullet point)\b", q):                    return "bullets"
    if re.search(r"\b(numbered list|numbered|serial number|list out)\b", q):  return "numbered"
    if re.search(r"\b(summary|summarise|summarize|brief|concise)\b", q):      return "summary"
    if re.search(r"\b(points?|point form|in points|list)\b", q):              return "bullets"
    return "default"


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI INIT
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="MTTR Local AI Assistant — Schema-Driven v7")

_client = chromadb.PersistentClient(path=DB_PATH)
_ef     = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL_PATH)


def get_collection():
    try:
        return _client.get_collection(name=COLLECTION_NAME, embedding_function=_ef)
    except Exception:
        raise HTTPException(status_code=503, detail="MTTR database not found. Run clean_excel.py first.")


def get_tsg_collection():
    try:
        return _client.get_collection(name=TSG_COLLECTION_NAME, embedding_function=_ef)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# LOSS TIME SORTING
# ─────────────────────────────────────────────────────────────────────────────
def sort_by_loss_time(records: list[dict]) -> list[dict]:
    def sort_key(meta: dict) -> float:
        lt = meta.get("loss_time", -1)
        try:
            lt = float(lt)
        except (TypeError, ValueError):
            lt = -1.0
        return lt if lt >= 0 else float("inf")
    return sorted(records, key=sort_key)


# ─────────────────────────────────────────────────────────────────────────────
# IN-MEMORY BM25 INDEX
# ─────────────────────────────────────────────────────────────────────────────
class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1; self.b = b
        self.docs: list[str] = []; self.metadatas: list[dict] = []; self.ids: list[str] = []
        self._tf: list[dict] = []; self._idf: dict[str, float] = {}; self._avgdl: float = 0.0

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def build(self, docs: list[str], metadatas: list[dict], ids: list[str]):
        self.docs = docs; self.metadatas = metadatas; self.ids = ids
        N = len(docs); self._tf = []; df: dict[str, int] = defaultdict(int); lengths = []
        for doc in docs:
            tokens = self._tokenize(doc); lengths.append(len(tokens))
            freq: dict[str, int] = defaultdict(int)
            for t in tokens: freq[t] += 1
            self._tf.append(dict(freq))
            for term in freq: df[term] += 1
        self._avgdl = sum(lengths) / max(N, 1)
        self._idf = {term: math.log((N - n + 0.5) / (n + 0.5) + 1) for term, n in df.items()}

    def search(self, query: str, top_n: int = TOP_K) -> list[tuple[float, dict, str]]:
        if not self.docs: return []
        tokens = self._tokenize(query); scores: list[float] = []
        for i, tf in enumerate(self._tf):
            dl = sum(tf.values()); sc = 0.0
            for t in tokens:
                if t not in tf: continue
                idf = self._idf.get(t, 0.0)
                sc += idf * (tf[t] * (self.k1 + 1)) / (tf[t] + self.k1 * (1 - self.b + self.b * dl / self._avgdl))
            scores.append(sc)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(scores[i], self.metadatas[i], self.ids[i]) for i, _ in ranked[:top_n]]


_bm25 = BM25Index(); _bm25_tsg = BM25Index()
_bm25_ready = False; _bm25_tsg_ready = False


@app.on_event("startup")
async def startup():
    init_sqlite()
    print(f"[SQLite] Initialized at {SQLITE_PATH}")
    try:
        conn = get_sqlite_conn()
        _schema.load(conn)
        _schema.refresh_values(conn)
        conn.close()
        print(f"[Schema] Loaded {len(_schema.columns)} columns: {list(_schema.columns.keys())}")
    except Exception as e:
        print(f"[Schema] Could not load schema: {e}")
    await build_bm25()
    await auto_sync_sqlite()
    try:
        await ask_ollama("ok", max_tokens=1)   # load model into memory before first user query
        print("[Ollama] Model pre-warmed.")
    except Exception as e:
        print(f"[Ollama] Pre-warm skipped: {e}")


async def build_bm25():
    global _bm25_ready, _bm25_tsg_ready, _corpus_vocab
    try:
        col      = get_collection()
        all_data = col.get(include=["documents", "metadatas"])
        enriched = []
        for meta, doc_id in zip(all_data["metadatas"], all_data["ids"]):
            m = dict(meta); m["chroma_id"] = doc_id; enriched.append(m)
        _bm25.build(docs=all_data["documents"], metadatas=enriched, ids=all_data["ids"])
        vocab: set[str] = set()
        for doc in all_data["documents"]:
            for tok in re.findall(r"[a-z]{4,}", doc.lower()): vocab.add(tok)
        _corpus_vocab = vocab
        _bm25_ready = True
        print(f"[BM25-MTTR] Index built with {len(all_data['documents'])} documents.")
    except Exception as e:
        print(f"[BM25-MTTR] Could not build index: {e}")

    try:
        tsg_col = get_tsg_collection()
        if tsg_col:
            tsg_data = tsg_col.get(include=["documents", "metadatas"])
            _bm25_tsg.build(docs=tsg_data["documents"], metadatas=tsg_data["metadatas"], ids=tsg_data["ids"])
            _bm25_tsg_ready = True
            print(f"[BM25-TSG] Index built with {len(tsg_data['documents'])} documents.")
    except Exception as e:
        print(f"[BM25-TSG] Could not build index: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# KEYWORD HELPERS
# ─────────────────────────────────────────────────────────────────────────────
STOPWORDS = {
    "the","a","an","is","are","was","were","be","been","have","has","do","does",
    "did","will","would","could","should","may","might","of","in","on","at","to",
    "for","with","by","from","about","into","through","and","or","but","not",
    "what","how","why","when","where","which","me","my","i","we","you","it","its",
    "this","that","give","tell","show","list","any","some","all","top","common",
    "problem","issue","fault","error","help","fix","please","can","most","there",
    "also","only",
}


def extract_keywords(query: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9_\-]+", query.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def keyword_score(meta: dict, keywords: list[str]) -> float:
    if not keywords: return 0.0
    haystack = " ".join([
        str(meta.get("problem", "")), str(meta.get("solution", "")),
        str(meta.get("machine", "")), str(meta.get("model_name", "")),
    ]).lower()
    hits = sum(1 for kw in keywords if kw in haystack)
    return hits / len(keywords)


# ─────────────────────────────────────────────────────────────────────────────
# RRF MERGE
# ─────────────────────────────────────────────────────────────────────────────
def rrf_merge(
    vector_hits: list[tuple[float, dict]],
    bm25_hits: list[tuple[float, dict, str]],
    k: int = 60,
) -> list[tuple[float, dict]]:
    def meta_key(m: dict) -> str:
        return f"{m.get('machine','')}||{m.get('model_name','')}||{m.get('problem', m.get('issue', ''))}||{m.get('solution', m.get('corrective', ''))}"

    rrf: dict[str, float] = defaultdict(float)
    meta_store: dict[str, dict] = {}
    for rank, (dist, meta) in enumerate(vector_hits, start=1):
        key = meta_key(meta); rrf[key] += 1.0 / (k + rank); meta_store[key] = meta
    for rank, (score, meta, _) in enumerate(bm25_hits, start=1):
        key = meta_key(meta); rrf[key] += 1.0 / (k + rank); meta_store[key] = meta
    merged = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
    return [(score, meta_store[key]) for key, score in merged]


# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA
# ─────────────────────────────────────────────────────────────────────────────
# async def ask_ollama(prompt: str, max_tokens: int = MAX_TOKENS) -> str:
#     payload = {
#         "model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
#         "options": {"num_predict": max_tokens, "temperature": 0.2, "top_p": 0.9, "repeat_penalty": 1.15},
#     }
#     async with httpx.AsyncClient(timeout=180.0) as client:
#         try:
#             resp = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
#             resp.raise_for_status()
#             return resp.json().get("response", "").strip()
#         except httpx.ConnectError:
#             raise HTTPException(status_code=503, detail="Ollama is not running. Start with: ollama serve")

# Module-level singleton — avoids building a new connection pool every call
_ollama_client = httpx.AsyncClient(timeout=180.0)


async def ask_ollama(prompt: str, max_tokens: int = MAX_TOKENS, model: str = OLLAMA_MODEL) -> str:
    payload = {
        "model": model, "prompt": prompt, "stream": False,
        "keep_alive": "30m",          # don't unload the model between the calls in one request
        "options": {
            "num_predict": max_tokens, "temperature": 0.2,
            "top_p": 0.9, "repeat_penalty": 1.15,
            "num_ctx": 4096,           # explicit, comfortably fits your prompts; avoids a larger default
        },
    }
    try:
        resp = await _ollama_client.post(f"{OLLAMA_URL}/api/generate", json=payload)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Ollama is not running. Start with: ollama serve")

# ─────────────────────────────────────────────────────────────────────────────
# MULTI-QUERY EXPANSION
# ─────────────────────────────────────────────────────────────────────────────
async def generate_query_variants(query: str) -> list[str]:
    prompt = f"""You are a search query optimizer for an industrial maintenance database.
Rewrite the following maintenance query into 3 different phrasings using different technical terms or synonyms.
Output ONLY a JSON array of 3 strings. No explanation:
["variant 1", "variant 2", "variant 3"]

Original: {query}
Output:"""
    try:
        raw = await ask_ollama(prompt, max_tokens=150)
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        m   = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            variants = json.loads(m.group(0))
            if isinstance(variants, list):
                seen = {query.lower()}; result = [query]
                for v in variants:
                    if isinstance(v, str) and v.lower() not in seen:
                        seen.add(v.lower()); result.append(v.strip())
                return result[:4]
    except Exception:
        pass
    return [query]


# ─────────────────────────────────────────────────────────────────────────────
# MACHINE & MODEL HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_all_machines() -> list[str]:
    col = get_collection()
    res = col.get(include=["metadatas"])
    return sorted({m.get("machine", "") for m in res["metadatas"] if m.get("machine")})


def get_all_models() -> list[str]:
    col = get_collection()
    res = col.get(include=["metadatas"])
    return sorted({m.get("model_name", "") for m in res["metadatas"] if m.get("model_name")})


def build_chroma_where_filter(machine_filter: Optional[str], model_filter: Optional[str]) -> Optional[dict]:
    conditions = []
    if machine_filter: conditions.append({"machine": {"$eq": machine_filter}})
    if model_filter:   conditions.append({"model_name": {"$eq": model_filter}})
    if not conditions:  return None
    if len(conditions) == 1: return conditions[0]
    return {"$and": conditions}


# ─────────────────────────────────────────────────────────────────────────────
# SEMANTIC SEARCH
# ─────────────────────────────────────────────────────────────────────────────
async def semantic_search(
    query: str,
    machine_filter: Optional[str] = None,
    model_filter: Optional[str] = None,
    use_multi_query: bool = False,
) -> tuple[list[dict], float]:
    collection = get_collection()
    where      = build_chroma_where_filter(machine_filter, model_filter)
    expanded   = expand_query(query)
    base_queries = list({query, expanded})
    if use_multi_query:
        variants    = await generate_query_variants(expanded)
        all_queries = list({*base_queries, *variants})
    else:
        all_queries = base_queries

    all_vector_hits: list[tuple[float, dict]] = []
    seen_vector: set[str] = set()
    n_results = min(TOP_K, max(collection.count(), 1))

    try:
        res = collection.query(
            query_texts=all_queries, n_results=n_results, where=where,
            include=["metadatas", "distances", "ids"],
        )
        for qi in range(len(all_queries)):
            for meta, dist, doc_id in zip(res["metadatas"][qi], res["distances"][qi], res["ids"][qi]):
                meta = dict(meta); meta["chroma_id"] = doc_id
                key = f"{meta.get('machine')}||{meta.get('model_name','')}||{meta.get('problem')}||{meta.get('solution')}"
                if key not in seen_vector:
                    seen_vector.add(key); all_vector_hits.append((dist, meta))
    except Exception:
        pass

    # for q in all_queries:
    #     try:
    #         res = collection.query(
    #             query_texts=[q], n_results=n_results, where=where,
    #             include=["metadatas", "distances", "ids"],
    #         )
    #         for meta, dist, doc_id in zip(res["metadatas"][0], res["distances"][0], res["ids"][0]):
    #             meta = dict(meta); meta["chroma_id"] = doc_id
    #             key = f"{meta.get('machine')}||{meta.get('model_name','')}||{meta.get('problem')}||{meta.get('solution')}"
    #             if key not in seen_vector:
    #                 seen_vector.add(key); all_vector_hits.append((dist, meta))
    #     except Exception:
    #         pass
    all_vector_hits.sort(key=lambda x: x[0])

    bm25_hits: list[tuple[float, dict, str]] = []
    if _bm25_ready:
        for q in all_queries:
            for score, meta, doc_id in _bm25.search(q, top_n=TOP_K):
                if machine_filter and meta.get("machine", "").lower() != machine_filter.lower(): continue
                if model_filter   and meta.get("model_name", "").lower() != model_filter.lower(): continue
                bm25_hits.append((score, meta, doc_id))
        seen_bm = set(); dedup_bm: list[tuple[float, dict, str]] = []
        for score, meta, doc_id in sorted(bm25_hits, key=lambda x: x[0], reverse=True):
            key = f"{meta.get('machine')}||{meta.get('model_name','')}||{meta.get('problem')}||{meta.get('solution')}"
            if key not in seen_bm: seen_bm.add(key); dedup_bm.append((score, meta, doc_id))
        bm25_hits = dedup_bm[:TOP_K]

    merged = rrf_merge(all_vector_hits, bm25_hits)
    if not merged: return [], 0.0
    best_score = merged[0][0]
    keywords   = extract_keywords(query)
    reranked   = []
    for rrf_score, meta in merged:
        kw = keyword_score(meta, keywords)
        reranked.append((0.65 * rrf_score + 0.35 * kw, meta))
    reranked.sort(key=lambda x: x[0], reverse=True)

    final_records: list[dict] = []
    if not keywords:
        final_records = [meta for _, meta in reranked[:TOP_K]]
    else:
        for final_score, meta in reranked:
            kw = keyword_score(meta, keywords)
            if final_score >= CONFIDENCE_THR or (kw >= 0.50 and final_score >= CONFIDENCE_THR * 0.60):
                final_records.append(meta)
        if not final_records and reranked:
            final_records = [meta for _, meta in reranked[:PAGE_SIZE]]
    # for final_score, meta in reranked:
    #     kw = keyword_score(meta, keywords)
    #     if final_score >= CONFIDENCE_THR or (kw >= 0.50 and final_score >= CONFIDENCE_THR * 0.60):
    #         final_records.append(meta)

    return sort_by_loss_time(final_records), best_score


# ─────────────────────────────────────────────────────────────────────────────
# TSG RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────
async def tsg_retrieve(
    query: str,
    machine_filter: Optional[str] = None,
    use_multi_query: bool = False,
) -> list[dict]:
    tsg_col = get_tsg_collection()
    if tsg_col is None: return []
    expanded     = expand_query(query)
    base_queries = list({query, expanded})
    if use_multi_query:
        variants    = await generate_query_variants(expanded)
        all_queries = list({*base_queries, *variants})
    else:
        all_queries = base_queries

    where = {"machine": {"$eq": machine_filter}} if machine_filter else None
    all_vector_hits: list[tuple[float, dict]] = []
    seen_vector: set[str] = set()
    n_results = min(TOP_K, max(tsg_col.count(), 1))

    for q in all_queries:
        try:
            res = tsg_col.query(query_texts=[q], n_results=n_results, where=where, include=["metadatas", "distances"])
            for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
                key = f"{meta.get('machine')}||{meta.get('issue')}||{meta.get('corrective')}"
                if key not in seen_vector: seen_vector.add(key); all_vector_hits.append((dist, meta))
        except Exception:
            pass
    all_vector_hits.sort(key=lambda x: x[0])

    bm25_hits: list[tuple[float, dict, str]] = []
    if _bm25_tsg_ready:
        for q in all_queries:
            for score, meta, doc_id in _bm25_tsg.search(q, top_n=TOP_K):
                if machine_filter and meta.get("machine", "").lower() != machine_filter.lower(): continue
                bm25_hits.append((score, meta, doc_id))
        seen_bm = set(); dedup: list[tuple[float, dict, str]] = []
        for score, meta, doc_id in sorted(bm25_hits, key=lambda x: x[0], reverse=True):
            key = f"{meta.get('machine')}||{meta.get('issue')}||{meta.get('corrective')}"
            if key not in seen_bm: seen_bm.add(key); dedup.append((score, meta, doc_id))
        bm25_hits = dedup[:TOP_K]

    merged = rrf_merge(all_vector_hits, bm25_hits)
    keywords = extract_keywords(query)
    final_records: list[dict] = []
    for rrf_score, meta in merged:
        kw = keyword_score(meta, keywords)
        if 0.65 * rrf_score + 0.35 * kw >= CONFIDENCE_THR or kw >= 0.30:
            final_records.append(meta)
    return final_records


# ─────────────────────────────────────────────────────────────────────────────
# INTENT DETECTION
# ─────────────────────────────────────────────────────────────────────────────
CONVERSATIONAL_GREETINGS = re.compile(
    r"^(hi|hello|hey|good\s*(morning|afternoon|evening|day)|howdy|sup|what'?s\s*up"
    r"|greetings|namaste|hii+|helo|hellow|heya|yo)\b", re.IGNORECASE,
)
CONVERSATIONAL_SMALLTALK = re.compile(
    r"^(thanks?|thank\s*you|ty|thx|ok(ay)?|alright|cool|got\s*it|nice|great|awesome"
    r"|perfect|sure|understood|makes?\s*sense|good\s*(to\s*know)?|bye|goodbye|see\s*ya"
    r"|that'?s?\s*(all|fine|good|great|helpful)|no\s*(problem|issue|worries)"
    r"|you'?re\s*(welcome|great|helpful))\s*[.!]?\s*$", re.IGNORECASE,
)
TSG_EXPLICIT_REQUEST = re.compile(
    r"\b(troubleshooting\s*guide|tsg|trouble\s*shoot(ing)?\s*guide"
    r"|from\s*(the\s*)?(troubleshooting|tsg)\b"
    r"|troubleshooting\s*(data|record|entry|result)"
    r"|guide\s*(answer|result|record))\b", re.IGNORECASE,
)
TSG_YES_FOLLOWUP = re.compile(
    r"^\s*(yes|yeah|yep|yup|sure|ok(ay)?|please|go ahead|show|do it|give|y)\s*[.!]?\s*$", re.IGNORECASE,
)

# Selected-text "Ask MTTR" follow-ups and conceptual questions that should be
# answered by reasoning over context, NOT by re-querying the database.
_EXPLAIN_FOLLOWUP = re.compile(
    r"^\s*(?:regarding\s+.*?:\s*)?"              # the "Regarding '...':" prefix your UI adds
    r"(explain|elaborate|what\s+does\s+(?:this|that|it)\s+mean|"
    r"what\s+do\s+you\s+mean|tell\s+me\s+more|clarify|"
    r"why|how\s+come|what\s+is\s+meant|break\s+(?:this|it)\s+down|"
    r"simplify|in\s+simple\s+(?:words|terms)|"
    # Hinglish
    r"samjhao|samjha\s+do|matlab\s+kya|iska\s+matlab|"
    r"detail\s+(?:me|mein)\s+batao)\b",
    re.IGNORECASE,
)


def detect_intent(query: str) -> str:
    q = query.lower().strip()
    if CONVERSATIONAL_GREETINGS.match(q): return "conversational"
    if CONVERSATIONAL_SMALLTALK.match(q): return "conversational"
    if TSG_EXPLICIT_REQUEST.search(query): return "tsg_lookup"
    context_diagram = re.search(
        r"\b(above|previous|last|that|this|the\s*(above|previous|last|mentioned|given|said)).{0,40}"
        r"(diagram|flowchart|flow|chart|visual|steps?|process)", q,
    ) or re.search(
        r"\b(diagram|flowchart|flow).{0,40}"
        r"\b(above|previous|last|that|this|the\s*(above|previous|last|mentioned))\b", q,
    )
    if context_diagram: return "diagram_context"
    if re.search(
        r"\b(diagram|flowchart|flow chart|flow diagram|visuali[sz]e|draw|sketch|chart|"
        r"process flow|workflow|schematic|block diagram|step diagram|timeline|show.*(flow|process|steps))\b", q
    ): return "diagram"
    is_definition = bool(re.search(
        r"^(what is|what are|explain|describe|define|tell me about\s+(?!.*\b(work|done|perform|machine|problem|fault|issue|shift|repair|fix|maintenance)\b)|overview of|how does .+ work)\b", q
    ))
    has_maintenance = bool(re.search(
        r"\b(work|done|perform|machine|problem|fault|issue|shift|repair|fix|maintenance|"
        r"record|history|solution|downtime|loss|spare|part|who|when|which|date|"
        r"202[0-9]|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|"
        r"night|general|morning|day\s*shift|"
        r"kaam|kaise|theek|thik|sahi|kharab|dikkat|samasya|badlu|nikalu|hataye?)\b", q
    ))
    if is_definition and not has_maintenance: return "general"
    if re.search(
        r"\b(not working|not picking|not moving|not running|not responding|not printing|not feeding|"
        r"broken|damaged|failing|alarm|keeps?\s+on|sudden|loud|rattling|shaking)\b"
        r"|\b(how (do i|to) (fix|solve|repair|resolve))\b"
        r"|\b(help me fix|give me a solution for|how to resolve)\b"
        # ── Hinglish ──
        r"|\bkaise\s+(?:theek|thik|sahi|solve|fix|repair|karu|kare|badlu|nikalu|hataye?)\b"
        r"|\b(?:theek|thik|sahi|solve|fix)\s+kar(?:u|o|na|e|en)?\b", q
    ): return "troubleshoot"
    return "db_lookup"


# ─────────────────────────────────────────────────────────────────────────────
# ANALYTICS HANDLER
# ─────────────────────────────────────────────────────────────────────────────
# async def handle_analytics_query(filters: dict, query: str, lang: str = "english") -> str:
#     response_type = filters.get("_response_type", "")
#     rows = sql_analytics(filters)

#     def _describe() -> str:
#         parts = []
#         if filters.get("work_done_by"): parts.append(f" done by **{filters['work_done_by']}**")
#         if filters.get("machine"):      parts.append(f" on {filters['machine']}")
#         if filters.get("smd_line"):     parts.append(f" on Line {filters['smd_line']}")
#         if filters.get("year"):         parts.append(f" in {filters['year']}")
#         if filters.get("shift"):        parts.append(f" in {filters['shift']} shift")
#         if filters.get("date_from"):    parts.append(f" from {filters['date_from']}")
#         if filters.get("date_to"):      parts.append(f" to {filters['date_to']}")
#         for k, v in filters.items():
#             if k.startswith("_") or k in ("machine","smd_line","year","shift","date_from","date_to","work_done_by"):
#                 continue
#             if v: parts.append(f" [{k}: {v}]")
#         return "".join(parts)

#     desc = _describe()

#     if response_type == "count":
#         count = rows[0]["cnt"] if rows else 0
#         return f"Found **{count} maintenance record{'s' if count != 1 else ''}**{desc}."
#     if response_type == "sum_downtime":
#         total = rows[0]["total_loss"] if rows and rows[0]["total_loss"] is not None else 0
#         if total <= 0: return f"No downtime data found{desc}."
#         if total < 60: return f"Total downtime{desc}: **{total:.0f} minutes**."
#         return f"Total downtime{desc}: **{total/60:.1f} hours** ({total:.0f} minutes)."
#     if response_type == "analytics_shift":
#         if not rows: return "No shift data found in the database."
#         lines = [f"**Failures by shift**{desc}:\n"]
#         for row in rows:
#             shift = row["shift"] or "Unknown"; cnt = row["cnt"]; avg = row["avg_loss"]
#             avg_s = f", avg downtime: {avg:.0f} min" if avg and avg > 0 else ""
#             lines.append(f"• **{shift}**: {cnt} fault{'s' if cnt != 1 else ''}{avg_s}")
#         return "\n".join(lines)
#     if response_type == "analytics_machine":
#         if not rows: return "No machine data found in the database."
#         lines = [f"**Problems by machine**{desc}:\n"]
#         for i, row in enumerate(rows[:10], 1):
#             machine = row["machine"] or "Unknown"; cnt = row["cnt"]; avg = row["avg_loss"]
#             avg_s   = f", avg downtime: {avg:.0f} min" if avg and avg > 0 else ""
#             lines.append(f"{i}. **{machine}**: {cnt} record{'s' if cnt != 1 else ''}{avg_s}")
#         return "\n".join(lines)
#     if response_type == "analytics_worker":
#         if not rows: return "No worker data found in the database."
#         lines = [f"**Work done by person**{desc}:\n"]
#         for i, row in enumerate(rows[:10], 1):
#             worker = row["work_done_by"] or "Unknown"; cnt = row["cnt"]
#             lines.append(f"{i}. **{worker}**: {cnt} job{'s' if cnt != 1 else ''}")
#         return "\n".join(lines)
#     if response_type == "analytics_problem":
#         if not rows: return "No problem data found in the database."
#         lines = [f"**Most common problems**{desc}:\n"]
#         for i, row in enumerate(rows[:10], 1):
#             problem = row["problem"] or "Unknown"; cnt = row["cnt"]
#             lines.append(f"{i}. {problem} — **{cnt} occurrence{'s' if cnt != 1 else ''}**")
#         return "\n".join(lines)
#     return "Analytics query completed."

async def handle_analytics_query(filters: dict, query: str, lang: str = "english") -> str:
    response_type = filters.get("_response_type", "")
    rows = sql_analytics(filters)

    def _describe() -> str:
        parts = []
        if filters.get("work_done_by"): parts.append(f" done by **{filters['work_done_by']}**")
        if filters.get("machine"):      parts.append(f" on {filters['machine']}")
        if filters.get("smd_line"):     parts.append(f" on Line {filters['smd_line']}")
        if filters.get("year"):         parts.append(f" in {filters['year']}")
        if filters.get("shift"):        parts.append(f" in {filters['shift']} shift")
        if filters.get("date_from"):    parts.append(f" from {filters['date_from']}")
        if filters.get("date_to"):      parts.append(f" to {filters['date_to']}")
        for k, v in filters.items():
            if k.startswith("_") or k in ("machine","smd_line","year","shift","date_from","date_to","work_done_by"):
                continue
            if v: parts.append(f" [{k}: {v}]")
        return "".join(parts)

    desc = _describe()

    if response_type == "count":
        count = rows[0]["cnt"] if rows else 0
        result = f"Found **{count} maintenance record{'s' if count != 1 else ''}**{desc}."

    elif response_type == "sum_downtime":
        total = rows[0]["total_loss"] if rows and rows[0]["total_loss"] is not None else 0
        if total <= 0:
            result = f"No downtime data found{desc}."
        elif total < 60:
            result = f"Total downtime{desc}: **{total:.0f} minutes**."
        else:
            result = f"Total downtime{desc}: **{total/60:.1f} hours** ({total:.0f} minutes)."

    elif response_type == "analytics_shift":
        if not rows:
            result = "No shift data found in the database."
        else:
            lines = [f"**Failures by shift**{desc}:\n"]
            for row in rows:
                shift = row["shift"] or "Unknown"; cnt = row["cnt"]; avg = row["avg_loss"]
                avg_s = f", avg downtime: {avg:.0f} min" if avg and avg > 0 else ""
                lines.append(f"• **{shift}**: {cnt} fault{'s' if cnt != 1 else ''}{avg_s}")
            result = "\n".join(lines)

    elif response_type == "analytics_machine":
        if not rows:
            result = "No machine data found in the database."
        else:
            lines = [f"**Problems by machine**{desc}:\n"]
            for i, row in enumerate(rows[:10], 1):
                machine = row["machine"] or "Unknown"; cnt = row["cnt"]; avg = row["avg_loss"]
                avg_s   = f", avg downtime: {avg:.0f} min" if avg and avg > 0 else ""
                lines.append(f"{i}. **{machine}**: {cnt} record{'s' if cnt != 1 else ''}{avg_s}")
            result = "\n".join(lines)

    elif response_type == "analytics_worker":
        if not rows:
            result = "No worker data found in the database."
        else:
            lines = [f"**Work done by person**{desc}:\n"]
            for i, row in enumerate(rows[:10], 1):
                worker = row["work_done_by"] or "Unknown"; cnt = row["cnt"]
                lines.append(f"{i}. **{worker}**: {cnt} job{'s' if cnt != 1 else ''}")
            result = "\n".join(lines)

    # elif response_type == "analytics_problem":
    #     if not rows:
    #         result = "No problem data found in the database."
    #     else:
    #         lines = [f"**Most common problems**{desc}:\n"]
    #         for i, row in enumerate(rows[:10], 1):
    #             problem = row["problem"] or "Unknown"; cnt = row["cnt"]
    #             lines.append(f"{i}. {problem} — **{cnt} occurrence{'s' if cnt != 1 else ''}**")
    #         result = "\n".join(lines)
    elif response_type == "analytics_problem":
        # Cluster free-text problems instead of GROUP BY exact string, so the
        # same fault written three different ways (across models) ranks as ONE.
        raw_rows = _fetch_analytics_rows(filters)          # respects machine/model/shift/year/month
        clusters = _cluster_problems([r.get("problem", "") for r in raw_rows])
        if not clusters:
            result = f"No problem records found{desc}."
        else:
            total = sum(c["count"] for c in clusters) or 1
            top   = clusters[0]
            top_pct = round(top["count"] / total * 100, 1)
            lines = [f"**Most frequently occurring problem**{desc}:\n"]
            lines.append(
                f"🔝 **{top['label']}** — {top['count']} "
                f"occurrence{'s' if top['count'] != 1 else ''} "
                f"({top_pct}% of {total} recorded failures)\n"
            )
            if len(clusters) > 1:
                lines.append("Full ranking:")
                for i, c in enumerate(clusters[:10], 1):
                    pct = round(c["count"] / total * 100, 1)
                    lines.append(f"{i}. {c['label']} — **{c['count']}** ({pct}%)")
            result = "\n".join(lines)

    else:
        result = "Analytics query completed."

    if lang == "hinglish":
        result = await _hinglish_analytics(result)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# ★★★ DATA-CHART ANALYTICS ENGINE ★★★
# All data computed in SQL/Python. Chart JSON emitted deterministically.
# The LLM is NOT involved in producing numbers — only optional Hinglish prose.
# ─────────────────────────────────────────────────────────────────────────────

_WANTS_CHART = re.compile(
    r"(?:chart|graph|plot|histogram|"
    r"visuali[sz]e|visuali[sz]ation|"
    r"\bpie\b|\bbar\b|"
    r"\btrend|distribution|breakdown|percentage|percent)",
    re.IGNORECASE,
)
# Explicit chart-type cues (longest/most-specific first).
_CHART_TYPE_PATTERNS = [
    (r"\bhorizontal\s+bar\b",                 "hbar"),
    (r"\b(stacked\s+bar|stacked)\b",          "stacked_bar"),
    (r"\b(bar\s*chart|bar\s*graph|bar)\b",    "bar"),
    (r"\b(pie\s*chart|pie|donut)\b",          "pie"),
    (r"\b(line\s*chart|line\s*graph)\b",      "line"),
    (r"\barea\s*chart\b",                     "area"),
    (r"\bscatter(\s*plot)?\b",                "scatter"),
    (r"\b(heat\s*map|heatmap)\b",             "heatmap"),
]


def _detect_chart_dimension(query: str) -> str:
    q = query.lower()
    if re.search(r"\b(monthly|per\s+month|by\s+month|over\s+time|timeline|month[\s-]*wise)\b", q): return "month"
    if re.search(r"\b(yearly|annual|per\s+year|by\s+year|year[\s-]*wise)\b", q):                    return "year"
    if re.search(r"\btrend", q):                                                                    return "month"
    if re.search(r"\b(by\s+model|across\s+models|each\s+model|model[\s-]*wise|per\s+model)\b", q):   return "model_name"
    if re.search(r"\b(by\s+machine|across\s+machines|each\s+machine|machine[\s-]*wise|per\s+machine)\b", q): return "machine"
    if re.search(r"\b(by\s+shift|shift[\s-]*wise|per\s+shift)\b", q):                                return "shift"
    if re.search(r"\b(by\s+line|line[\s-]*wise|per\s+line|across\s+lines)\b", q):                    return "line_no"
    return "problem"


def _detect_chart_metric(query: str) -> str:
    q = query.lower()
    if re.search(r"\bmttr\b|\b(average|avg|mean)\s+(downtime|repair|loss)\b|\brepair\s+time\b", q): return "mttr"
    if re.search(r"\b(downtime|loss\s*time|hours?\s+lost|total\s+downtime)\b", q):                  return "downtime"
    return "count"


def _detect_chart_spec(query: str) -> Optional[dict]:
    """Return {chart_type, dimension, metric} only when a chart is explicitly requested."""
    if not _WANTS_CHART.search(query):
        return None
    chart_type = None
    for pat, ct in _CHART_TYPE_PATTERNS:
        if re.search(pat, query, re.IGNORECASE):
            chart_type = ct
            break
    dimension = _detect_chart_dimension(query)
    metric    = _detect_chart_metric(query)
    if chart_type is None:                       # generic "chart"/"graph" → pick a sensible default
        chart_type = "line" if dimension in ("month", "year") else \
                     "pie"  if (dimension == "problem" and re.search(r"\b(distribution|percent)", query, re.I)) else \
                     "bar"
    return {"chart_type": chart_type, "dimension": dimension, "metric": metric}


# ── Deterministic problem clustering (majority-wording grouping) ─────────────
_PROBLEM_STOPWORDS = {
    "the","a","an","is","are","was","were","of","in","on","at","to","for","with",
    "and","or","due","because","problem","issue","issues","fault","faults","error",
    "errors","occurred","occurring","happening","happened","machine","not","this",
    "that","it","its","being","been","had","has","have",
}

def _normalize_problem(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", str(text or "").lower())
    keep = [t for t in toks if t not in _PROBLEM_STOPWORDS and len(t) > 1]
    return keep or toks            # if everything got stripped, fall back to raw tokens


def _problem_similarity(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    union = len(sa | sb)
    return len(sa & sb) / union if union else 0.0


def _cluster_problems(problems: list[str], threshold: float = 0.5) -> list[dict]:
    """
    Greedy token-overlap clustering of free-text problems.
    Returns [{label: majority_wording, count: n}] sorted by count desc.
    """
    from collections import Counter
    clusters = []   # {tok: set, wordings: Counter, count: int}
    for p in problems:
        p_clean = str(p or "").strip()
        if not p_clean:
            continue
        toks = _normalize_problem(p_clean)
        best, best_sim = None, 0.0
        for c in clusters:
            sim = _problem_similarity(toks, c["tok"])
            if sim > best_sim:
                best_sim, best = sim, c
        if best and best_sim >= threshold:
            best["count"] += 1
            best["wordings"][p_clean] += 1
            best["tok"] = list(set(best["tok"]) | set(toks))      # widen for recall
        else:
            clusters.append({"tok": toks, "wordings": Counter({p_clean: 1}), "count": 1})
    out = [{"label": c["wordings"].most_common(1)[0][0], "count": c["count"]} for c in clusters]
    out.sort(key=lambda x: x["count"], reverse=True)
    return out


def _fetch_analytics_rows(filters: dict) -> list[dict]:
    """ALL rows matching the structured filters (no pagination) — for aggregation."""
    where_clause, params = _build_where(filters)
    conn = get_sqlite_conn()
    try:
        rows = conn.execute(f"SELECT * FROM mttr_records {where_clause}", params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _agg_by_key(rows: list[dict], key_fn, metric: str):
    """Aggregate rows by key_fn(row). metric ∈ count|downtime|mttr."""
    from collections import defaultdict
    counts, sums, valid = defaultdict(int), defaultdict(float), defaultdict(int)
    for r in rows:
        k = key_fn(r)
        if k in (None, ""):
            continue
        counts[k] += 1
        try:
            lt = float(r.get("loss_time", -1))
        except (TypeError, ValueError):
            lt = -1.0
        if lt >= 0:
            sums[k] += lt
            valid[k] += 1
    out = {}
    for k in counts:
        if metric == "downtime":
            out[k] = round(sums[k], 1)
        elif metric == "mttr":
            out[k] = round(sums[k] / valid[k], 1) if valid[k] else 0.0
        else:
            out[k] = counts[k]
    return out


_METRIC_LABELS = {"count": "Failure Count", "downtime": "Total Downtime (min)", "mttr": "Avg Repair Time (min)"}
_DIM_LABELS    = {"problem": "Problem", "machine": "Machine", "model_name": "Model",
                  "shift": "Shift", "line_no": "Line", "month": "Month", "year": "Year"}

_VIZ_TYPE = {"bar": "bar_chart", "hbar": "horizontal_bar_chart", "pie": "pie_chart",
             "line": "line_chart", "area": "area_chart", "stacked_bar": "stacked_bar_chart",
             "scatter": "scatter_plot", "heatmap": "heatmap"}


def _month_label(ym: str) -> str:
    try:
        y, m = ym.split("-")[:2]
        names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        return f"{names[int(m)]} {y}"
    except Exception:
        return ym


def build_analytics_visualization(filters: dict, query: str, chart_spec: dict) -> tuple[Optional[dict], str]:
    """Build (visualization_json, summary_text). Returns (None, msg) when no data."""
    chart_type = chart_spec["chart_type"]
    dimension  = chart_spec["dimension"]
    metric     = chart_spec["metric"]
    scope      = build_filter_context_str(filters) or "all records"

    rows = _fetch_analytics_rows(filters)
    if not rows:
        return None, f"No maintenance records found for {scope}, so there's nothing to chart."

    metric_label = _METRIC_LABELS[metric]
    dim_label    = _DIM_LABELS.get(dimension, dimension.replace("_", " ").title())

    # Stacked / scatter / heatmap need 2-D data we don't compute yet → fall back to bar.
    rt = _VIZ_TYPE.get(chart_type, "bar_chart")
    if rt in ("stacked_bar_chart", "scatter_plot", "heatmap"):
        rt = "bar_chart"

    # ── PROBLEM distribution (clustered, % within filter) ──
    if dimension == "problem":
        clusters = _cluster_problems([r.get("problem", "") for r in rows])
        if not clusters:
            return None, f"No problem descriptions recorded for {scope}."
        total = sum(c["count"] for c in clusters)
        top   = clusters[:8]
        data  = [{"label": _short_label(c["label"], 6), "value": c["count"],
                  "percent": round(c["count"] / total * 100, 1)} for c in top]
        top1 = data[0]
        summary = (f"{top1['label']} is the most common problem for {scope}, "
                   f"making up {top1['percent']}% of {total} recorded failures.")
        viz = {"responseType": rt, "title": f"Problem Distribution — {scope}",
               "xAxis": dim_label, "yAxis": "Failure Count", "data": data, "summary": summary}
        return viz, summary

    # ── MONTH / YEAR trend ──
    if dimension in ("month", "year"):
        span = 7 if dimension == "month" else 4
        agg  = _agg_by_key(
            rows,
            lambda r, s=span: (str(r.get("iso_date", "") or "")[:s]
                               if len(str(r.get("iso_date", "") or "")) >= s else None),
            metric,
        )
        if not agg:
            return None, f"No dated records found for {scope} to build a trend."
        keys = sorted(agg.keys())
        data = [{"label": _month_label(k) if dimension == "month" else k, "value": agg[k]} for k in keys]
        peak = max(data, key=lambda d: d["value"])
        direction = ("increasing" if data[-1]["value"] > data[0]["value"]
                     else "decreasing" if data[-1]["value"] < data[0]["value"] else "stable")
        summary = (f"{metric_label} for {scope} peaked in {peak['label']} ({peak['value']}); "
                   f"the trend is {direction} across {len(data)} {dimension}s.")
        if rt not in ("line_chart", "area_chart", "bar_chart"):
            rt = "line_chart"
        viz = {"responseType": rt, "title": f"{metric_label} by {dim_label} — {scope}",
               "xAxis": dim_label, "yAxis": metric_label, "data": data, "summary": summary}
        return viz, summary

    # ── CATEGORY dimensions (machine / model / shift / line) ──
    def keyfn(r, d=dimension):
        v = r.get(d)
        if v in (None, ""):
            return None
        return f"Line {v}" if d == "line_no" else str(v).strip()

    agg = _agg_by_key(rows, keyfn, metric)
    if not agg:
        return None, f"No data to group by {dim_label.lower()} for {scope}."
    items = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:12]
    total = sum(v for _, v in items) or 1
    data  = [{"label": _short_label(k, 5), "value": v,
              "percent": round(v / total * 100, 1)} for k, v in items]
    top1 = data[0]
    summary = (f"{top1['label']} has the highest {metric_label.lower()} by {dim_label.lower()} "
               f"for {scope} ({top1['value']}, {top1['percent']}% of the total).")
    viz = {"responseType": rt, "title": f"{metric_label} by {dim_label} — {scope}",
           "xAxis": dim_label, "yAxis": metric_label, "data": data, "summary": summary}
    return viz, summary


# ─────────────────────────────────────────────────────────────────────────────
# FORMAT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _loss_time_label(meta: dict) -> str:
    lt = meta.get("loss_time", -1)
    try: lt = float(lt)
    except (TypeError, ValueError): return ""
    if lt < 0:  return ""
    if lt < 60: return f"{lt:.0f} min downtime"
    return f"{lt/60:.1f} hr downtime"


def format_records_for_prompt(records: list) -> str:
    if not records: return ""
    blocks = []
    for i, meta in enumerate(records, 1):
        lt_label  = _loss_time_label(meta)
        lt_line   = f"Loss Time  : {lt_label}" if lt_label else "Loss Time  : Unknown"
        lines = [f"[Record {i}]", f"Machine  : {meta.get('machine', 'Unknown')}"]
        if meta.get("smd_line"):     lines.append(f"SMD Line : {meta['smd_line']}")
        if meta.get("model_name"):   lines.append(f"Model    : {meta['model_name']}")
        lines += [f"Problem  : {meta.get('problem', '')}", f"Solution : {meta.get('solution', '')}", lt_line]
        if meta.get("work_done_by"): lines.append(f"Done By  : {meta['work_done_by']}")
        if meta.get("date"):         lines.append(f"Date     : {meta['date']}")
        if meta.get("shift"):        lines.append(f"Shift    : {meta['shift']}")
        if meta.get("spare_parts"):  lines.append(f"Spare Parts: {meta['spare_parts']}")
        known = {"machine","smd_line","model_name","problem","solution","loss_time",
                 "work_done_by","date","shift","spare_parts","image_b64","image_name",
                 "image_mime","chroma_id","id"}
        for k, v in meta.items():
            if k not in known and v and str(v).strip():
                lines.append(f"{k.replace('_',' ').title()}: {v}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def format_tsg_records_for_prompt(records: list) -> str:
    if not records: return ""
    blocks = []
    for i, meta in enumerate(records, 1):
        lines = [
            f"[TSG Entry {i}]", f"Line No.    : {meta.get('line_no', 'N/A')}",
            f"Machine     : {meta.get('machine', 'Unknown')}",
            f"Issue       : {meta.get('issue', '')}",
            f"Cause       : {meta.get('cause', '')}",
            f"Corrective  : {meta.get('corrective', '')}",
        ]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def build_filter_context_str(filters: dict) -> str:
    parts = []
    label_map = {
        "work_done_by": "work done by", "machine": "machine",
        "model_name": "model",          "smd_line": "SMD line",
        "year": "year",                 "shift": "shift",
        "date_from": "from",            "date_to": "to",
        "spare_parts": "spare part",
    }
    for k, v in filters.items():
        if k.startswith("_") or not v: continue
        if k in ("date_from", "date_to"):
            parts.append(f"{label_map.get(k, k)} {v}")
        elif k in label_map:
            parts.append(f"{label_map[k]}: {v}")
        elif k not in _SPECIAL_FILTER_COLS:
            parts.append(f"{k.replace('_',' ')}: {v}")
    return ", ".join(parts) if parts else ""


def build_field_aware_prompt(query: str, records: list, filters: dict, date_label: str = "") -> str:
    requested_fields = filters.get("_requested_fields", ["problem", "solution", "loss_time", "date"])
    response_type    = filters.get("_response_type", "list")
    filter_ctx       = build_filter_context_str(filters)
    date_note        = f" (filtered: {date_label})" if date_label else ""
    scope_str        = filter_ctx or "all records"

    if not records:
        worker = filters.get("work_done_by") or filters.get("worker")
        if worker:
            return (f"No maintenance records found in the database for work done by **{worker}**"
                    + (f" {date_note}" if date_note else "")
                    + ". Please check the name spelling or try a different filter.")
        return (f"No maintenance records found in the database matching your query"
                + (f" ({date_note})" if date_note else "")
                + f" for {scope_str}. Try broadening your search.")

    def build_record_block(meta: dict, idx: int) -> str:
        lines = [f"[Record {idx}]", f"Machine    : {meta.get('machine', '')}"]
        if meta.get("smd_line"):     lines.append(f"SMD Line   : {meta['smd_line']}")
        field_map = {
            "model":       f"Model      : {meta.get('model_name', '')}",
            "model_name":  f"Model      : {meta.get('model_name', '')}",
            "problem":     f"Problem    : {meta.get('problem', '')}",
            "solution":    f"Solution   : {meta.get('solution', '')}",
            "loss_time":   f"Loss Time  : {_loss_time_label(meta) or 'Unknown'}",
            "worker":      f"Done By    : {meta.get('work_done_by', '')}",
            "work_done_by":f"Done By    : {meta.get('work_done_by', '')}",
            "date":        f"Date       : {meta.get('date', '')}",
            "shift":       f"Shift      : {meta.get('shift', '')}",
            "spare_parts": f"Spare Parts: {meta.get('spare_parts', '')}",
        }
        for field in requested_fields:
            if field in field_map and field not in ("machine", "smd_line"):
                val = field_map[field].split(":", 1)[-1].strip()
                if val: lines.append(field_map[field])
        known = set(field_map.keys()) | {"machine", "smd_line", "image_b64", "image_name", "image_mime", "chroma_id", "id"}
        for k, v in meta.items():
            if k not in known and v and str(v).strip():
                lines.append(f"{k.replace('_',' ').title()}: {v}")
        return "\n".join(lines)

    records_block = "\n\n".join(build_record_block(m, i + 1) for i, m in enumerate(records))

    wants_worker = "work_done_by" in requested_fields or "worker" in requested_fields
    if wants_worker:
        instruction = "List all work by the person in 'Done By'. For each: date, machine, SMD line (if any), problem, solution, shift, loss time."
    elif response_type == "summary":
        instruction = "Write a SHORT summary (3-5 sentences): record count, main machines, most common problems, any patterns."
    elif response_type == "table":
        instruction = f"Present as a markdown table with columns: {', '.join(requested_fields)}. One row per record. Use '—' for missing."
    elif response_type == "troubleshoot":
        return build_troubleshoot_prompt(query, records, filters.get("machine"), filters.get("model_name"))
    else:
        instruction = f"Present each record showing: {', '.join(requested_fields)}. Number each. Be concise."

    return f"""You are a senior industrial maintenance engineer answering a specific question.

USER QUESTION: {query}
ACTIVE FILTERS: {scope_str}{date_note}

━━━ MAINTENANCE RECORDS ━━━
{records_block}
━━━ END ━━━

INSTRUCTION: {instruction}

STRICT RULES:
- ONLY use data from the records above. Do NOT invent anything.
- Do NOT include fields the user did not ask for.
- Do NOT add preamble. Start directly with the answer.
- If a field is empty/unknown, skip it rather than showing "Unknown".
"""


# ── was: def build_conversational_prompt(query: str) -> str:
def build_conversational_prompt(query: str, lang: str = "english") -> str:
    return (
        f"""You are a friendly industrial maintenance AI assistant called MTTR Assistant.
USER SAYS: "{query}"
Respond naturally and briefly — 1-3 sentences max. Be warm and friendly.
"""
        + _lang_instruction(lang)
    )


# ── was: def build_general_prompt(query: str) -> str:
def build_general_prompt(query: str, lang: str = "english") -> str:
    return (
        f"""You are an expert SMD and industrial maintenance engineer with 20 years of experience.
USER QUESTION: {query}
Answer clearly: 1) What it is, 2) How it works, 3) Where it's used, 4) Common issues (2-3 points).
Keep it concise and suitable for a maintenance technician.
"""
        + _lang_instruction(lang)
    )


# ── was: def build_db_lookup_prompt(query, records, machine, model=None) -> str:
def build_db_lookup_prompt(
    query: str, records: list,
    machine: Optional[str], model: Optional[str] = None,
    lang: str = "english",
) -> str:
    machine_str = machine or "the machine"
    model_str   = f" (Model: {model})" if model else ""
    if records:
        return (
            f"""You are a senior industrial maintenance engineer analysing REAL maintenance records.
USER QUESTION: {query}
━━━ MAINTENANCE RECORDS (lowest downtime first) ━━━
{format_records_for_prompt(records)}
━━━ END ━━━
STRICT RULES:
1. ONLY report problems EXPLICITLY present in the records.
2. Do NOT invent anything.
3. Present records IN ORDER shown.
4. For each: exact symptom, solution, loss time, model if present.
FORMAT: Clear numbered points. Scope: {machine_str}{model_str}.
"""
            + _lang_instruction(lang)
        )
    return (
        f"""You are a senior industrial maintenance engineer.
USER QUESTION: {query}
No matching records found for {machine_str}{model_str}.
Answer using engineering knowledge. Start with: "No records found in database. Answering from engineering knowledge."
"""
        + _lang_instruction(lang)
    )


# ── was: def build_troubleshoot_prompt(query, records, machine, model=None) -> str:
def build_troubleshoot_prompt(
    query: str, records: list,
    machine: Optional[str], model: Optional[str] = None,
    lang: str = "english",
) -> str:
    model_str     = f" (Model: {model})" if model else ""
    scope_str     = f"{machine or 'the machine'}{model_str}"
    records_block = format_records_for_prompt(records) if records else ""
    db_section    = (
        f"━━━ PAST MAINTENANCE RECORDS (lowest downtime first) ━━━\n{records_block}\n━━━ END ━━━"
        if records else f"No matching records found for {scope_str}."
    )
    return (
        f"""You are a senior SMD maintenance engineer helping fix a live fault.
TECHNICIAN REPORTS: {query}
{db_section}
Respond in EXACTLY this format:
MOST LIKELY CAUSE:
[explanation]
RECOMMENDED FIX:
1. [Step 1]
2. [Step 2]
3. [Step 3]
WHY THIS HAPPENS:
[brief technical explanation]
SAFETY NOTE:
[key precaution]
"""
        + _lang_instruction(lang)
    )


# ── was: def build_tsg_prompt(query, tsg_records, machine=None) -> str:
def build_tsg_prompt(
    query: str, tsg_records: list,
    machine: Optional[str] = None,
    lang: str = "english",
) -> str:
    scope_str = machine or "the machine"
    if tsg_records:
        return (
            f"""You are a senior maintenance engineer using the official Troubleshooting Guide.
USER QUESTION / FAULT: {query}
━━━ TROUBLESHOOTING GUIDE ENTRIES ━━━
{format_tsg_records_for_prompt(tsg_records)}
━━━ END ━━━
Base your answer ONLY on the TSG entries. List all matching entries numbered. Machine scope: {scope_str}.
"""
            + _lang_instruction(lang)
        )
    return (
        f"""No matching entries in the Troubleshooting Guide for {scope_str}.
State: "No records found in the Troubleshooting Guide for this topic."
Then give one short engineering knowledge note if helpful.
"""
        + _lang_instruction(lang)
    )


# ── was: def build_field_aware_prompt(query, records, filters, date_label="") -> str:
def build_field_aware_prompt(
    query: str, records: list, filters: dict,
    date_label: str = "", lang: str = "english",
) -> str:
    requested_fields = filters.get("_requested_fields", ["problem", "solution", "loss_time", "date"])
    response_type    = filters.get("_response_type", "list")
    filter_ctx       = build_filter_context_str(filters)
    date_note        = f" (filtered: {date_label})" if date_label else ""
    scope_str        = filter_ctx or "all records"

    if not records:
        worker = filters.get("work_done_by") or filters.get("worker")
        if worker:
            return (
                f"No maintenance records found in the database for work done by **{worker}**"
                + (f" {date_note}" if date_note else "")
                + ". Please check the name spelling or try a different filter."
            )
        return (
            f"No maintenance records found in the database matching your query"
            + (f" ({date_note})" if date_note else "")
            + f" for {scope_str}. Try broadening your search."
        )

    def build_record_block(meta: dict, idx: int) -> str:
        lines = [f"[Record {idx}]", f"Machine    : {meta.get('machine', '')}"]
        if meta.get("smd_line"): lines.append(f"SMD Line   : {meta['smd_line']}")
        field_map = {
            "model":        f"Model      : {meta.get('model_name', '')}",
            "model_name":   f"Model      : {meta.get('model_name', '')}",
            "problem":      f"Problem    : {meta.get('problem', '')}",
            "solution":     f"Solution   : {meta.get('solution', '')}",
            "loss_time":    f"Loss Time  : {_loss_time_label(meta) or 'Unknown'}",
            "worker":       f"Done By    : {meta.get('work_done_by', '')}",
            "work_done_by": f"Done By    : {meta.get('work_done_by', '')}",
            "date":         f"Date       : {meta.get('date', '')}",
            "shift":        f"Shift      : {meta.get('shift', '')}",
            "spare_parts":  f"Spare Parts: {meta.get('spare_parts', '')}",
        }
        for field in requested_fields:
            if field in field_map and field not in ("machine", "smd_line"):
                val = field_map[field].split(":", 1)[-1].strip()
                if val: lines.append(field_map[field])
        known = set(field_map.keys()) | {"machine", "smd_line", "image_b64", "image_name", "image_mime", "chroma_id", "id"}
        for k, v in meta.items():
            if k not in known and v and str(v).strip():
                lines.append(f"{k.replace('_',' ').title()}: {v}")
        return "\n".join(lines)

    records_block = "\n\n".join(build_record_block(m, i + 1) for i, m in enumerate(records))

    wants_worker = "work_done_by" in requested_fields or "worker" in requested_fields
    if wants_worker:
        instruction = "List all work by the person in 'Done By'. For each: date, machine, SMD line (if any), problem, solution, shift, loss time."
    elif response_type == "summary":
        instruction = "Write a SHORT summary (3-5 sentences): record count, main machines, most common problems, any patterns."
    elif response_type == "table":
        instruction = f"Present as a markdown table with columns: {', '.join(requested_fields)}. One row per record. Use '—' for missing."
    elif response_type == "troubleshoot":
        return build_troubleshoot_prompt(query, records, filters.get("machine"), filters.get("model_name"), lang=lang)
    else:
        instruction = f"Present each record showing: {', '.join(requested_fields)}. Number each. Be concise."

    return (
        f"""You are a senior industrial maintenance engineer answering a specific question.

USER QUESTION: {query}
ACTIVE FILTERS: {scope_str}{date_note}

━━━ MAINTENANCE RECORDS ━━━
{records_block}
━━━ END ━━━

INSTRUCTION: {instruction}

STRICT RULES:
- ONLY use data from the records above. Do NOT invent anything.
- Do NOT include fields the user did not ask for.
- Do NOT add preamble. Start directly with the answer.
- If a field is empty/unknown, skip it rather than showing "Unknown".
"""
        + _lang_instruction(lang)
    )


# ── was: def build_table_prompt(query, records, date_label="") -> str:
def build_table_prompt(query: str, records: list, date_label: str = "", lang: str = "english") -> str:
    date_note = f" (filtered: {date_label})" if date_label else ""
    return (
        f"""You are a senior maintenance engineer presenting database records.
USER REQUEST: {query}
━━━ MAINTENANCE RECORDS{date_note} ━━━
{format_records_for_prompt(records)}
━━━ END ━━━
Present as a MARKDOWN TABLE. Columns: # | Machine | Model | Problem | Solution | Loss Time | Done By | Date
Each record = one row. Use "—" for missing. After the table, add one short "Summary:" paragraph.
"""
        + _lang_instruction(lang)
    )


# ── was: def build_summary_prompt(query, records, date_label="") -> str:
def build_summary_prompt(query: str, records: list, date_label: str = "", lang: str = "english") -> str:
    date_note = f" (filtered: {date_label})" if date_label else ""
    return (
        f"""You are a senior maintenance engineer summarising database records.
USER REQUEST: {query}
━━━ MAINTENANCE RECORDS{date_note} ━━━
{format_records_for_prompt(records)}
━━━ END ━━━
Write a SHORT summary (3-6 sentences): total records, machines, common problems, effective solutions, patterns.
"""
        + _lang_instruction(lang)
    )


# ── was: def build_structured_list_prompt(query, records, style, date_label="") -> str:
def build_structured_list_prompt(
    query: str, records: list, style: str,
    date_label: str = "", lang: str = "english",
) -> str:
    date_note = f" (filtered: {date_label})" if date_label else ""
    if style == "numbered":
        instruction = (
            "Present each record as a NUMBERED list (1., 2., 3., …). "
            "For each: Problem, then Solution, plus loss time/date if present. Be concise."
        )
    else:
        instruction = (
            "Present each record as BULLET points. For each record use a top bullet "
            "with the problem and a sub-line with the solution. Be concise."
        )
    return (
        f"""You are a senior maintenance engineer presenting database records.
USER REQUEST: {query}
━━━ MAINTENANCE RECORDS{date_note} ━━━
{format_records_for_prompt(records)}
━━━ END ━━━
{instruction}
STRICT RULES:
- ONLY use data from the records above. Do NOT invent anything.
- Do NOT add preamble. Start directly with the list.
"""
        + _lang_instruction(lang)
    )


# def build_conversational_prompt(query: str) -> str:
#     return f"""You are a friendly industrial maintenance AI assistant called MTTR Assistant.
# USER SAYS: "{query}"
# Respond naturally and briefly — 1-3 sentences max. Be warm and friendly.
# """


# def build_general_prompt(query: str) -> str:
#     return f"""You are an expert SMD and industrial maintenance engineer with 20 years of experience.
# USER QUESTION: {query}
# Answer clearly: 1) What it is, 2) How it works, 3) Where it's used, 4) Common issues (2-3 points).
# Keep it concise and suitable for a maintenance technician.
# """


# def build_db_lookup_prompt(query: str, records: list, machine: Optional[str], model: Optional[str] = None) -> str:
#     machine_str = machine or "the machine"
#     model_str   = f" (Model: {model})" if model else ""
#     if records:
#         return f"""You are a senior industrial maintenance engineer analysing REAL maintenance records.
# USER QUESTION: {query}
# ━━━ MAINTENANCE RECORDS (lowest downtime first) ━━━
# {format_records_for_prompt(records)}
# ━━━ END ━━━
# STRICT RULES:
# 1. ONLY report problems EXPLICITLY present in the records.
# 2. Do NOT invent anything.
# 3. Present records IN ORDER shown.
# 4. For each: exact symptom, solution, loss time, model if present.
# FORMAT: Clear numbered points. Scope: {machine_str}{model_str}.
# """
#     return f"""You are a senior industrial maintenance engineer.
# USER QUESTION: {query}
# No matching records found for {machine_str}{model_str}.
# Answer using engineering knowledge. Start with: "No records found in database. Answering from engineering knowledge."
# """


# def build_troubleshoot_prompt(query: str, records: list, machine: Optional[str], model: Optional[str] = None) -> str:
#     model_str = f" (Model: {model})" if model else ""
#     scope_str = f"{machine or 'the machine'}{model_str}"
#     records_block = format_records_for_prompt(records) if records else ""
#     db_section = f"━━━ PAST MAINTENANCE RECORDS (lowest downtime first) ━━━\n{records_block}\n━━━ END ━━━" if records else f"No matching records found for {scope_str}."
#     return f"""You are a senior SMD maintenance engineer helping fix a live fault.
# TECHNICIAN REPORTS: {query}
# {db_section}
# Respond in EXACTLY this format:
# MOST LIKELY CAUSE:
# [explanation]
# RECOMMENDED FIX:
# 1. [Step 1]
# 2. [Step 2]
# 3. [Step 3]
# WHY THIS HAPPENS:
# [brief technical explanation]
# SAFETY NOTE:
# [key precaution]
# """


# def build_tsg_prompt(query: str, tsg_records: list, machine: Optional[str] = None) -> str:
#     scope_str = machine or "the machine"
#     if tsg_records:
#         return f"""You are a senior maintenance engineer using the official Troubleshooting Guide.
# USER QUESTION / FAULT: {query}
# ━━━ TROUBLESHOOTING GUIDE ENTRIES ━━━
# {format_tsg_records_for_prompt(tsg_records)}
# ━━━ END ━━━
# Base your answer ONLY on the TSG entries. List all matching entries numbered. Machine scope: {scope_str}.
# """
#     return f"""No matching entries in the Troubleshooting Guide for {scope_str}.
# State: "No records found in the Troubleshooting Guide for this topic."
# Then give one short engineering knowledge note if helpful.
# """


# def build_table_prompt(query: str, records: list, date_label: str = "") -> str:
#     date_note = f" (filtered: {date_label})" if date_label else ""
#     return f"""You are a senior maintenance engineer presenting database records.
# USER REQUEST: {query}
# ━━━ MAINTENANCE RECORDS{date_note} ━━━
# {format_records_for_prompt(records)}
# ━━━ END ━━━
# Present as a MARKDOWN TABLE. Columns: # | Machine | Model | Problem | Solution | Loss Time | Done By | Date
# Each record = one row. Use "—" for missing. After the table, add one short "Summary:" paragraph.
# """


# def build_summary_prompt(query: str, records: list, date_label: str = "") -> str:
#     date_note = f" (filtered: {date_label})" if date_label else ""
#     return f"""You are a senior maintenance engineer summarising database records.
# USER REQUEST: {query}
# ━━━ MAINTENANCE RECORDS{date_note} ━━━
# {format_records_for_prompt(records)}
# ━━━ END ━━━
# Write a SHORT summary (3-6 sentences): total records, machines, common problems, effective solutions, patterns.
# """


def build_structured_list_prompt(query: str, records: list, style: str, date_label: str = "") -> str:
    date_note = f" (filtered: {date_label})" if date_label else ""
    if style == "numbered":
        instruction = ("Present each record as a NUMBERED list (1., 2., 3., …). "
                       "For each: Problem, then Solution, plus loss time/date if present. Be concise.")
    else:
        instruction = ("Present each record as BULLET points. For each record use a top bullet "
                       "with the problem and a sub-line with the solution. Be concise.")
    return f"""You are a senior maintenance engineer presenting database records.
USER REQUEST: {query}
━━━ MAINTENANCE RECORDS{date_note} ━━━
{format_records_for_prompt(records)}
━━━ END ━━━
{instruction}
STRICT RULES:
- ONLY use data from the records above. Do NOT invent anything.
- Do NOT add preamble. Start directly with the list.
"""


def build_diagram_prompt(query: str, context: str = "") -> str:
    """
    Build an LLM prompt that converts a maintenance insight into structured
    flowchart/tree JSON. Covers flowcharts, process flows, root-cause trees,
    decision trees, troubleshooting workflows, maintenance procedures, and
    failure-analysis sequences. Output is the same node/edge schema the
    frontend (buildDiagramSVG) already renders.
    """
    ctx_block = f'\nSOURCE CONTENT (base the diagram on this):\n"""\n{context}\n"""\n' if context.strip() else ""
    return f"""You are a maintenance diagram generator. Convert the request into a structured flowchart. Output ONLY valid JSON — no prose, no markdown fences.

USER REQUEST: {query}
{ctx_block}
HOW TO BUILD THE DIAGRAM:
- EXTRACT each distinct action, check, or cause as ONE node.
- IDENTIFY decision points: any "is it X?", "check whether", or branch where the path splits → node type "decision" with Yes/No (or labelled) outgoing edges.
- IDENTIFY dependencies: connect nodes in the order they must logically happen.
- For ROOT-CAUSE or FAILURE-ANALYSIS requests: start node = the symptom/failure, then one branch per candidate cause, each leading to its check and its corrective action.
- For TROUBLESHOOTING / PROCEDURE requests: linear or branching steps from start to resolution, with an escalate/end node for the unresolved path.
- Keep labels SHORT (max 5 words), faithful to the wording, NO decorative text.
- Produce 5–12 nodes connected into ONE coherent flow. Every node except the start must be reachable.

NODE TYPES: start (entry/symptom) | process (action step) | check (inspection) | decision (branch, needs labelled edges) | end (resolution/escalation)

OUTPUT JSON SHAPE (exactly this structure):
{{"title":"<short title>","type":"flowchart","description":"<one sentence>",
"nodes":[{{"id":"n1","label":"Symptom","type":"start"}},{{"id":"n2","label":"Check Bearing","type":"check"}},{{"id":"n3","label":"Worn?","type":"decision"}},{{"id":"n4","label":"Replace Bearing","type":"process"}},{{"id":"n5","label":"Escalate","type":"end"}}],
"edges":[{{"from":"n1","to":"n2","label":""}},{{"from":"n2","to":"n3","label":""}},{{"from":"n3","to":"n4","label":"Yes"}},{{"from":"n3","to":"n5","label":"No"}}]}}

EXAMPLE — input "Motor vibration from bearing wear, shaft misalignment, or loose bolts":
{{"title":"Motor Vibration Diagnosis","type":"flowchart","description":"Root-cause workflow for motor vibration.",
"nodes":[{{"id":"n1","label":"Motor Vibration","type":"start"}},{{"id":"n2","label":"Check Bearing","type":"check"}},{{"id":"n3","label":"Bearing Worn?","type":"decision"}},{{"id":"n4","label":"Replace Bearing","type":"process"}},{{"id":"n5","label":"Check Shaft Alignment","type":"check"}},{{"id":"n6","label":"Misaligned?","type":"decision"}},{{"id":"n7","label":"Realign Shaft","type":"process"}},{{"id":"n8","label":"Check Mounting Bolts","type":"check"}},{{"id":"n9","label":"Bolts Loose?","type":"decision"}},{{"id":"n10","label":"Tighten Bolts","type":"process"}},{{"id":"n11","label":"Escalate Investigation","type":"end"}}],
"edges":[{{"from":"n1","to":"n2","label":""}},{{"from":"n3","to":"n4","label":"Yes"}},{{"from":"n3","to":"n5","label":"No"}},{{"from":"n5","to":"n6","label":""}},{{"from":"n6","to":"n7","label":"Yes"}},{{"from":"n6","to":"n8","label":"No"}},{{"from":"n8","to":"n9","label":""}},{{"from":"n9","to":"n10","label":"Yes"}},{{"from":"n9","to":"n11","label":"No"}},{{"from":"n2","to":"n3","label":""}}]}}

Output ONLY the raw JSON object now:"""


def build_diagram_from_context_prompt(query: str, last_ai_response: str) -> str:
    return f"""You are a technical diagram generator for an industrial maintenance system.
Output ONLY a valid JSON object — no explanation, no markdown.
PREVIOUS AI RESPONSE (content to visualise):
{last_ai_response}
USER REQUEST: {query}
Extract every distinct step from the previous response and generate a flowchart.
Always produce 6-12 nodes.
Output JSON: {{"title":"Short title","type":"flowchart","description":"One sentence","nodes":[{{"id":"n1","label":"Label","type":"start"}}],"edges":[{{"from":"n1","to":"n2","label":""}}]}}
Node types: start(green)|end(red)|process(blue)|decision(yellow)|check(purple)
Output ONLY raw JSON.
"""


def build_diagram_from_context_prompt(query: str, last_ai_response: str) -> str:
    return f"""You convert an existing maintenance answer into a flowchart. Output ONLY valid JSON.

PREVIOUS ANSWER (this is the EXACT content to turn into a flowchart):
\"\"\"
{last_ai_response}
\"\"\"

USER REQUEST: {query}

RULES:
- Use ONLY the steps/causes/actions written in the PREVIOUS ANSWER above. Do NOT invent generic steps.
- Read it top to bottom. Each numbered step, each fix action, each cause becomes ONE node, in the SAME order.
- If the answer lists a cause then fixes, start node = the problem/symptom, then cause node(s), then each fix as a process node, then an end node.
- If a step is a question or check (e.g. "is it resolved?", "check X"), make it type "decision" with Yes/No edges.
- Keep node labels SHORT (max 5 words) but faithful to the wording used.
- Produce between 5 and 12 nodes. Connect them in a single logical flow.

Output JSON ONLY (no markdown, no commentary):
{{"title":"<short title from the answer topic>","type":"flowchart","description":"One sentence",
"nodes":[{{"id":"n1","label":"Start / Symptom","type":"start"}},{{"id":"n2","label":"Step","type":"process"}}],
"edges":[{{"from":"n1","to":"n2","label":""}}]}}
Node types: start(green)|end(red)|process(blue)|decision(yellow)|check(purple)
Output ONLY raw JSON."""


def parse_diagram_json(raw: str) -> Optional[dict]:
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    m = re.search(r"\{[\s\S]+\}", cleaned)
    if m: cleaned = m.group(0)
    for attempt in [cleaned, re.sub(r",\s*([}\]])", r"\1", cleaned).replace("'", '"')]:
        try:
            data = json.loads(attempt)
            if "nodes" in data and "edges" in data and len(data["nodes"]) >= 2:
                return data
        except Exception:
            pass
    return None


def build_fallback_diagram(query: str) -> dict:
    q = query.lower()
    is_troubleshoot = any(kw in q for kw in ["troubleshoot","fix","fault","problem","issue","error","alarm","broken"])
    subject = _extract_subject_label(query)
    if is_troubleshoot:
        return {
            "title": "Troubleshooting Flow", "type": "flowchart",
            "description": f"Fault diagnosis workflow for {subject}",
            "nodes": [
                {"id":"n1","label":"Fault Observed","type":"start"},{"id":"n2","label":"Check Error Codes","type":"process"},
                {"id":"n3","label":"Search DB Records","type":"process"},{"id":"n4","label":"Record Found?","type":"decision"},
                {"id":"n5","label":"Apply Known Fix","type":"process"},{"id":"n6","label":"Diagnose Manually","type":"check"},
                {"id":"n7","label":"Issue Resolved?","type":"decision"},{"id":"n8","label":"Log & Close","type":"end"},
                {"id":"n9","label":"Escalate Issue","type":"end"},
            ],
            "edges": [
                {"from":"n1","to":"n2","label":""},{"from":"n2","to":"n3","label":""},{"from":"n3","to":"n4","label":""},
                {"from":"n4","to":"n5","label":"Yes"},{"from":"n4","to":"n6","label":"No"},{"from":"n5","to":"n7","label":""},
                {"from":"n6","to":"n7","label":""},{"from":"n7","to":"n8","label":"Yes"},{"from":"n7","to":"n9","label":"No"},
            ],
        }
    return {
        "title": f"{subject} Process Flow", "type": "flowchart",
        "description": f"Operational process for {subject}",
        "nodes": [
            {"id":"n1","label":"System Start","type":"start"},{"id":"n2","label":"Load / Setup","type":"process"},
            {"id":"n3","label":"Pre-run Check","type":"check"},{"id":"n4","label":"Ready to Run?","type":"decision"},
            {"id":"n5","label":"Execute Process","type":"process"},{"id":"n6","label":"Monitor / Verify","type":"check"},
            {"id":"n7","label":"Output OK?","type":"decision"},{"id":"n8","label":"Unload / Complete","type":"process"},
            {"id":"n9","label":"Log & End","type":"end"},
        ],
        "edges": [
            {"from":"n1","to":"n2","label":""},{"from":"n2","to":"n3","label":""},{"from":"n3","to":"n4","label":""},
            {"from":"n4","to":"n5","label":"Yes"},{"from":"n4","to":"n2","label":"Adjust"},{"from":"n5","to":"n6","label":""},
            {"from":"n6","to":"n7","label":""},{"from":"n7","to":"n8","label":"Pass"},{"from":"n7","to":"n5","label":"Retry"},
            {"from":"n8","to":"n9","label":""},
        ],
    }


def _extract_subject_label(query: str) -> str:
    q = re.sub(r"\b(show|draw|create|generate|make|give me|diagram|flowchart|flow chart|"
               r"flow diagram|visualize|visualise|workflow|process flow|of|for|the|a|an|please)\b",
               " ", query, flags=re.IGNORECASE)
    q = re.sub(r"\s+", " ", q).strip()
    return " ".join(q.split()[:4]).strip() or "equipment"



def _short_label(text: str, max_words: int = 6) -> str:
    words = re.sub(r"\s+", " ", str(text or "")).strip().split(" ")
    out = " ".join(w for w in words[:max_words] if w)
    return out if out else "—"


def build_explain_prompt(
    query: str,
    selected_text: str = "",
    last_ai_response: str = "",
    lang: str = "english",
) -> str:
    """
    Answer a follow-up / explain request using conversation context + the model's
    own engineering knowledge. Does NOT pull fresh DB records — it reasons over
    what's already on screen plus general SMD/maintenance expertise.
    """
    focus = f'\nThe technician highlighted this specific text and wants it explained:\n"""\n{selected_text}\n"""\n' if selected_text.strip() else ""
    ctx   = f'\nPREVIOUS ASSISTANT ANSWER (conversation context — use it to understand what is being asked):\n"""\n{last_ai_response}\n"""\n' if last_ai_response.strip() else ""
    return (
        f"""You are a senior SMD / industrial maintenance engineer. The technician is asking a FOLLOW-UP question about something already shown on screen.
{ctx}{focus}
TECHNICIAN'S FOLLOW-UP: {query}

HOW TO ANSWER:
- This is NOT a database lookup. Do NOT list maintenance records.
- Explain clearly using the context above PLUS your own engineering knowledge.
- If they highlighted a term/action (e.g. "replaced worn coupling and aligned drive"), explain what it means, why it's done, and what the technician should understand about it.
- Be practical and concise — a few short paragraphs or a short numbered list. No preamble.
"""
        + _lang_instruction(lang)
    )

def build_records_tree_diagram(query: str, records: list, fmt: str = "tree") -> dict:
    """Deterministically build a tree/graph diagram FROM the found records.
    Far more reliable than asking the 3B model to do it."""
    subject = _short_label(_extract_subject_label(query), 5) or "Records"
    nodes = [{"id": "root", "label": subject, "type": "start"}]
    edges = []
    for i, r in enumerate(records[:6], 1):
        prob = (r.get("problem") or "").strip()
        sol  = (r.get("solution") or "").strip()
        pid, sid = f"p{i}", f"s{i}"
        nodes.append({"id": pid, "label": _short_label(prob, 6) or f"Problem {i}", "type": "process"})
        edges.append({"from": "root", "to": pid, "label": ""})
        if sol:
            nodes.append({"id": sid, "label": _short_label(sol, 6), "type": "check"})
            edges.append({"from": pid, "to": sid, "label": "fix"})
    word = "Tree" if fmt == "tree" else "Graph"
    return {
        "title": f"{_short_label(subject, 4)} {word}",
        "type": "flowchart",
        "description": f"{word} view of matching maintenance records",
        "nodes": nodes,
        "edges": edges,
    }


def tsg_is_available() -> bool:
    tsg_col = get_tsg_collection()
    if tsg_col is None: return False
    try: return tsg_col.count() > 0
    except Exception: return False


# ═════════════════════════════════════════════════════════════════════════════
# ★★★ V2 PIPELINE — PLAN → EXECUTE → SYNTHESIZE ★★★
# One planner LLM call decides everything; deterministic code executes;
# one synthesis LLM call writes the answer in the requested format.
# ═════════════════════════════════════════════════════════════════════════════

_SQL_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|"
    r"vacuum|reindex|begin|commit|rollback)\b", re.IGNORECASE)

_ALLOWED_TABLES = re.compile(r"\b(mttr_records|tsg_records)\b", re.IGNORECASE)


def run_safe_select(sql: str, limit: int = PLANNER_MAX_ROWS) -> list[dict]:
    """Execute ONE read-only SELECT against the SQLite DB. Raises on anything unsafe."""
    s = sql.strip().rstrip(";")
    if ";" in s:
        raise ValueError("Multiple statements are not allowed.")
    if not re.match(r"^\s*select\b", s, re.IGNORECASE):
        raise ValueError("Only SELECT statements are allowed.")
    if _SQL_FORBIDDEN.search(s):
        raise ValueError("Forbidden SQL keyword.")
    if not _ALLOWED_TABLES.search(s):
        raise ValueError("Query must target mttr_records or tsg_records.")
    if not re.search(r"\blimit\b", s, re.IGNORECASE):
        s += f" LIMIT {limit}"
    conn = sqlite3.connect(f"file:{SQLITE_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(s).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _planner_value_hints() -> str:
    """Real distinct values from the DB so the planner spells filters correctly."""
    lines = []
    for col in ("machine", "model_name", "work_done_by", "shift", "spare_parts"):
        vals = _schema.column_values.get(col, [])[:15]
        if vals:
            lines.append(f"  {col}: {', '.join(repr(str(v)) for v in vals)}")
    # TSG machines too
    try:
        conn = get_sqlite_conn()
        tsg_m = [r[0] for r in conn.execute(
            "SELECT DISTINCT machine FROM tsg_records LIMIT 15").fetchall() if r[0]]
        conn.close()
        if tsg_m:
            lines.append(f"  tsg_records.machine: {', '.join(repr(v) for v in tsg_m)}")
    except Exception:
        pass
    return "\n".join(lines)


_PLANNER_PROMPT = """You are a query planner for an industrial maintenance assistant.
Today's date is {today}.

DATABASE: SQLite with two tables.

Table mttr_records (maintenance history):
  machine TEXT          -- machine type, e.g. 'Pick and Place', 'Reflow Oven'
  model_name TEXT       -- brand/model, e.g. 'Panasonic NPM', 'Heller 1809'
  line_no INTEGER       -- SMD line number (1..100)
  problem TEXT          -- free-text fault description
  solution TEXT         -- free-text action taken
  loss_time REAL        -- downtime in MINUTES; -1 means unknown (exclude with loss_time >= 0)
  work_done_by TEXT     -- technician name
  iso_date TEXT         -- 'YYYY-MM-DD' (ALWAYS use this for dates; never the 'date' column)
  shift TEXT            -- e.g. night / general / morning
  spare_parts TEXT      -- comma-separated parts used

Table tsg_records (official troubleshooting guide):
  line_no TEXT, machine TEXT, issue TEXT, cause TEXT, corrective TEXT

KNOWN VALUES (use these exact spellings in LIKE filters when they match the user's words):
{value_hints}

USER QUESTION: {query}

Decide how to answer. Output ONLY a JSON object, no markdown, no explanation:
{{
  "mode": "sql" | "semantic" | "hybrid" | "tsg_sql" | "tsg_semantic" | "general" | "chat",
  "sql": "<one read-only SELECT, or empty string>",
  "semantic_query": "<symptom/fault text for vector search, or empty>",
  "output_format": "default" | "table" | "list" | "summary" | "problems_only" | "solutions_only" | "flowchart",
  "wants_chart": false,
  "chart_sql": "<SELECT returning exactly (label, value) columns for the chart, or empty>",
  "chart_type": "bar" | "pie" | "line" | "hbar" | "",
  "answer_focus": "<one sentence: exactly what the user wants returned>"
}}

DECISION RULES:
- "sql": filtering/listing/counting/aggregating/ranking by machine, model, line, worker, date, shift, spare part. Use LIKE '%term%' with LOWER() for text, BETWEEN for line ranges, ORDER BY iso_date DESC LIMIT 1 for "last time / most recent", strftime or SUBSTR(iso_date,1,7) for months.
- "semantic": user describes a SYMPTOM with no structured filter ("motor overheating, how to fix").
- "hybrid": symptom PLUS a structured filter ("board jam on Panasonic NPM line 3") → write the sql for the filters AND put the symptom in semantic_query.
- "tsg_sql" / "tsg_semantic": user explicitly mentions the troubleshooting guide / TSG. Use tsg_sql when there's a clear machine/keyword filter, tsg_semantic for symptom descriptions.
- "general": definition / how-it-works question needing no database ("what is a wave soldering machine").
- "chat": greetings, thanks, smalltalk.
- wants_chart=true ONLY if user asks for chart/graph/pie/bar/trend/distribution. chart_sql MUST return exactly two columns aliased label and value, e.g. SELECT machine AS label, COUNT(*) AS value FROM mttr_records GROUP BY machine ORDER BY value DESC LIMIT 10.
- In LIKE filters, prefer the KNOWN VALUES spellings. For multi-word machine names match flexibly: machine LIKE '%pick%place%'.
- NEVER invent filters the user didn't state.

EXAMPLES:

Q: "give me all work done by Ram in January 2026"
{{"mode":"sql","sql":"SELECT iso_date, machine, line_no, problem, solution, shift, loss_time FROM mttr_records WHERE LOWER(work_done_by) LIKE '%ram%' AND SUBSTR(iso_date,1,7)='2026-01' ORDER BY iso_date","semantic_query":"","output_format":"default","wants_chart":false,"chart_sql":"","chart_type":"","answer_focus":"all jobs Ram performed in Jan 2026"}}

Q: "show all problems of pick and place machine"
{{"mode":"sql","sql":"SELECT problem FROM mttr_records WHERE LOWER(machine) LIKE '%pick%place%'","semantic_query":"","output_format":"problems_only","wants_chart":false,"chart_sql":"","chart_type":"","answer_focus":"only the problem descriptions"}}

Q: "who performed the hardening work on the AOI machine last time?"
{{"mode":"sql","sql":"SELECT work_done_by, iso_date, problem, solution FROM mttr_records WHERE LOWER(machine) LIKE '%aoi%' AND (LOWER(problem) LIKE '%harden%' OR LOWER(solution) LIKE '%harden%') ORDER BY iso_date DESC LIMIT 1","semantic_query":"","output_format":"default","wants_chart":false,"chart_sql":"","chart_type":"","answer_focus":"who did the most recent hardening job"}}

Q: "table of all issues of pick and place from line 1 to 7"
{{"mode":"sql","sql":"SELECT line_no, problem, solution, iso_date FROM mttr_records WHERE LOWER(machine) LIKE '%pick%place%' AND line_no BETWEEN 1 AND 7 ORDER BY line_no","semantic_query":"","output_format":"table","wants_chart":false,"chart_sql":"","chart_type":"","answer_focus":"tabular issue list lines 1-7"}}

Q: "pie chart of failures by shift in 2026"
{{"mode":"sql","sql":"","semantic_query":"","output_format":"default","wants_chart":true,"chart_sql":"SELECT shift AS label, COUNT(*) AS value FROM mttr_records WHERE SUBSTR(iso_date,1,4)='2026' AND shift != '' GROUP BY shift ORDER BY value DESC","chart_type":"pie","answer_focus":"shift-wise failure distribution for 2026"}}

Q: "conveyor jam from troubleshooting guide"
{{"mode":"tsg_sql","sql":"SELECT line_no, machine, issue, cause, corrective FROM tsg_records WHERE LOWER(issue) LIKE '%jam%' AND LOWER(machine) LIKE '%conveyor%'","semantic_query":"","output_format":"default","wants_chart":false,"chart_sql":"","chart_type":"","answer_focus":"TSG entries for conveyor jam"}}

Q: "reflow oven heating zone not reaching temperature, kaise fix karu"
{{"mode":"hybrid","sql":"SELECT * FROM mttr_records WHERE LOWER(machine) LIKE '%reflow%'","semantic_query":"heating zone not reaching temperature","output_format":"default","wants_chart":false,"chart_sql":"","chart_type":"","answer_focus":"fix steps for reflow heating fault, grounded in past records"}}

Now output the JSON for the USER QUESTION above:"""


def _parse_plan_json(raw: str) -> Optional[dict]:
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    m = re.search(r"\{[\s\S]+\}", cleaned)
    if not m:
        return None
    for attempt in (m.group(0), re.sub(r",\s*([}\]])", r"\1", m.group(0))):
        try:
            plan = json.loads(attempt)
            if isinstance(plan, dict) and "mode" in plan:
                plan.setdefault("sql", "")
                plan.setdefault("semantic_query", "")
                plan.setdefault("output_format", "default")
                plan.setdefault("wants_chart", False)
                plan.setdefault("chart_sql", "")
                plan.setdefault("chart_type", "")
                plan.setdefault("answer_focus", "")
                return plan
        except Exception:
            continue
    return None


async def make_plan(query: str) -> dict:
    """
    Call the planner model. Uses a richer schema-aware prompt so the model
    sees real column values and never needs hardcoded examples.
    """
    # Build live schema description with real values from DB
    schema_lines = []
    try:
        conn = get_sqlite_conn()
        # Main table
        schema_lines.append("Table: mttr_records")
        schema_lines.append("Columns (with real sample values from YOUR database):")
        col_info = conn.execute("PRAGMA table_info(mttr_records)").fetchall()
        _SKIP = {"id", "image_b64", "image_mime", "chroma_id", "image_name"}
        for col in col_info:
            cname = col[1]
            if cname in _SKIP:
                continue
            try:
                samples = conn.execute(
                    f"SELECT {cname}, COUNT(*) c FROM mttr_records "
                    f"WHERE {cname} IS NOT NULL AND CAST({cname} AS TEXT) != '' "
                    f"GROUP BY {cname} ORDER BY c DESC LIMIT 12"
                ).fetchall()
                sample_vals = [str(r[0])[:25] for r in samples if r[0] is not None]
                val_str = f"  →  values: {', '.join(repr(v) for v in sample_vals)}" if sample_vals else ""
                schema_lines.append(f"  {cname}{val_str}")
            except Exception:
                schema_lines.append(f"  {cname}")

        # Date range
        dr = conn.execute(
            "SELECT MIN(iso_date), MAX(iso_date) FROM mttr_records "
            "WHERE iso_date IS NOT NULL AND iso_date != ''"
        ).fetchone()
        if dr and dr[0]:
            schema_lines.append(f"\nDate range in database: {dr[0]} to {dr[1]}")

        # TSG table
        try:
            tsg_cols = conn.execute("PRAGMA table_info(tsg_records)").fetchall()
            tsg_machines = conn.execute(
                "SELECT DISTINCT machine FROM tsg_records LIMIT 15"
            ).fetchall()
            schema_lines.append("\nTable: tsg_records (Troubleshooting Guide)")
            schema_lines.append("Columns: " + ", ".join(c[1] for c in tsg_cols))
            if tsg_machines:
                schema_lines.append(
                    "Machines in TSG: " + ", ".join(repr(r[0]) for r in tsg_machines if r[0])
                )
        except Exception:
            pass

        conn.close()
    except Exception as e:
        schema_lines.append(f"(Schema read failed: {e})")

    schema_desc = "\n".join(schema_lines)

    prompt = f"""You are an expert SQL assistant for an industrial maintenance database.
Today: {date.today().isoformat()}

YOUR DATABASE SCHEMA (real values shown — use these exact spellings in filters):
{schema_desc}

SQLite RULES you must follow:
- Text: LOWER(col) LIKE LOWER('%term%')  — never exact equals for names
- Date: ALWAYS use iso_date column (format YYYY-MM-DD)
  - Year: SUBSTR(iso_date,1,4) = '2026'
  - Month: SUBSTR(iso_date,1,7) = '2026-01'
  - Most recent: ORDER BY iso_date DESC LIMIT 1
- loss_time is MINUTES, -1=unknown → exclude: loss_time >= 0
- Line range: line_no BETWEEN 1 AND 10
- Machine multi-word: LOWER(machine) LIKE '%pick%place%'
- Charts: SELECT col AS label, COUNT(*) AS value FROM ... GROUP BY col ORDER BY value DESC
- SELECT * for full records; SELECT specific cols for focused answers
- NO LIMIT if user wants "all" records — use LIMIT 100 as safety max
- Use tsg_records ONLY when user says "troubleshooting guide" or "TSG"

CRITICAL: The user wants COMPLETE data. Do NOT write SQL that returns less than what they asked for.
If user says "all problems" → no artificial LIMIT. If user says "table" → SELECT all relevant columns.

Output ONLY valid JSON, no markdown, no explanation:
{{
  "mode": "sql | semantic | hybrid | tsg | general | chat",
  "sql": "complete SELECT statement or empty string",
  "semantic_query": "symptom text for vector search or empty",
  "output_format": "default | table | problems_only | solutions_only | summary",
  "wants_chart": false,
  "chart_sql": "SELECT col AS label, COUNT(*) AS value ... or empty",
  "chart_type": "bar | pie | line | hbar | empty",
  "answer_focus": "one sentence: exactly what the user wants"
}}

MODE rules:
- sql: any structured filter (machine/model/worker/date/line/shift) → write SQL
- semantic: pure symptom/fault with no filter ("motor overheating how to fix")
- hybrid: filter + symptom together → sql for filter + semantic_query for symptom
- tsg: user says troubleshooting guide / TSG
- general: definition question, no DB ("what is a reflow oven")
- chat: greeting/smalltalk

USER QUESTION: {query}

JSON:"""

    _FALLBACK = {
        "mode": "semantic", "sql": "", "semantic_query": query,
        "output_format": "default", "wants_chart": False,
        "chart_sql": "", "chart_type": "", "answer_focus": query,
    }

    try:
        raw  = await ask_ollama(prompt, max_tokens=600, model=OLLAMA_MODEL_PLANNER)
        plan = _parse_plan_json(raw)
        if plan is None:
            print(f"[Planner] Could not parse JSON from: {raw[:300]}")
            return _FALLBACK
        print(
    f"[Planner] mode={plan['mode']} "
    f"fmt={plan['output_format']} "
    f"sql={plan['sql'][:120] if plan['sql'] else '(none)'}"
)
        return plan
    except Exception as e:
        print(f"[Planner] failed: {e}")
        return _FALLBACK

async def execute_sql_with_repair(sql: str) -> tuple[list[dict], str]:
    """Run planner SQL; on failure, give the error back to the planner model ONCE."""
    try:
        return run_safe_select(sql), sql
    except Exception as e:
        repair_prompt = (
            f"This SQLite query failed.\nQUERY: {sql}\nERROR: {e}\n"
            f"Tables: mttr_records(machine, model_name, line_no, problem, solution, "
            f"loss_time, work_done_by, iso_date, shift, spare_parts), "
            f"tsg_records(line_no, machine, issue, cause, corrective).\n"
            f"Return ONLY the corrected SELECT statement, nothing else."
        )
        try:
            fixed = await ask_ollama(repair_prompt, max_tokens=300, model=OLLAMA_MODEL_PLANNER)
            fixed = re.sub(r"```(?:sql)?|```", "", fixed).strip()
            m = re.search(r"select[\s\S]+", fixed, re.IGNORECASE)
            if m:
                fixed_sql = m.group(0).strip()
                return run_safe_select(fixed_sql), fixed_sql
        except Exception as e2:
            print(f"[Planner] SQL repair also failed: {e2}")
        return [], sql


def _rows_to_chart_viz(rows: list[dict], chart_type: str, title: str) -> Optional[dict]:
    """Deterministically convert (label, value) rows into your frontend viz JSON."""
    data = []
    for r in rows:
        label = r.get("label");  value = r.get("value")
        if label in (None, "") or value is None:
            # tolerate un-aliased columns: take first col as label, second as value
            vals = list(r.values())
            if len(vals) >= 2:
                label, value = vals[0], vals[1]
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        data.append({"label": _short_label(str(label), 6),
                     "value": int(value) if value == int(value) else round(value, 1)})
    if not data:
        return None
    total = sum(d["value"] for d in data) or 1
    for d in data:
        d["percent"] = round(d["value"] / total * 100, 1)
    rt = {"bar": "bar_chart", "pie": "pie_chart", "line": "line_chart",
          "hbar": "horizontal_bar_chart"}.get(chart_type, "bar_chart")
    return {"responseType": rt, "title": title or "Chart",
            "xAxis": "Category", "yAxis": "Value", "data": data[:12],
            "summary": ""}


_V2_FORMAT_HINTS = {
    "table":          "Present the answer as a MARKDOWN TABLE (header row, separator row, data rows). After the table add one short Summary line.",
    "list":           "Present the answer as a concise numbered list.",
    "summary":        "Write a SHORT 3-5 sentence summary. No lists, no tables.",
    "problems_only":  "Output ONLY the problem descriptions as a numbered list. Nothing else — no solutions, no dates, no extra fields, no preamble.",
    "solutions_only": "Output ONLY the solution/action descriptions as a numbered list. Nothing else.",
    "default":        "Choose the clearest format yourself: a direct sentence for a single-fact answer, a short list for multiple records, a table only if the user implied one.",
}

def build_v2_answer_prompt(query: str, plan: dict, rows: list[dict],
                           lang: str, source_label: str) -> str:
    fmt_hint = _V2_FORMAT_HINTS.get(plan.get("output_format", "default"),
                                    _V2_FORMAT_HINTS["default"])

    # Give the LLM ALL rows, not truncated — this is why Claude gives complete answers
    # Strip only binary/system fields
    clean_rows = []
    for r in rows:
        clean = {}
        for k, v in r.items():
            if k in ("image_b64", "image_mime", "chroma_id", "image_name", "id"):
                continue
            if v in (None, -1, -1.0):
                continue
            if isinstance(v, str) and v.strip() == "":
                continue
            clean[k] = v
        if clean:
            clean_rows.append(clean)

    # Use the human-readable record formatter (not raw JSON) so the LLM
    # can read it the same way a technician would read a report
    is_tsg = any("issue" in r or "corrective" in r for r in clean_rows[:3])

    if not clean_rows:
        records_section = "NO RECORDS FOUND."
    elif is_tsg:
        records_section = format_tsg_records_for_prompt(clean_rows)
    else:
        records_section = format_records_for_prompt(clean_rows)

    # For table format: give explicit column instructions
    if plan.get("output_format") == "table":
        fmt_instruction = (
            "Present ALL records as a MARKDOWN TABLE.\n"
            "- Include a proper header row with column names\n"
            "- One row per record — do NOT skip any record\n"
            "- Choose columns relevant to the question (e.g. Problem, Solution, Date, Machine, Loss Time)\n"
            "- Use '—' for missing values\n"
            "- After the table write: 'Total: N records found.'"
        )
    elif plan.get("output_format") == "problems_only":
        fmt_instruction = (
            "List ONLY the problem/fault descriptions, numbered.\n"
            "One line per problem. No solutions, no dates, no other fields.\n"
            "Do NOT skip any problem. List ALL of them."
        )
    elif plan.get("output_format") == "solutions_only":
        fmt_instruction = (
            "List ONLY the solution/action descriptions, numbered.\n"
            "One line per solution. No other fields. List ALL of them."
        )
    elif plan.get("output_format") == "summary":
        fmt_instruction = "Write a clear 3-5 sentence summary of the key findings."
    else:
        fmt_instruction = (
            "Answer directly in the most natural format:\n"
            "- Single fact → one sentence\n"
            "- Multiple records → numbered list or table depending on what fits best\n"
            "- Include all relevant fields the question asks about\n"
            "- Do NOT skip records or truncate the list"
        )

    return (
        f"You are a senior SMD/industrial maintenance engineer answering a technician.\n\n"
        f"QUESTION: {query}\n"
        f"WHAT THEY WANT: {plan.get('answer_focus') or query}\n"
        f"DATA SOURCE: {source_label}\n"
        f"TOTAL RECORDS RETRIEVED: {len(clean_rows)}\n\n"
        f"━━━ DATA ━━━\n"
        f"{records_section}\n"
        f"━━━ END OF DATA ━━━\n\n"
        f"FORMAT INSTRUCTION:\n{fmt_instruction}\n\n"
        f"STRICT RULES:\n"
        f"- Use ONLY the data above. NEVER invent names, dates, or numbers.\n"
        f"- Include ALL {len(clean_rows)} records — do not skip or truncate.\n"
        f"- loss_time values are in MINUTES.\n"
        f"- If no data: say 'No matching records found.' then add "
        f"2-3 lines of engineering guidance labelled 'From engineering knowledge:'.\n"
        f"- No preamble. Start directly with the answer.\n"
        + _lang_instruction(lang)
    )




# ─────────────────────────────────────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    machine_filter: Optional[str] = None
    model_filter: Optional[str] = None
    last_ai_response: Optional[str] = None
    selected_text: Optional[str] = None
    offset: int = 0
    tsg_followup_query: Optional[str] = None
    output_format: Optional[str] = None
    date_filter: Optional[dict] = None


class QueryResponse(BaseModel):
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


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    history: list[ChatMessage]


class ChatResponse(BaseModel):
    response: str


class DiagramRequest(BaseModel):
    query: str
    context: Optional[str] = ""


class DiagramResponse(BaseModel):
    title: str
    type: str
    nodes: list
    edges: list
    description: str


class TranslateRequest(BaseModel):
    text: str
    language: str


class TranslateResponse(BaseModel):
    translated: str


class OcrRequest(BaseModel):
    image_base64: str
    filename: str = "opl_image.jpg"
    user_prompt: str = "Extract and format the text from this OPL image"


class OcrResponse(BaseModel):
    raw_text: str
    formatted_text: str
    confidence: float
    language_detected: str


# ─────────────────────────────────────────────────────────────────────────────
# /diagram ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/diagram", response_model=DiagramResponse)
async def generate_diagram(req: DiagramRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    prompt = build_diagram_prompt(req.query, req.context or "")
    try:
        raw  = await ask_ollama(prompt, max_tokens=700)
        data = parse_diagram_json(raw)
    except Exception:
        data = None
    if data is None:
        data = build_fallback_diagram(req.query)
    return DiagramResponse(
        title=data.get("title", "Process Diagram"), type=data.get("type", "flowchart"),
        nodes=data.get("nodes", []), edges=data.get("edges", []), description=data.get("description", ""),
    )


# ─────────────────────────────────────────────────────────────────────────────
# ★★★ /query ENDPOINT — SCHEMA-DRIVEN PIPELINE ★★★
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
async def query_records(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    query  = req.query.strip()
    intent = detect_intent(query)
    lang = detect_language(query)

    corrected_query     = normalize_query(query)
    corrected_query_out = corrected_query if corrected_query.lower() != query.lower() else None
    output_format       = req.output_format or detect_output_format(query)

    base_response = dict(
        intent=intent, db_records_used=0, db_records_summary=[], diagram_data=None,
        corrected_query=corrected_query_out, total_records_found=0, has_more=False,
        current_offset=0, suggest_tsg=False, tsg_records_used=0, tsg_records_summary=[],
        output_format=output_format if output_format != "default" else None,
        date_filter_label=None, date_filtered_count=None, parsed_filters=None, execution_path=None,
    )

    if intent == "conversational":
        ai_response = await ask_ollama(build_conversational_prompt(query, lang=lang), max_tokens=120)
        return QueryResponse(ai_suggestion=ai_response, **base_response)
    
     # ── ★ SELECTED-TEXT / EXPLAIN FOLLOW-UP ─────────────────────────────────────
    # "Ask MTTR" on a highlighted snippet, or any explain/clarify question.
    # Answer from conversation context + engineering knowledge — NOT a DB re-query.
    _is_explain = bool(_EXPLAIN_FOLLOWUP.search(query)) and (
        bool(req.last_ai_response and req.last_ai_response.strip())
        or bool(req.selected_text and req.selected_text.strip())
    )
    if _is_explain:
        # Recover the highlighted snippet from the "Regarding '...':" prefix if the
        # frontend didn't send it explicitly.
        selected = (req.selected_text or "").strip()
        if not selected:
            m = re.search(r"regarding\s+[\"'](.+?)[\"']\s*:", query, re.IGNORECASE)
            if m:
                selected = m.group(1).strip()

        ai_response = await ask_ollama(
            build_explain_prompt(
                query=query,
                selected_text=selected,
                last_ai_response=req.last_ai_response or "",
                lang=lang,
            ),
            max_tokens=MAX_TOKENS,
        )
        return QueryResponse(
            ai_suggestion=ai_response,
            **{**base_response, "intent": "explain", "execution_path": "explain"},
        )
    
    # ── ★ DATA-CHART ANALYTICS (explicit chart/graph request) ──────────────────
    chart_spec = _detect_chart_spec(query)
    if chart_spec:
        # Strip chart vocabulary so filter extraction sees only machine/model/etc.
        clean_q = re.sub(
            r"\b(bar|pie|line|area|stacked|horizontal|scatter|heat\s*map|heatmap)?\s*"
            r"(chart|graph|plot|visuali[sz]ation|visuali[sz]e)\b", " ",
            corrected_query, flags=re.IGNORECASE)
        clean_q = re.sub(r"\b(distribution|breakdown|percentages?|percent|trends?)\b", " ",
                         clean_q, flags=re.IGNORECASE)
        clean_q = _strip_format_phrases(clean_q)

        filters = await extract_filters_dynamic(clean_q)
        if req.machine_filter: filters["machine"]    = req.machine_filter
        if req.model_filter:   filters["model_name"] = req.model_filter
        filters = _scan_known_values(clean_q, filters)
        filters = _reconcile_filter_columns(filters)

        viz, summary = build_analytics_visualization(filters, query, chart_spec)
        if lang == "hinglish" and summary:
            summary = await _hinglish_analytics(summary)
            if viz:
                viz["summary"] = summary

        public_filters = {k: v for k, v in filters.items() if not k.startswith("_") and v}
        print(f"[Chart] type={chart_spec} filters={public_filters} "
              f"-> {'viz' if viz else 'no-data'}")

        return QueryResponse(
            ai_suggestion=summary, intent="analytics",
            db_records_used=0, db_records_summary=[],
            diagram_data=None, visualization=viz,
            retrieval_confidence=None, corrected_query=corrected_query_out,
            detected_model=filters.get("model_name"),
            total_records_found=0, has_more=False, current_offset=0,
            suggest_tsg=False, tsg_records_used=0, tsg_records_summary=[],
            output_format=None, date_filter_label=None, date_filtered_count=None,
            parsed_filters=public_filters or None, execution_path="analytics",
        )


    if req.tsg_followup_query and TSG_YES_FOLLOWUP.match(query):
        query  = req.tsg_followup_query
        intent = "tsg_lookup"
        base_response["intent"] = intent

    if intent == "tsg_lookup":
        if not tsg_is_available():
            return QueryResponse(
                ai_suggestion="The Troubleshooting Guide has not been indexed yet. Run `python clean_tsg.py --file <your_tsg_file.xlsx>` and restart.",
                **base_response,
            )
        machine_vals = _schema.column_values.get("machine", get_all_machines())
        detected_machine = req.machine_filter or fuzzy_match_value(
            re.sub(r"\b(troubleshooting|guide|tsg|from|the)\b", "", query, flags=re.IGNORECASE).strip(),
            machine_vals, threshold=1
        )
        tsg_records = await tsg_retrieve(query, machine_filter=detected_machine, use_multi_query=False)
        tsg_summary = [
            {"line_no": m.get("line_no",""), "machine": m.get("machine",""),
             "issue": m.get("issue",""), "cause": m.get("cause",""), "corrective": m.get("corrective","")}
            for m in tsg_records[:PAGE_SIZE]
        ]
        ai_response = await ask_ollama(build_tsg_prompt(query, tsg_records[:PAGE_SIZE], detected_machine, lang=lang), max_tokens=MAX_TOKENS)
        return QueryResponse(
            ai_suggestion=ai_response, intent="tsg_lookup",
            db_records_used=0, db_records_summary=[], total_records_found=len(tsg_records),
            has_more=False, current_offset=0, suggest_tsg=False,
            tsg_records_used=len(tsg_summary), tsg_records_summary=tsg_summary, execution_path="tsg",
            **{k: v for k, v in base_response.items()
               if k not in ("intent","db_records_used","db_records_summary","total_records_found",
                            "has_more","current_offset","suggest_tsg","tsg_records_used",
                            "tsg_records_summary","execution_path")},
        )

    if intent == "diagram_context":
        last_response = (req.last_ai_response or "").strip()
        prompt = build_diagram_from_context_prompt(query, last_response) if last_response else build_diagram_prompt(query)
        try:
            raw = await ask_ollama(prompt, max_tokens=900)
            diagram_data = parse_diagram_json(raw)
        except Exception:
            diagram_data = None
        if diagram_data is None:
            diagram_data = build_fallback_diagram(query)
        return QueryResponse(
            ai_suggestion=f"Here is the flowchart — **{diagram_data.get('title','Process Diagram')}**.",
            intent="diagram",
            diagram_data=diagram_data,
            **{k: v for k, v in base_response.items() if k not in ("intent", "diagram_data")},
        )

    if intent == "diagram":
        ai_response = await ask_ollama(build_general_prompt(query))
        try:
            raw = await ask_ollama(build_diagram_prompt(query, ai_response), max_tokens=900)
            diagram_data = parse_diagram_json(raw)
        except Exception:
            diagram_data = None
        if diagram_data is None:
            diagram_data = build_fallback_diagram(query)
        return QueryResponse(
            ai_suggestion=ai_response,
            diagram_data=diagram_data,
            **{k: v for k, v in base_response.items() if k != "diagram_data"},
        )

    if intent == "general":
        ai_response = await ask_ollama(build_general_prompt(query, lang = lang))
        return QueryResponse(ai_suggestion=ai_response, **base_response)

    # ════════════════════════════════════════════════════════════════════════
    # ★★★ DB LOOKUP / TROUBLESHOOT — SCHEMA-DRIVEN PIPELINE ★★★
    # ════════════════════════════════════════════════════════════════════════

    # filters = await extract_filters_dynamic(query)

    # if req.machine_filter:
    #     filters["machine"]    = req.machine_filter
    # if req.model_filter:
    #     filters["model_name"] = req.model_filter


    # filters = _scan_known_values(query, filters)
    # filters = _reconcile_filter_columns(filters)

    # Strip presentation words ("in a tabular form", "as a tree", …) so they
    # never contaminate filter extraction or semantic search.
    search_query = _strip_format_phrases(corrected_query)

    filters = await extract_filters_dynamic(search_query)

    if req.machine_filter:
        filters["machine"]    = req.machine_filter
    if req.model_filter:
        filters["model_name"] = req.model_filter

    # filters = _scan_known_values(search_query, filters)
    # filters = _reconcile_filter_columns(filters)

   
    # semantic_q_raw     = filters.pop("_semantic_query", "")
    # has_semantic_query = bool(filters.get("_has_fault_keyword")) or bool(semantic_q_raw.strip())
    # # semantic_q         = semantic_q_raw or corrected_query
    # semantic_q         = semantic_q_raw or search_query
    # if lang == "hinglish":                       # ★ NEW
    #     semantic_q = _strip_hinglish_filler(semantic_q)
    filters = _scan_known_values(search_query, filters)
    filters = _reconcile_filter_columns(filters)

    # # ── ★ TROUBLESHOOT / FAULT PARITY FIX ────────────────────────────────────
    # # A fault query (English OR Hinglish) must be driven by the fault TEXT via
    # # semantic search — exactly like the working English path. On Hinglish input
    # # the small LLM sometimes drops the fault words into a WEAK structured column
    # # (e.g. spare_parts), which then becomes a strict SQL gate and returns ZERO
    # # rows. After reconcile, any surviving machine/model/worker is genuine, so if
    # # NO strong entity is present we strip the weak guesses to stay semantic_only.
    # _STRONG_ENTITY_COLS = ("machine", "model_name", "work_done_by")
    # _is_troubleshoot = (intent == "troubleshoot") or \
    #                    (filters.get("_response_type") == "troubleshoot") or \
    #                    bool(filters.get("_has_fault_keyword"))
    # if _is_troubleshoot:
    #     has_strong_entity = (
    #         any(filters.get(c) for c in _STRONG_ENTITY_COLS)
    #         or any(filters.get(c) for c in ("line_no", "year", "date_from", "date_to"))
    #     )
    #     if not has_strong_entity:
    #         # Drop every weak / fault-derived structured guess (spare_parts, and any
    #         # other non-entity column the LLM may have set) so the planner picks
    #         # semantic_only instead of an empty SQL gate. Shift is kept.
    #         for c in list(filters.keys()):
    #             if c.startswith("_") or c in _SPECIAL_FILTER_COLS or c == "shift":
    #                 continue
    #             if c not in _STRONG_ENTITY_COLS and c in _schema.columns:
    #                 filters.pop(c, None)
    _STRONG_ENTITY_COLS = ("machine", "model_name", "work_done_by")
    _is_troubleshoot = (intent == "troubleshoot") or \
                       (filters.get("_response_type") == "troubleshoot") or \
                       bool(filters.get("_has_fault_keyword"))

    if _is_troubleshoot:
        # A genuine entity is one the user TYPED verbatim — not a fault word like
        # "motor"/"belt" that also happens to be a spare_parts value.
        q_low = search_query.lower()
        genuine_entity = False
        for c in _STRONG_ENTITY_COLS:
            v = filters.get(c)
            if not v:
                continue
            # keep it only if the value is a multi-char token actually in the query
            # AND it isn't a generic fault/part word
            if str(v).lower() in q_low and str(v).lower() not in _FAULT_WORDS:
                genuine_entity = True
            else:
                filters.pop(c, None)          # drop spurious entity (e.g. "motor")

        has_hard_filter = genuine_entity or any(
            filters.get(c) for c in ("line_no", "year", "date_from", "date_to")
        )
        if not has_hard_filter:
            # Pure symptom → strip EVERY structured guess so the planner goes
            # semantic_only. Keep only meta keys and shift.
            for c in list(filters.keys()):
                if c.startswith("_") or c in _SPECIAL_FILTER_COLS or c == "shift":
                    continue
                filters.pop(c, None)

    semantic_q_raw     = filters.pop("_semantic_query", "")
    has_semantic_query = bool(filters.get("_has_fault_keyword")) or bool(semantic_q_raw.strip())
    semantic_q         = semantic_q_raw or search_query
    if lang == "hinglish":
        semantic_q = _strip_hinglish_filler(semantic_q)

    # ★ For fault/troubleshoot queries, build the semantic query from the
    #   technical CORE of the cleaned query, so retrieval never depends on the
    #   flaky LLM semantic_query field (the main English↔Hinglish divergence).
    if _is_troubleshoot:
        core = " ".join(extract_keywords(_strip_hinglish_filler(search_query)))
        if len(core) >= 3:
            semantic_q = core

    detected_model_out = filters.get("model_name")

    _sqlite_ready  = sqlite_is_ready()
    execution_path = plan_execution(filters, has_semantic_query, _sqlite_ready)

    print(f"[Pipeline v7] intent={intent} path={execution_path} "
          f"filters={{{', '.join(f'{k}={v!r}' for k, v in filters.items() if not k.startswith('_'))}}}"
          f" semantic={semantic_q!r}")

    if execution_path == "analytics":
        ai_response  = await handle_analytics_query(filters, query, lang=lang)
        public_filters = {k: v for k, v in filters.items() if not k.startswith("_") and v}
        return QueryResponse(
            ai_suggestion=ai_response, intent=intent,
            db_records_used=0, db_records_summary=[], total_records_found=0,
            has_more=False, current_offset=0, suggest_tsg=False,
            tsg_records_used=0, tsg_records_summary=[],
            parsed_filters=public_filters or None, execution_path=execution_path,
            **{k: v for k, v in base_response.items()
               if k not in ("intent","db_records_used","db_records_summary","total_records_found",
                            "has_more","current_offset","suggest_tsg","tsg_records_used",
                            "tsg_records_summary","parsed_filters","execution_path")},
        )

    # all_records:  list[dict] = []
    # total_found:  int        = 0
    # confidence:   float      = 0.0
    # page_records: list[dict] = []
    # has_more:     bool       = False
    # offset = max(0, req.offset)
    all_records:  list[dict] = []
    total_found:  int        = 0
    confidence:   float      = 0.0
    page_records: list[dict] = []
    has_more:     bool       = False
    diagram_data_out: Optional[dict] = None
    offset = max(0, req.offset)

    if execution_path == "sql_only" and _sqlite_ready:
        total_found  = sql_count(filters)
        page_records = sql_fetch(filters, offset=offset, limit=PAGE_SIZE)
        has_more     = (offset + len(page_records)) < total_found
        confidence   = 1.0

    elif execution_path == "hybrid" and _sqlite_ready:
        # 1. Hard gate — every record that satisfies ALL structured filters
        candidate_ids = fetch_candidate_ids(filters)
        if not candidate_ids:
            all_records = []                       # filter matched nothing → strictly empty
            confidence  = 1.0
        else:
            candidate_set = set(candidate_ids)
            sem_records, confidence = await semantic_search(
                semantic_q,
                machine_filter=filters.get("machine"),
                model_filter=filters.get("model_name"),
                use_multi_query=False,
            )
            # 2. Keep ONLY gated records, in semantic-relevance order
            ranked     = [r for r in sem_records if r.get("chroma_id", "") in candidate_set]
            ranked_set = {r.get("chroma_id", "") for r in ranked}
            # 3. Append gated records the semantic pass didn't surface (so nothing is lost)
            leftover = [cid for cid in candidate_ids if cid not in ranked_set]
            if leftover:
                ranked += sort_by_loss_time(fetch_rows_by_chroma_ids(leftover))
            all_records = ranked
        total_found  = len(all_records)
        page_records = all_records[offset: offset + PAGE_SIZE]
        has_more     = (offset + PAGE_SIZE) < total_found

    elif execution_path == "semantic_filter":
        sem_records, confidence = await semantic_search(
            semantic_q or query,
            machine_filter=filters.get("machine"),
            model_filter=filters.get("model_name"),
            use_multi_query=False,
        )
        for col, val in list(filters.items()):
            if col.startswith("_") or col in _SPECIAL_FILTER_COLS or not val:      continue
            if not _schema._ready or col not in _schema.columns:                    continue
            mt = _schema.columns[col].get("match_type", "partial")
            if mt == "exact":
                sem_records = [r for r in sem_records
                               if str(r.get(col, "")).strip().lower() == str(val).strip().lower()]
            elif mt in ("partial", "text"):
                sem_records = [r for r in sem_records
                               if str(val).lower() in str(r.get(col, "")).lower()]
        if filters.get("line_no") not in (None, ""):
            try:
                ln = int(filters["line_no"])
                sem_records = [r for r in sem_records if int(r.get("line_no", -1) or -1) == ln]
            except (ValueError, TypeError):
                pass
        if filters.get("year"):
            yr = str(filters["year"])
            sem_records = [r for r in sem_records if str(r.get("iso_date", "")).startswith(yr)]
        if filters.get("date_from"):
            df = str(filters["date_from"])
            sem_records = [r for r in sem_records if r.get("iso_date", "") and r["iso_date"] >= df]
        if filters.get("date_to"):
            dt = str(filters["date_to"])
            sem_records = [r for r in sem_records if r.get("iso_date", "") and r["iso_date"] <= dt]
        all_records  = sem_records
        total_found  = len(all_records)
        page_records = all_records[offset: offset + PAGE_SIZE]
        has_more     = (offset + PAGE_SIZE) < total_found

    else:
        sem_records, confidence = await semantic_search(
            semantic_q or corrected_query,
            machine_filter=filters.get("machine"),
            model_filter=filters.get("model_name"),
            use_multi_query=False,
        )
        if not sem_records and lang == "hinglish":
            core = " ".join(extract_keywords(semantic_q or corrected_query))
            if len(core) >= 3:
                sem_records, confidence = await semantic_search(
                    core,
                    machine_filter=filters.get("machine"),
                    model_filter=filters.get("model_name"),
                    use_multi_query=False,
                )
        all_records  = sem_records
        total_found  = len(all_records)
        page_records = all_records[offset: offset + PAGE_SIZE]
        has_more     = (offset + PAGE_SIZE) < total_found

    records_summary = []
    for m in page_records:
        rec = {
            "machine":      m.get("machine", ""),
            "model_name":   m.get("model_name", ""),
            "problem":      m.get("problem", ""),
            "solution":     m.get("solution", ""),
            "loss_time":    _loss_time_label(m),
            "work_done_by": m.get("work_done_by", ""),
            "date":         m.get("date", ""),
            "shift":        m.get("shift", ""),
            "spare_parts":  m.get("spare_parts", ""),
            "image_b64":    m.get("image_b64", ""),
            "image_name":   m.get("image_name", ""),
            "image_mime":   m.get("image_mime", ""),
        }
        known = set(rec.keys()) | _SYSTEM_COLS
        for k, v in m.items():
            if k not in known: rec[k] = v
        records_summary.append(rec)
        
    date_label  = ""
    date_filter = parse_date_filter(query)
    if date_filter: date_label = date_filter.get("label", "")

    if req.offset > 0:
        end_idx     = offset + len(page_records)
        showing_str = f"{offset + 1}–{end_idx} of {total_found}"
        more_str    = "More records available — click Give more to continue." if has_more else "All matching records have now been shown."
        ai_response = f"Showing records {showing_str}. {more_str}"

    elif not page_records:
        worker  = filters.get("work_done_by")
        machine = filters.get("machine")
        year    = filters.get("year")
        shift   = filters.get("shift")
        parts   = []
        if worker:  parts.append(f"work done by **{worker}**")
        if machine: parts.append(f"machine **{machine}**")
        if year:    parts.append(f"year **{year}**")
        if shift:   parts.append(f"**{shift}** shift")
        for k, v in filters.items():
            if k.startswith("_") or k in ("machine","work_done_by","year","shift") or not v: continue
            if k in _schema.columns:
                parts.append(f"**{k.replace('_',' ')}**: {v}")
        scope       = ", ".join(parts) if parts else "the given filters"
        ai_response = (f"No maintenance records found in the database for {scope}. "
                       f"Please check the spelling or try broader search terms.")
        

    else:
        r_type = filters.get("_response_type", "list")
        if output_format in ("tree", "graph"):
            diagram_data_out = build_records_tree_diagram(search_query or query, page_records, output_format)
            n = len(page_records)
            ai_response = (f"Here is a {output_format} view of the "
                           f"{n} matching record{'s' if n != 1 else ''}. "
                           f"Each branch shows a problem and its solution.")
        elif output_format == "table" or r_type == "table":
            ai_response = await ask_ollama(
                build_table_prompt(query, page_records, date_label, lang=lang), max_tokens=900)
        elif output_format in ("bullets", "numbered"):
            ai_response = await ask_ollama(
                build_structured_list_prompt(query, page_records, output_format, date_label, lang=lang),
                max_tokens=MAX_TOKENS)
        elif output_format == "summary" or r_type == "summary":
            ai_response = await ask_ollama(
                build_summary_prompt(query, page_records, date_label, lang=lang), max_tokens=400)
        elif r_type == "troubleshoot" or intent == "troubleshoot":
            ai_response = await ask_ollama(
                build_troubleshoot_prompt(
                    query, page_records,
                    filters.get("machine"), filters.get("model_name"),
                    lang=lang,
                ), max_tokens=MAX_TOKENS)
        else:
            ai_response = await ask_ollama(
                build_field_aware_prompt(
                    query=query, records=page_records,
                    filters=filters, date_label=date_label, lang=lang,
                ), max_tokens=MAX_TOKENS)
        # if output_format in ("tree", "graph"):
        #     diagram_data_out = build_records_tree_diagram(search_query or query, page_records, output_format)
        #     n = len(page_records)
        #     ai_response = (f"Here is a {output_format} view of the "
        #                    f"{n} matching record{'s' if n != 1 else ''}. "
        #                    f"Each branch shows a problem and its solution.")
        # elif output_format == "table" or r_type == "table":
        #     prompt = build_table_prompt(query, page_records, date_label)
        #     ai_response = await ask_ollama(prompt, max_tokens=900)
        # elif output_format in ("bullets", "numbered"):
        #     prompt = build_structured_list_prompt(query, page_records, output_format, date_label)
        #     ai_response = await ask_ollama(prompt, max_tokens=MAX_TOKENS)
        # elif output_format == "summary" or r_type == "summary":
        #     prompt = build_summary_prompt(query, page_records, date_label)
        #     ai_response = await ask_ollama(prompt, max_tokens=400)
        # elif r_type == "troubleshoot" or intent == "troubleshoot":
        #     prompt = build_troubleshoot_prompt(query, page_records, filters.get("machine"), filters.get("model_name"))
        #     ai_response = await ask_ollama(prompt, max_tokens=MAX_TOKENS)
        # else:
        #     prompt = build_field_aware_prompt(query=query, records=page_records, filters=filters, date_label=date_label)
        #     ai_response = await ask_ollama(prompt, max_tokens=MAX_TOKENS)

    # else:
    #     r_type = filters.get("_response_type", "list")
    #     if output_format == "table" or r_type == "table":
    #         prompt = build_table_prompt(query, page_records, date_label)
    #         ai_response = await ask_ollama(prompt, max_tokens=700)
    #     elif output_format == "summary" or r_type == "summary":
    #         prompt = build_summary_prompt(query, page_records, date_label)
    #         ai_response = await ask_ollama(prompt, max_tokens=400)
    #     elif r_type == "troubleshoot" or intent == "troubleshoot":
    #         prompt = build_troubleshoot_prompt(query, page_records, filters.get("machine"), filters.get("model_name"))
    #         ai_response = await ask_ollama(prompt, max_tokens=MAX_TOKENS)
    #     else:
    #         prompt = build_field_aware_prompt(query=query, records=page_records, filters=filters, date_label=date_label)
    #         ai_response = await ask_ollama(prompt, max_tokens=MAX_TOKENS)

    _is_structured = execution_path in ("sql_only", "analytics")
    suggest_tsg = (
        req.offset == 0 and tsg_is_available()
        and intent in ("db_lookup", "troubleshoot")
        and not _is_structured
        and not filters.get("work_done_by")
    )

    public_filters = {k: v for k, v in filters.items() if not k.startswith("_") and v}

    return QueryResponse(
        ai_suggestion=ai_response, intent=intent,
        db_records_used=len(page_records), db_records_summary=records_summary,
        diagram_data=None,
        retrieval_confidence=round(float(confidence), 4),
        # db_records_used=len(page_records), db_records_summary=records_summary,
        # diagram_data=None,
        # retrieval_confidence=round(float(confidence), 4),
        corrected_query=corrected_query_out, detected_model=detected_model_out,
        total_records_found=total_found, has_more=has_more, current_offset=offset,
        suggest_tsg=suggest_tsg, tsg_records_used=0, tsg_records_summary=[],
        output_format=output_format if output_format != "default" else None,
        date_filter_label=date_label or None, date_filtered_count=None,
        parsed_filters=public_filters or None, execution_path=execution_path,
    )

def _infer_analytics_type(sql: str, rows: list[dict]) -> str:
    """Detect what kind of aggregate result we have."""
    sql_lower = sql.lower()
    if not rows:
        return "empty"
    first = rows[0]
    keys  = list(first.keys())
    if len(rows) == 1 and len(keys) == 1:
        k = str(keys[0]).lower()
        if "cnt" in k or "count" in k:   return "count"
        if "total" in k or "sum" in k:   return "sum"
    if "group by" in sql_lower:
        if "shift"        in sql_lower:  return "by_shift"
        if "machine"      in sql_lower:  return "by_machine"
        if "work_done_by" in sql_lower:  return "by_worker"
        if "problem"      in sql_lower:  return "by_problem"
        if "model_name"   in sql_lower:  return "by_model"
        if "line_no"      in sql_lower:  return "by_line"
        return "grouped"
    return "records"


def _format_sql_aggregate(rows: list[dict], result_type: str,
                           query: str, plan: dict) -> str:
    """Format aggregate SQL results into clean text without calling the LLM."""
    if not rows:
        return f"No matching records found for: {query}"
    first = rows[0]
    focus = plan.get("answer_focus", "")

    if result_type == "count":
        cnt = list(first.values())[0]
        return f"Found **{cnt} matching record{'s' if cnt != 1 else ''}**. {focus}"

    if result_type == "sum":
        total = list(first.values())[0]
        if total is None or float(total) <= 0:
            return "No downtime data found."
        total = float(total)
        return (f"Total downtime: **{total:.0f} minutes**."
                if total < 60 else
                f"Total downtime: **{total/60:.1f} hours** ({total:.0f} minutes).")

    # Grouped results
    keys      = list(first.keys())
    name_col  = keys[0]
    count_col = keys[1] if len(keys) > 1 else None
    avg_col   = keys[2] if len(keys) > 2 else None

    titles = {
        "by_shift":   "Failures by shift",
        "by_machine": "Failures by machine",
        "by_worker":  "Work done by person",
        "by_problem": "Most common problems",
        "by_model":   "Failures by model",
        "by_line":    "Failures by line",
        "grouped":    "Results",
    }
    title = titles.get(result_type, "Results")
    lines = [f"**{title}**:\n"]
    for i, row in enumerate(rows[:15], 1):
        name  = str(row.get(name_col, "Unknown") or "Unknown")
        count = row.get(count_col, "") if count_col else ""
        avg   = row.get(avg_col)       if avg_col  else None
        avg_s = (f", avg downtime: {float(avg):.0f} min"
                 if avg and isinstance(avg, (int, float)) and float(avg) > 0 else "")
        count_s = f"**{count}**" if count != "" else ""
        lines.append(f"{i}. {name}: {count_s}{avg_s}")
    return "\n".join(lines)

@app.post("/query2", response_model=QueryResponse)
async def query_records_v2(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    query = req.query.strip()
    lang  = detect_language(query)

    # ── Keep your battle-tested special routes (these already work well) ──
    intent0 = detect_intent(query)
    if intent0 == "conversational":
        ai = await ask_ollama(build_conversational_prompt(query, lang=lang), max_tokens=120)
        return QueryResponse(ai_suggestion=ai, intent="conversational",
                             db_records_used=0, db_records_summary=[])
    if intent0 in ("diagram", "diagram_context"):
        # Delegate to your existing handler logic via the old endpoint behavior
        return await query_records(req)
    _is_explain = bool(_EXPLAIN_FOLLOWUP.search(query)) and (
        bool(req.last_ai_response) or bool(req.selected_text))
    if _is_explain:
        return await query_records(req)

    # ── 1. PLAN ──────────────────────────────────────────────────────────────
    plan = await make_plan(query)
    mode = plan.get("mode", "semantic")
    print(f"[V2] mode={mode} fmt={plan.get('output_format')} "
          f"chart={plan.get('wants_chart')} sql={plan.get('sql')!r}")

    # Honor explicit header model filter from the UI by injecting it as context
    if req.model_filter and plan.get("sql") and "model_name" not in plan["sql"].lower():
        # wrap: AND model_name LIKE %filter%
        s = plan["sql"]
        if re.search(r"\bwhere\b", s, re.IGNORECASE):
            plan["sql"] = re.sub(r"(?i)\bwhere\b",
                                 f"WHERE LOWER(model_name) LIKE LOWER('%{req.model_filter}%') AND ",
                                 s, count=1)

    # ── 2. EXECUTE ───────────────────────────────────────────────────────────
    rows: list[dict] = []
    executed_sql = ""
    source_label = "maintenance database (mttr_records)"
    confidence   = 1.0

    if mode == "chat":
        ai = await ask_ollama(build_conversational_prompt(query, lang=lang), max_tokens=120)
        return QueryResponse(ai_suggestion=ai, intent="conversational",
                             db_records_used=0, db_records_summary=[])

    if mode == "general":
        ai = await ask_ollama(build_general_prompt(query, lang=lang))
        return QueryResponse(ai_suggestion=ai, intent="general",
                             db_records_used=0, db_records_summary=[],
                             execution_path="general")

    if mode in ("sql", "hybrid", "tsg_sql") and plan.get("sql"):
        rows, executed_sql = await execute_sql_with_repair(plan["sql"])
        if mode == "tsg_sql":
            source_label = "official Troubleshooting Guide (tsg_records)"

    if mode == "hybrid":
        sem_q = plan.get("semantic_query") or query
        if lang == "hinglish":
            sem_q = _strip_hinglish_filler(sem_q)
        if rows:
            # Gate using chroma_id (unique) with problem-text as fallback
            gate_chroma = {r.get("chroma_id","") for r in rows if r.get("chroma_id")}
            gate_prob   = {(r.get("machine",""), (r.get("problem","") or "")[:80])
                           for r in rows}
            sem_records, confidence = await semantic_search(sem_q, use_multi_query=False)
            ranked = [r for r in sem_records
                      if r.get("chroma_id","") in gate_chroma
                      or (r.get("machine",""), (r.get("problem","") or "")[:80]) in gate_prob]
            ranked_keys = {r.get("chroma_id","") for r in ranked}
            leftover = [r for r in rows if r.get("chroma_id","") not in ranked_keys]
            rows = ranked + sort_by_loss_time(leftover)
        else:
            sem_records, confidence = await semantic_search(sem_q, use_multi_query=False)
            rows = sem_records
            source_label = "maintenance database (semantic fallback)"

    if mode == "semantic":
        sem_q = plan.get("semantic_query") or query
        if lang == "hinglish":
            sem_q = _strip_hinglish_filler(sem_q)
        rows, confidence = await semantic_search(sem_q, use_multi_query=False)
        source_label = "maintenance database (semantic match)"

    # SQL planned but planner gave empty SQL string → semantic fallback
    if mode == "sql" and not executed_sql and not rows:
        sem_q = plan.get("semantic_query") or query
        if lang == "hinglish":
            sem_q = _strip_hinglish_filler(sem_q)
        rows, confidence = await semantic_search(sem_q, use_multi_query=False)
        source_label = "maintenance database (semantic fallback)"
        print(f"[V2] sql mode but empty SQL, fell back to semantic")

    if mode == "tsg_semantic":
        rows = await tsg_retrieve(plan.get("semantic_query") or query)
        source_label = "official Troubleshooting Guide (tsg_records)"

    # SQL ran fine but returned nothing AND the question has a symptom → try semantic
    if mode == "sql" and executed_sql and not rows and plan.get("semantic_query"):
        rows, confidence = await semantic_search(plan["semantic_query"], use_multi_query=False)
        source_label = "maintenance database (semantic fallback)"

    # ── 2b. CHART (deterministic — LLM never touches the numbers) ───────────
    visualization = None
    if plan.get("wants_chart") and plan.get("chart_sql"):
        chart_rows, _ = await execute_sql_with_repair(plan["chart_sql"])
        visualization = _rows_to_chart_viz(chart_rows, plan.get("chart_type", "bar"),
                                           plan.get("answer_focus", "Chart"))

    # ── 2c. FLOWCHART output ─────────────────────────────────────────────────
    diagram_data = None
    if plan.get("output_format") == "flowchart" and rows:
        diagram_data = build_records_tree_diagram(query, rows[:6], "tree")

    # ── 3. SYNTHESIZE ────────────────────────────────────────────────────────
    # ── 3. SYNTHESIZE ────────────────────────────────────────────────────────
    # Pass ALL rows to synthesis (no truncation) — this is what Claude does
    page = rows[:PAGE_SIZE]   # only for record cards display
    synthesis_rows = rows     # full set goes to LLM

    if visualization and not rows:
        ai_response = visualization.get("summary") or "Here is the requested chart."

    elif (mode == "sql" and executed_sql and rows
          and not any("problem" in r or "solution" in r or "issue" in r
                      for r in rows[:3])):
        # Aggregate rows (COUNT/SUM/GROUP BY) — format deterministically, no LLM needed
        result_type = _infer_analytics_type(plan.get("sql", ""), rows)
        if result_type != "records":
            ai_response = _format_sql_aggregate(rows, result_type, query, plan)
        else:
            ai_response = await ask_ollama(
            build_v2_answer_prompt(query, plan, synthesis_rows, lang, source_label),
            max_tokens=1500 if plan.get("output_format") == "table" else MAX_TOKENS,
        )
    else:
        ai_response = await ask_ollama(
            build_v2_answer_prompt(query, plan, synthesis_rows, lang, source_label),
            max_tokens=1500 if plan.get("output_format") == "table" else MAX_TOKENS,
        )

    # ── Map rows to the record-card schema your frontend already renders ─────
    records_summary, tsg_summary = [], []
    if mode in ("tsg_sql", "tsg_semantic"):
        tsg_summary = [{"line_no": r.get("line_no",""), "machine": r.get("machine",""),
                        "issue": r.get("issue",""), "cause": r.get("cause",""),
                        "corrective": r.get("corrective","")} for r in page]
    else:
        for m in page:
            if "problem" not in m and "machine" not in m:
                continue   # aggregate rows (counts etc.) aren't record cards
            records_summary.append({
                "machine": m.get("machine",""), "model_name": m.get("model_name",""),
                "problem": m.get("problem",""), "solution": m.get("solution",""),
                "loss_time": _loss_time_label(m), "work_done_by": m.get("work_done_by",""),
                "date": m.get("iso_date", m.get("date","")), "shift": m.get("shift",""),
                "spare_parts": m.get("spare_parts",""),
                "image_b64": m.get("image_b64",""), "image_name": m.get("image_name",""),
                "image_mime": m.get("image_mime",""),
            })

    intent_out = ("tsg_lookup" if mode.startswith("tsg")
                  else "troubleshoot" if mode in ("semantic","hybrid")
                  else "db_lookup")

    return QueryResponse(
        ai_suggestion=ai_response, intent=intent_out,
        db_records_used=len(records_summary), db_records_summary=records_summary,
        diagram_data=diagram_data, visualization=visualization,
        retrieval_confidence=round(float(confidence), 4),
        total_records_found=len(rows), has_more=len(rows) > PAGE_SIZE,
        current_offset=0, suggest_tsg=False,
        tsg_records_used=len(tsg_summary), tsg_records_summary=tsg_summary,
        parsed_filters={"plan": plan.get("answer_focus","")} if plan.get("answer_focus") else None,
        execution_path=mode,
    )

# ─────────────────────────────────────────────────────────────────────────────
# /chat ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.history:
        raise HTTPException(status_code=400, detail="History cannot be empty.")
    latest_question = next((m.content for m in reversed(req.history) if m.role == "user"), "")
    lang = detect_language(latest_question)  
    history_text = "".join(
        f"\nTECHNICIAN: {m.content}\n" if m.role == "user" else f"\nASSISTANT: {m.content}\n"
        for m in req.history
    )
    db_section = ""
    try:
        machine_vals = _schema.column_values.get("machine", [])
        model_vals   = _schema.column_values.get("model_name", [])
        detected_machine = fuzzy_match_value(latest_question, machine_vals, threshold=1)
        detected_model   = fuzzy_match_value(latest_question, model_vals, threshold=1)
        follow_records, _ = await semantic_search(
            latest_question, machine_filter=detected_machine,
            model_filter=detected_model, use_multi_query=False,
        )
        if follow_records:
            db_section = f"\nRELEVANT MAINTENANCE RECORDS:\n{format_records_for_prompt(follow_records[:3])}\nUse if relevant.\n"
    except Exception:
        pass
    prompt = f"""You are an expert industrial maintenance engineer continuing a technical conversation.
{db_section}
CONVERSATION:
{history_text}
Continue naturally. Answer the technician's latest question directly and concisely.
{_lang_instruction(lang)}
"""
    ai_response = await ask_ollama(prompt, max_tokens=400)
    return ChatResponse(response=ai_response)


# ─────────────────────────────────────────────────────────────────────────────
# /translate ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/translate", response_model=TranslateResponse)
async def translate_text(req: TranslateRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    if req.language == "hindi":
        prompt = f"Translate to Hindi (Devanagari). Keep machine names/numbers in English. Output only:\n{req.text}\nHINDI:"
    elif req.language == "hinglish":
        prompt = f"Convert to Hinglish (Roman script). Keep technical terms in English. Output only:\n{req.text}\nHINGLISH:"
    else:
        raise HTTPException(status_code=400, detail="Language must be 'hindi' or 'hinglish'.")
    translated = await ask_ollama(prompt, max_tokens=600)
    return TranslateResponse(translated=translated)


# ─────────────────────────────────────────────────────────────────────────────
# METRICS / DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def _month_key(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"

def _prev_month_key(d: date) -> str:
    return f"{d.year - 1:04d}-12" if d.month == 1 else f"{d.year:04d}-{d.month - 1:02d}"


@app.get("/metrics")
async def metrics_overview(machine: Optional[str] = None):
    """
    MTTR dashboard data. Per machine + overall:
      failures, total downtime, MTTR (avg loss_time of known rows),
      this-month vs last-month failures, trend, and the worst offender.
    loss_time == -1 means 'unknown' and is excluded from MTTR.
    """
    filters: dict[str, Any] = {"_response_type": "list"}
    if machine:
        filters["machine"] = machine
    rows = _fetch_analytics_rows(filters)
    if not rows:
        return {"available": False, "message": "No maintenance records found."}

    today  = date.today()
    cur_mk, prev_mk = _month_key(today), _prev_month_key(today)

    blank = lambda: {"failures": 0, "downtime": 0.0, "valid": 0, "this_month": 0, "last_month": 0}
    per_machine: dict[str, dict] = defaultdict(blank)
    overall = blank()

    for r in rows:
        mc = (r.get("machine") or "Unknown").strip() or "Unknown"
        try:
            lt = float(r.get("loss_time", -1))
        except (TypeError, ValueError):
            lt = -1.0
        mk = str(r.get("iso_date", "") or "")[:7]
        for bucket in (per_machine[mc], overall):
            bucket["failures"] += 1
            if lt >= 0:
                bucket["downtime"] += lt
                bucket["valid"]    += 1
            if   mk == cur_mk:  bucket["this_month"] += 1
            elif mk == prev_mk: bucket["last_month"] += 1

    def finalize(b: dict) -> dict:
        delta = b["this_month"] - b["last_month"]
        return {
            "failures":           b["failures"],
            "total_downtime_min": round(b["downtime"], 1),
            "mttr_min":           round(b["downtime"] / b["valid"], 1) if b["valid"] else None,
            "this_month":         b["this_month"],
            "last_month":         b["last_month"],
            "trend":              "up" if delta > 0 else "down" if delta < 0 else "flat",
        }

    machine_cards = sorted(
        ({"machine": mc, **finalize(b)} for mc, b in per_machine.items()),
        key=lambda x: x["failures"], reverse=True,
    )
    worst = max(machine_cards, key=lambda x: x["total_downtime_min"], default=None)
    return {
        "available":      True,
        "overall":        finalize(overall),
        "worst_offender": worst["machine"] if worst else None,
        "machines":       machine_cards,
    }


@app.get("/repeat-offenders")
async def repeat_offenders(days: int = 90, min_occurrences: int = 3,
                           machine: Optional[str] = None):
    """
    Recurring-failure detection. Clusters (machine + problem) over a recent
    window and flags any combo that recurs >= min_occurrences. This is the
    'stop patching, do root-cause' signal.
    """
    from datetime import timedelta
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    filters: dict[str, Any] = {"_response_type": "list"}
    if machine:
        filters["machine"] = machine
    rows   = _fetch_analytics_rows(filters)
    recent = [r for r in rows if str(r.get("iso_date", "") or "") >= cutoff]

    by_machine: dict[str, list[str]] = defaultdict(list)
    for r in recent:
        mc = (r.get("machine") or "Unknown").strip() or "Unknown"
        by_machine[mc].append(r.get("problem", ""))

    offenders = []
    for mc, problems in by_machine.items():
        for c in _cluster_problems(problems):
            if c["count"] >= min_occurrences:
                offenders.append({"machine": mc, "problem": c["label"],
                                  "occurrences": c["count"], "window_days": days})
    offenders.sort(key=lambda x: x["occurrences"], reverse=True)
    return {"window_days": days, "min_occurrences": min_occurrences,
            "count": len(offenders), "offenders": offenders}

@app.get("/costly-problems")
async def costly_problems(machine: Optional[str] = None, top: int = 10):
    """Rank problem clusters by TOTAL downtime (frequency × severity)."""
    filters: dict[str, Any] = {"_response_type": "list"}
    if machine:
        filters["machine"] = machine
    rows = _fetch_analytics_rows(filters)
    # Tag each row with its cluster label, then sum downtime per cluster.
    clusters = _cluster_problems([r.get("problem", "") for r in rows])
    label_for = {}
    for c in clusters:
        for tok_label in [c["label"]]:
            label_for.setdefault(tok_label, c["label"])

    agg: dict[str, dict] = defaultdict(lambda: {"count": 0, "downtime": 0.0, "valid": 0})
    # Re-match each row to its nearest cluster label by token overlap.
    cluster_tokens = [(c["label"], set(_normalize_problem(c["label"]))) for c in clusters]
    for r in rows:
        toks = set(_normalize_problem(r.get("problem", "")))
        best, best_sim = None, 0.0
        for lbl, ctoks in cluster_tokens:
            sim = len(toks & ctoks) / len(toks | ctoks) if (toks | ctoks) else 0
            if sim > best_sim:
                best_sim, best = sim, lbl
        if best is None:
            continue
        try:
            lt = float(r.get("loss_time", -1))
        except (TypeError, ValueError):
            lt = -1.0
        agg[best]["count"] += 1
        if lt >= 0:
            agg[best]["downtime"] += lt
            agg[best]["valid"]    += 1

    out = [{"problem": k, "occurrences": v["count"],
            "total_downtime_min": round(v["downtime"], 1),
            "avg_downtime_min": round(v["downtime"] / v["valid"], 1) if v["valid"] else None}
           for k, v in agg.items()]
    out.sort(key=lambda x: x["total_downtime_min"], reverse=True)
    return {"top": out[:top]}

@app.get("/spare-parts-usage")
async def spare_parts_usage(machine: Optional[str] = None, top: int = 15):
    filters: dict[str, Any] = {"_response_type": "list"}
    if machine:
        filters["machine"] = machine
    counts: dict[str, int] = defaultdict(int)
    for r in _fetch_analytics_rows(filters):
        for part in re.split(r"[,;/]+", str(r.get("spare_parts", "") or "")):
            p = part.strip().lower()
            if len(p) >= 2:
                counts[p] += 1
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top]
    return {"parts": [{"part": p, "used_count": c} for p, c in ranked]}

# ─────────────────────────────────────────────────────────────────────────────
# /sync-sqlite ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/sync-sqlite")
async def sync_sqlite():
    try:
        col      = get_collection()
        all_data = col.get(include=["documents", "metadatas"])
        metas    = all_data["metadatas"]
        ids      = all_data["ids"]
        conn     = get_sqlite_conn()
        conn.execute("DELETE FROM mttr_records")
        batch = []
        for i, (meta, chroma_id) in enumerate(zip(metas, ids)):
            batch.append((
                chroma_id, meta.get("smd_line",""),
                (int(meta["line_no"]) if str(meta.get("line_no","")).lstrip("-").isdigit()
                 and int(meta.get("line_no", -1)) >= 0 else None),   # line_no
                meta.get("machine",""), meta.get("model_name",""),
                meta.get("problem",""), meta.get("solution",""),
                float(meta.get("loss_time",-1) or -1), meta.get("work_done_by",""),
                meta.get("date",""), meta.get("iso_date",""),         # iso_date
                meta.get("shift",""), meta.get("spare_parts",""),
                meta.get("image_b64",""), meta.get("image_name",""), meta.get("image_mime",""),
            ))
            if len(batch) >= 500:
                conn.executemany(
                    "INSERT INTO mttr_records (chroma_id, smd_line, line_no, machine, model_name, problem, solution, "
                    "loss_time, work_done_by, date, iso_date, shift, spare_parts, image_b64, image_name, image_mime) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch,
                ); batch = []
        if batch:
            conn.executemany(
                "INSERT INTO mttr_records (chroma_id, smd_line, line_no, machine, model_name, problem, solution, "
                "loss_time, work_done_by, date, iso_date, shift, spare_parts, image_b64, image_name, image_mime) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch,
            )
        conn.commit()
        count = conn.execute("SELECT COUNT(*) as cnt FROM mttr_records").fetchone()["cnt"]
        conn.close()
        conn2 = get_sqlite_conn()
        _schema.load(conn2)
        _schema.refresh_values(conn2)
        conn2.close()
        return {"status": "ok", "records_synced": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sync failed: {e}")

async def auto_sync_sqlite():
    if not sqlite_is_ready():
        try:
            result = await sync_sqlite()
            print(f"[SQLite] Auto-synced {result.get('records_synced', 0)} records from ChromaDB.")
        except Exception as e:
            print(f"[SQLite] Auto-sync failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# /schema ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/schema")
async def get_schema():
    return {
        "columns": {
            col: {
                "label":      meta["label"],
                "match_type": meta.get("match_type", "partial"),
                "col_type":   meta["col_type"],
                "known_values_count": len(_schema.column_values.get(col, [])),
                "sample_values":      _schema.column_values.get(col, [])[:10],
            }
            for col, meta in _schema.columns.items()
        },
        "total_columns": len(_schema.columns),
        "schema_ready":  _schema._ready,
    }


# ─────────────────────────────────────────────────────────────────────────────
# OCR HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(
            ['hi', 'en'], gpu=False, model_storage_directory='./ocr_models',
            download_enabled=False, verbose=False,
        )
    return _ocr_reader


def preprocess_image_for_ocr(image_bytes: bytes):
    import cv2, numpy as np
    from PIL import Image
    pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_np  = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    h, w    = img_bgr.shape[:2]
    if max(h, w) < 1800:
        scale   = 1800 / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=15)
    v1 = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 12)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    v1 = cv2.dilate(v1, kernel, iterations=1)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, v2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    alpha = 1.8; beta = 20
    v3 = np.clip(alpha * gray.astype(np.float32) + beta, 0, 255).astype(np.uint8)
    variants = [("clahe_adaptive", v1), ("otsu", v2), ("contrast_gray", v3)]
    if np.mean(gray) < 127: variants.append(("inverted", cv2.bitwise_not(v1)))
    return variants


@app.post("/opl-ocr", response_model=OcrResponse)
async def opl_ocr(req: OcrRequest):
    if not OCR_AVAILABLE:
        raise HTTPException(status_code=503, detail="OCR libraries not installed.")
    if not req.image_base64:
        raise HTTPException(status_code=400, detail="image_base64 is required.")
    try:
        image_bytes = base64.b64decode(req.image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data.")
    try:
        image_variants = preprocess_image_for_ocr(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Image preprocessing failed: {e}")

    reader = get_ocr_reader()
    all_results = []

    def run_ocr_pass(img_array, pass_name):
        try:
            return reader.readtext(
                img_array, detail=1, paragraph=False, decoder='beamsearch', beamWidth=10,
                batch_size=1, contrast_ths=0.1, adjust_contrast=0.5,
                text_threshold=0.5, low_text=0.3, link_threshold=0.3,
                mag_ratio=2.0, slope_ths=0.2, ycenter_ths=0.5,
                height_ths=0.5, width_ths=0.5, add_margin=0.15,
            )
        except Exception as ex:
            print(f"[OCR] Pass '{pass_name}' failed: {ex}"); return []

    loop = asyncio.get_event_loop()
    for pass_name, img_array in image_variants:
        results = await loop.run_in_executor(None, run_ocr_pass, img_array, pass_name)
        if results: all_results.extend(results)

    if not all_results:
        return OcrResponse(raw_text="", formatted_text="No text found.", confidence=0.0, language_detected="unknown")

    def bbox_iou(box1, box2) -> float:
        try:
            xs1=[p[0] for p in box1]; ys1=[p[1] for p in box1]
            xs2=[p[0] for p in box2]; ys2=[p[1] for p in box2]
            ix1,iy1=max(min(xs1),min(xs2)),max(min(ys1),min(ys2))
            ix2,iy2=min(max(xs1),max(xs2)),min(max(ys1),max(ys2))
            inter=max(0,ix2-ix1)*max(0,iy2-iy1)
            a1=(max(xs1)-min(xs1))*(max(ys1)-min(ys1)); a2=(max(xs2)-min(xs2))*(max(ys2)-min(ys2))
            union=a1+a2-inter; return inter/union if union>0 else 0
        except Exception: return 0

    deduplicated = []
    for (bbox, text, conf) in all_results:
        if not text.strip(): continue
        is_dup = False
        for i, (ebbox, etext, econf) in enumerate(deduplicated):
            if bbox_iou(bbox, ebbox) > 0.4:
                if conf > econf: deduplicated[i] = (bbox, text, conf)
                is_dup = True; break
        if not is_dup: deduplicated.append((bbox, text, conf))

    deduplicated.sort(key=lambda item: (round(min(p[1] for p in item[0]) / 30) * 30, min(p[0] for p in item[0])))
    raw_text = "\n".join(t for (_, t, _) in deduplicated)
    avg_conf  = sum(c for (_, _, c) in deduplicated) / len(deduplicated) if deduplicated else 0.0
    lang      = "Hindi" if sum(1 for c in raw_text if '\u0900' <= c <= '\u097F') > len(raw_text) * 0.1 else "English/Mixed"

    formatting_prompt = f"""You are an expert at correcting OCR-extracted text from Indian industrial maintenance OPL documents.
Fix broken Devanagari characters, restore missing matras, keep English words/numbers exactly.
Re-structure into: Title at top, then numbered steps. Do NOT translate. Output ONLY the cleaned text:
{raw_text}
CLEANED OPL TEXT:"""
    try:
        formatted_text = await ask_ollama(formatting_prompt, max_tokens=800)
    except Exception:
        formatted_text = raw_text

    return OcrResponse(raw_text=raw_text, formatted_text=formatted_text, confidence=round(avg_conf, 3), language_detected=lang)


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH & UTILITY ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/tsg-health")
async def tsg_health():
    tsg_col = get_tsg_collection()
    if tsg_col is None: return {"tsg_available": False, "tsg_records": 0}
    try:
        count = tsg_col.count()
        return {"tsg_available": count > 0, "tsg_records": count}
    except Exception:
        return {"tsg_available": False, "tsg_records": 0}


@app.get("/machines")
async def list_machines():
    return {"machines": _schema.column_values.get("machine", get_all_machines())}


@app.get("/models")
async def list_models():
    return {"models": _schema.column_values.get("model_name", get_all_models())}


@app.get("/health")
async def health():
    try:
        count = get_collection().count()
    except Exception:
        count = 0
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            ollama_ok = (await client.get(f"{OLLAMA_URL}/api/tags")).status_code == 200
    except Exception:
        ollama_ok = False
    tsg_col   = get_tsg_collection()
    tsg_count = 0
    if tsg_col:
        try: tsg_count = tsg_col.count()
        except Exception: pass
    sqlite_count = 0
    try:
        conn = get_sqlite_conn()
        sqlite_count = conn.execute("SELECT COUNT(*) as cnt FROM mttr_records").fetchone()["cnt"]
        conn.close()
    except Exception:
        pass
    return {
        "records_indexed":     count,
        "ollama_running":      ollama_ok,
        "tsg_records_indexed": tsg_count,
        "sqlite_records":      sqlite_count,
        "schema_columns":      list(_schema.columns.keys()),
        "schema_ready":        _schema._ready,
    }


@app.get("/", response_class=FileResponse)
async def serve_ui():
    ui_path = Path(__file__).parent / "index.html"
    if not ui_path.exists():
        return HTMLResponse("<h1>index.html not found.</h1>")
    return FileResponse(ui_path)

app.mount("/js", StaticFiles(directory="js"), name="js")


@app.get("/opl", response_class=FileResponse)
async def serve_opl():
    opl_path = Path(__file__).parent / "opl.html"
    if not opl_path.exists():
        return HTMLResponse("<h1>opl.html not found.</h1>")
    return FileResponse(opl_path)


register_v2_routes(app)

