

import argparse
import re
import sqlite3
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

import base64
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
COLLECTION_NAME  = "mttr_records"
DB_PATH          = "./chroma_db"
# EMBED_MODEL_PATH = "./local_model"      # BAAI/bge-base-en-v1.5 saved locally
# clean_excel.py AND app.py — must be identical
EMBED_MODEL_PATH = "./local_model_bhasha"
SQLITE_PATH      = "./mttr_records.db"
IMAGES_DIR       = "./images"

# ── Column names in YOUR Excel — adjust if your headers differ ────────────────
COL_SMD_LINE     = "Line No."
COL_MACHINE      = "Machine"
COL_MODEL        = "Model Name"         # optional
COL_PROBLEM      = "Issue"
COL_SOLUTION     = "Action"
COL_LOSS_TIME    = "Loss Time"          # optional
COL_WORK_DONE_BY = "Work Done By"       # optional
COL_DATE         = "Date"               # optional
COL_SHIFT        = "Shift"              # optional
COL_SPARE_PARTS  = "Spare Parts"        # optional
COL_IMAGE        = "Image Path"         # optional


# ─────────────────────────────────────────────────────────────────────────────
# TEXT / IMAGE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def clean_text(val) -> str:
    if pd.isna(val):
        return ""
    text = str(val).strip()
    text = re.sub(r"\s+", " ", text)
    # Keep printable ASCII + Devanagari (Hindi)
    text = re.sub(r"[^\x20-\x7E\u0900-\u097F]", "", text)
    return text.strip()


def parse_loss_time(val) -> float:
    """
    Parse loss time from various formats:
      - Plain number:        45       → 45.0 (minutes)
      - String with unit:   "45 min"  → 45.0
                            "1.5 hr"  → 90.0
                            "1h 30m"  → 90.0
      - Empty / NaN                   → -1.0  (sentinel: unknown, sorts last)
    """
    if pd.isna(val):
        return -1.0
    s = str(val).strip().lower()
    if not s:
        return -1.0
    try:
        return float(s)
    except ValueError:
        pass
    hours   = re.search(r"(\d+(?:\.\d+)?)\s*h", s)
    minutes = re.search(r"(\d+(?:\.\d+)?)\s*m", s)
    total = 0.0
    if hours:
        total += float(hours.group(1)) * 60
    if minutes:
        total += float(minutes.group(1))
    if total > 0:
        return total
    digits = re.findall(r"\d+(?:\.\d+)?", s)
    if digits:
        return float(digits[0])
    return -1.0


def parse_iso_date(val) -> str:
    """
    ★ NEW (Phase 2 — 4.4): Normalize any date format to ISO 'YYYY-MM-DD'.
    Returns '' if unparseable. This makes SQL date comparisons correct
    regardless of how dates were stored in the original Excel file.
    """
    if pd.isna(val):
        return ""
    s = str(val).strip()
    if not s:
        return ""
    # pandas can parse most common formats with dayfirst=True for DD/MM/YYYY
    try:
        parsed = pd.to_datetime(s, dayfirst=True, errors="raise")
        return parsed.strftime("%Y-%m-%d")
    except Exception:
        pass
    # Fallback: try common explicit formats
    for fmt in (
        "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y",
        "%d %b %Y", "%d %B %Y", "%b %d %Y", "%B %d %Y",
        "%d-%b-%Y", "%d-%B-%Y", "%Y/%m/%d",
    ):
        try:
            from datetime import datetime
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    # Last resort: extract 4-digit year only
    m = re.search(r"\b(20\d{2})\b", s)
    if m:
        return f"{m.group(1)}-01-01"
    return ""


def parse_line_no(val) -> int | None:
    """
    ★ NEW (Phase 2 — 4.4): Extract the integer SMD line number from
    values like "Line 3", "3", "SMD-03", etc.
    Returns None if no integer found (stored as NULL in SQLite).
    """
    if pd.isna(val):
        return None
    s = str(val).strip()
    m = re.search(r"(\d+)", s)
    if m:
        return int(m.group(1))
    return None


def encode_image_to_base64(image_filename: str) -> str:
    if not image_filename:
        return ""
    path = Path(IMAGES_DIR) / image_filename.strip()
    if not path.exists():
        print(f"  ⚠  Image not found: {path}")
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_mime(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png",  ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(ext, "image/jpeg")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD & CLEAN EXCEL
# ─────────────────────────────────────────────────────────────────────────────
def load_and_clean(filepath: str) -> pd.DataFrame:
    print(f"[1/4] Reading MTTR file: {filepath}")
    df = pd.read_excel(filepath, engine="openpyxl")
    df.columns = df.columns.str.strip()

    # Required columns — script fails if any are absent
    required = [COL_SMD_LINE, COL_MACHINE, COL_PROBLEM, COL_SOLUTION]
    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"Missing required column: '{col}'\n"
                f"Available columns: {list(df.columns)}\n"
                f"Tip: update the COL_* variables at the top of this script."
            )
    print("  Required columns verified ✅")

    # Optional columns — warn if absent, default to empty
    def _opt(col_name: str, default="") -> bool:
        present = col_name in df.columns
        if not present:
            print(f"  ⚠  Optional column '{col_name}' not found — defaulting to '{default}'.")
        else:
            print(f"  Optional column '{col_name}' found ✅")
        return present

    has_model        = _opt(COL_MODEL)
    has_loss_time    = _opt(COL_LOSS_TIME)
    has_work_done_by = _opt(COL_WORK_DONE_BY)
    has_date         = _opt(COL_DATE)
    has_shift        = _opt(COL_SHIFT)
    has_spare_parts  = _opt(COL_SPARE_PARTS)
    has_image        = _opt(COL_IMAGE)

    # Build the cleaned dataframe
    cleaned = pd.DataFrame({
        "smd_line"   : df[COL_SMD_LINE].apply(clean_text),
        "machine"    : df[COL_MACHINE].apply(clean_text),
        "model_name" : df[COL_MODEL].apply(clean_text)          if has_model        else pd.Series([""] * len(df)),
        "problem"    : df[COL_PROBLEM].apply(clean_text),
        "solution"   : df[COL_SOLUTION].apply(clean_text),
        "loss_time"  : df[COL_LOSS_TIME].apply(parse_loss_time) if has_loss_time    else pd.Series([-1.0] * len(df)),
        "work_done_by": df[COL_WORK_DONE_BY].apply(clean_text)  if has_work_done_by else pd.Series([""] * len(df)),
        "date"       : (
            df[COL_DATE].apply(lambda v: str(v).strip()[:10] if pd.notna(v) else "")
            if has_date else pd.Series([""] * len(df))
        ),
        # ★ NEW: iso_date — always-correct YYYY-MM-DD for SQL date filtering
        "iso_date"   : (
            df[COL_DATE].apply(parse_iso_date)
            if has_date else pd.Series([""] * len(df))
        ),
        # ★ NEW: line_no — integer for exact SMD line matching
        "line_no"    : df[COL_SMD_LINE].apply(parse_line_no),
        "shift"      : df[COL_SHIFT].apply(clean_text)          if has_shift        else pd.Series([""] * len(df)),
        "spare_parts": df[COL_SPARE_PARTS].apply(clean_text)    if has_spare_parts  else pd.Series([""] * len(df)),
        # Image fields
        "image_b64"  : (
            df[COL_IMAGE].apply(lambda v: encode_image_to_base64(clean_text(v)))
            if has_image else pd.Series([""] * len(df))
        ),
        "image_name" : df[COL_IMAGE].apply(clean_text)          if has_image        else pd.Series([""] * len(df)),
        "image_mime" : (
            df[COL_IMAGE].apply(lambda v: get_image_mime(clean_text(v)) if pd.notna(v) and str(v).strip() else "")
            if has_image else pd.Series([""] * len(df))
        ),
    })

    # Drop rows with empty problem or solution; deduplicate
    before = len(cleaned)
    cleaned = cleaned[
        (cleaned["problem"].str.len() > 5) &
        (cleaned["solution"].str.len() > 5)
    ].drop_duplicates(
        subset=["smd_line", "machine", "model_name", "problem", "solution"]
    )
    print(f"  Rows: {before} → {len(cleaned)} after cleaning & deduplication")

    # Summary of unique values found
    if has_model:
        unique_models = sorted([m for m in cleaned["model_name"].dropna().unique() if m])
        print(f"  Unique models  ({len(unique_models)}): {unique_models}")
    if has_shift:
        unique_shifts = sorted([s for s in cleaned["shift"].dropna().unique() if s])
        print(f"  Unique shifts  ({len(unique_shifts)}): {unique_shifts}")
    if has_work_done_by:
        unique_workers = sorted([w for w in cleaned["work_done_by"].dropna().unique() if w])
        print(f"  Unique workers ({len(unique_workers)}): {unique_workers}")

    # ★ NEW: Report iso_date parse success rate
    if has_date:
        parsed_count = (cleaned["iso_date"] != "").sum()
        total_count  = len(cleaned)
        print(f"  Date parsing   : {parsed_count}/{total_count} rows have valid iso_date ✅")
        if parsed_count < total_count:
            bad_dates = cleaned[cleaned["iso_date"] == ""]["date"].unique()[:5]
            print(f"  ⚠  Unparseable date samples: {list(bad_dates)}")

    # ★ NEW: Report line_no parse success rate
    parsed_lines = cleaned["line_no"].notna().sum()
    print(f"  Line no parsing: {parsed_lines}/{len(cleaned)} rows have valid line_no ✅")

    return cleaned.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — INDEX INTO SQLITE
# ─────────────────────────────────────────────────────────────────────────────
def init_sqlite(conn: sqlite3.Connection):
    """Drop and recreate mttr_records to ensure schema is always up to date."""
    conn.execute("DROP TABLE IF EXISTS mttr_records")
    conn.execute("""
        CREATE TABLE mttr_records (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            chroma_id    TEXT,
            smd_line     TEXT,
            line_no      INTEGER,
            machine      TEXT COLLATE NOCASE,
            model_name   TEXT COLLATE NOCASE,
            problem      TEXT,
            solution     TEXT,
            loss_time    REAL DEFAULT -1,
            work_done_by TEXT COLLATE NOCASE,
            date         TEXT,
            iso_date     TEXT,
            shift        TEXT COLLATE NOCASE,
            spare_parts  TEXT,
            image_b64    TEXT,
            image_name   TEXT,
            image_mime   TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_machine      ON mttr_records(machine)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_model        ON mttr_records(model_name)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_work_done_by ON mttr_records(work_done_by)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_date         ON mttr_records(date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_iso_date     ON mttr_records(iso_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_shift        ON mttr_records(shift)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_spare_parts  ON mttr_records(spare_parts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_smd_line     ON mttr_records(smd_line)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_line_no      ON mttr_records(line_no)")
    conn.commit()

def build_fts5(conn: sqlite3.Connection):
    """
    ★ NEW (Phase 2 — 4.5): Build an FTS5 virtual table over problem + solution.
    This lets SQL keyword searches run in the same query as structured filters,
    removing the dependency on the separate BM25 path for keyword matching.
    """
    # Drop and recreate so it's always in sync after a full rebuild
    conn.execute("DROP TABLE IF EXISTS mttr_fts")
    conn.execute("""
        CREATE VIRTUAL TABLE mttr_fts
        USING fts5(
            problem,
            solution,
            spare_parts,
            content='mttr_records',
            content_rowid='id'
        )
    """)
    conn.execute("""
        INSERT INTO mttr_fts(rowid, problem, solution, spare_parts)
        SELECT id, problem, solution, spare_parts FROM mttr_records
    """)
    count = conn.execute("SELECT COUNT(*) FROM mttr_fts").fetchone()[0]
    print(f"  FTS5 index built — {count} rows ✅")


def write_to_sqlite(df: pd.DataFrame, chroma_ids: list[str]):
    """
    Write all cleaned rows into SQLite, then build the FTS5 index.
    chroma_ids maps each DataFrame row to its ChromaDB document ID (rec_0, rec_1, …).
    """
    print(f"[3/4] Writing {len(df)} records to SQLite at '{SQLITE_PATH}'")
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    init_sqlite(conn)

    # # Wipe existing MTTR records so this is always a clean rebuild
    # conn.execute("DELETE FROM mttr_records")
    # conn.commit()

    batch = []
    for i, (_, row) in enumerate(df.iterrows()):
        # line_no is int or None (NULL in SQLite)
        line_no_val = row["line_no"]
        if pd.isna(line_no_val):
            line_no_val = None
        else:
            line_no_val = int(line_no_val)

        batch.append((
            chroma_ids[i],              # chroma_id  — links back to ChromaDB doc
            row["smd_line"],
            line_no_val,                # ★ NEW
            row["machine"],
            row["model_name"],
            row["problem"],
            row["solution"],
            float(row["loss_time"]),
            row["work_done_by"],
            row["date"],
            row["iso_date"],            # ★ NEW
            row["shift"],
            row["spare_parts"],
            row["image_b64"],
            row["image_name"],
            row["image_mime"],
        ))
        if len(batch) >= 500:
            conn.executemany(
                """INSERT INTO mttr_records
                   (chroma_id, smd_line, line_no, machine, model_name, problem, solution,
                    loss_time, work_done_by, date, iso_date, shift, spare_parts,
                    image_b64, image_name, image_mime)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                batch,
            )
            conn.commit()
            batch = []

    if batch:
        conn.executemany(
            """INSERT INTO mttr_records
               (chroma_id, smd_line, line_no, machine, model_name, problem, solution,
                loss_time, work_done_by, date, iso_date, shift, spare_parts,
                image_b64, image_name, image_mime)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            batch,
        )
        conn.commit()

    count = conn.execute("SELECT COUNT(*) as cnt FROM mttr_records").fetchone()[0]
    print(f"  SQLite write complete — {count} rows in mttr_records ✅")

    # ★ NEW: Build FTS5 index after all rows are committed
    build_fts5(conn)

    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — INDEX INTO CHROMADB
# ─────────────────────────────────────────────────────────────────────────────
def index_to_chromadb(df: pd.DataFrame) -> list[str]:
    """
    Index all records into ChromaDB.
    Returns the list of chroma_ids in the same row order as df,
    so write_to_sqlite() can store the cross-reference.

    The embedding document includes all filterable fields so semantic
    search can surface records based on context even without explicit SQL filters.
    """
    print(f"[2/4] Connecting to ChromaDB at '{DB_PATH}'")
    client = chromadb.PersistentClient(path=DB_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL_PATH
    )

    # Always rebuild the collection from scratch for consistency
    try:
        client.delete_collection(COLLECTION_NAME)
        print("  Deleted existing MTTR collection (rebuilding fresh)")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    documents = []
    metadatas = []
    ids       = []

    for i, row in df.iterrows():
        chroma_id = f"rec_{i}"

        # Richer embedding document — includes all filterable fields
        model_part      = f" Model: {row['model_name']}."          if row["model_name"]    else ""
        shift_part      = f" Shift: {row['shift']}."               if row["shift"]         else ""
        worker_part     = f" Performed by: {row['work_done_by']}." if row["work_done_by"]  else ""
        spare_part_text = f" Spare parts used: {row['spare_parts']}." if row["spare_parts"] else ""
        date_part       = f" Date: {row['date']}."                 if row["date"]          else ""
        line_no_part    = f" Line number: {int(row['line_no'])}."  if pd.notna(row.get("line_no")) else ""  # ★ NEW

        doc = (
            f"SMD Line: {row['smd_line']}."
            f"{line_no_part}"
            f" Machine: {row['machine']}."
            f"{model_part}"
            f" Problem: {row['problem']}."
            f" Solution: {row['solution']}."
            f"{shift_part}"
            f"{worker_part}"
            f"{spare_part_text}"
            f"{date_part}"
        )
        documents.append(doc)

        metadatas.append({
            "smd_line"    : row["smd_line"],
            "line_no"     : int(row["line_no"]) if pd.notna(row.get("line_no")) else -1,  # ★ NEW (-1 = unknown)
            "machine"     : row["machine"],
            "model_name"  : row["model_name"],
            "problem"     : row["problem"],
            "solution"    : row["solution"],
            "loss_time"   : float(row["loss_time"]),
            "work_done_by": row["work_done_by"],
            "date"        : row["date"],
            "iso_date"    : row["iso_date"],     # ★ NEW
            "shift"       : row["shift"],
            "spare_parts" : row["spare_parts"],
            "image_b64"   : row["image_b64"],
            "image_name"  : row["image_name"],
            "image_mime"  : row["image_mime"],
        })
        ids.append(chroma_id)

    # Batch insert (ChromaDB recommends ≤500 per call)
    batch_size = 500
    for start in range(0, len(documents), batch_size):
        end = min(start + batch_size, len(documents))
        collection.add(
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )
        print(f"  Indexed {end}/{len(documents)} MTTR records into ChromaDB")

    print(f"  ChromaDB indexing complete — {len(documents)} records ✅")
    return ids


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Clean MTTR Excel and index into ChromaDB + SQLite"
    )
    parser.add_argument("--file",         required=True, help="Path to your MTTR Excel file")
    parser.add_argument("--skip-sqlite",  action="store_true",
                        help="Skip SQLite write (ChromaDB only — filters won't work)")
    args = parser.parse_args()

    # Step 1: Load and clean
    df = load_and_clean(args.file)

    # Step 2: Index into ChromaDB (richer chunks) — returns chroma_ids
    chroma_ids = index_to_chromadb(df)

    # Step 3: Write to SQLite using the same chroma_ids as cross-reference keys
    if not args.skip_sqlite:
        write_to_sqlite(df, chroma_ids)
    else:
        print("  [--skip-sqlite] SQLite write skipped.")

    print("\n✅ MTTR indexing complete!")
    print(f"   ChromaDB : {DB_PATH}  (collection: {COLLECTION_NAME})")
    if not args.skip_sqlite:
        print(f"   SQLite   : {SQLITE_PATH}  (table: mttr_records)")
    print("\nRestart the backend: uvicorn app:app --host 127.0.0.1 --port 8000 --reload")


if __name__ == "__main__":
    main()
