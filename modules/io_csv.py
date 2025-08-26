# modules/io_csv.py
import csv, io, re
from typing import List, Tuple, Optional
import pandas as pd

def clean_text(s) -> str:
    if s is None: return ""
    s = str(s)
    return re.sub(r"\s+", " ", s).strip()

def _decode_best(raw: bytes):
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace"), "latin-1"

def _sniff_delim(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",",";","\t","|"]).delimiter
    except csv.Error:
        return ","

def _read_csv_attempts(text: str, sep: str) -> Optional[pd.DataFrame]:
    attempts = [
        dict(sep=sep, engine="python", on_bad_lines="skip"),
        dict(sep=sep, engine="python", on_bad_lines="skip", quoting=csv.QUOTE_MINIMAL),
        dict(sep=sep, engine="python", on_bad_lines="skip", quoting=csv.QUOTE_NONE, escapechar="\\"),
        dict(sep=sep, engine="python", on_bad_lines="skip", header=None),
        dict(sep=sep, engine="python", on_bad_lines="skip", header=None, quoting=csv.QUOTE_NONE, escapechar="\\"),
    ]
    for opts in attempts:
        try:
            df = pd.read_csv(io.StringIO(text), **opts)
            if df is not None:
                return df
        except Exception:
            continue
    return None

def read_csv_flex(file_storage) -> Tuple[List[str], str]:
    """Tolerant CSV reader; returns (texts, error_message)."""
    try:
        raw = file_storage.read()
        if not raw:
            return [], "Uploaded file is empty."

        text, _enc = _decode_best(raw)
        sample = text[:20000]
        primary_sep = _sniff_delim(sample)

        for sep in [primary_sep, ",", ";", "\t", "|"]:
            df = _read_csv_attempts(text, sep)
            if df is None or df.empty:
                continue

            # normalize headers
            try:
                df.columns = [str(c).strip().lower() for c in df.columns]
            except Exception:
                pass

            # pick column
            if hasattr(df, "columns") and "text" in df.columns:
                col = "text"
            else:
                col = df.columns[0]
                for c in df.columns:
                    if df[c].dtype == object:
                        col = c; break

            series = df[col].fillna("").astype(str)
            texts = [clean_text(x) for x in series.tolist()]
            texts = [t for t in texts if t]
            if texts:
                return texts, None

        return [], "Could not find any usable text column/rows (after skipping malformed lines)."
    except Exception as e:
        return [], f"Failed to parse CSV: {e}"
