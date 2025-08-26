import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from flask import Flask, request, render_template, jsonify
from transformers import pipeline
import pandas as pd
import io, csv, re
from typing import List, Tuple, Optional

app = Flask(__name__)

# ----- Emotion set for bhadresh-savani/distilbert-base-uncased-emotion -----
EMOTION_LABELS = ["anger", "fear", "joy", "love", "sadness", "surprise"]
POS_EMOS = {"joy", "love", "surprise"}
NEG_EMOS = {"anger", "fear", "sadness"}
EMOTION_EMOJI = {
    "anger": "😠",
    "fear": "😨",
    "joy": "😊",
    "love": "❤️",
    "sadness": "😢",
    "surprise": "😲",
}

# ----- Lazy model loaders -----
_pipes = {"sentiment": None, "rating": None, "emotion": None}

def get_pipes():
    if _pipes["sentiment"] is None:
        _pipes["sentiment"] = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            truncation=True
        )
    if _pipes["rating"] is None:
        _pipes["rating"] = pipeline(
            "text-classification",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            truncation=True
        )
    if _pipes["emotion"] is None:
        _pipes["emotion"] = pipeline(
            "text-classification",
            model="bhadresh-savani/distilbert-base-uncased-emotion",
            return_all_scores=True,
            truncation=True
        )
    return _pipes

# ----- Helpers -----
def clean_text(s) -> str:
    if s is None:
        return ""
    s = str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _decode_best(raw: bytes) -> Tuple[str, str]:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace"), "latin-1"

def _sniff_delim(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"]).delimiter
    except csv.Error:
        return ","

def _read_csv_attempts(text: str, sep: str) -> Optional[pd.DataFrame]:
    """
    Try several tolerant parses for nasty CSVs. Returns DataFrame or None.
    """
    attempts = [
        dict(sep=sep, engine="python", on_bad_lines="skip"),
        dict(sep=sep, engine="python", on_bad_lines="skip", quoting=csv.QUOTE_MINIMAL),
        dict(sep=sep, engine="python", on_bad_lines="skip", quoting=csv.QUOTE_NONE, escapechar="\\"),
        # headerless fallback
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
    """
    Robust CSV reader:
      - tries encodings
      - sniffs delimiter
      - parses with tolerant options, skipping bad lines
      - chooses 'text' column else first text-like column else first column
      - drops empties
    Returns (texts, error_message). error_message is None if OK.
    """
    try:
        raw = file_storage.read()
        if not raw:
            return [], "Uploaded file is empty."

        text, enc = _decode_best(raw)
        sample = text[:20000]
        primary_sep = _sniff_delim(sample)

        # Try primary sep first, then other common seps
        for sep in [primary_sep, ",", ";", "\t", "|"]:
            df = _read_csv_attempts(text, sep)
            if df is None or df.empty:
                continue

            # Normalize headers when available
            if df.columns.dtype == "object":
                norm_cols = [str(c).strip().lower() for c in df.columns]
                df.columns = norm_cols

            # Select column
            col = None
            if "text" in df.columns:
                col = "text"
            else:
                # Pick first object-like column
                for c in df.columns:
                    if df[c].dtype == object:
                        col = c
                        break
                if col is None:
                    col = df.columns[0]

            series = df[col].fillna("").astype(str)
            texts = [clean_text(x) for x in series.tolist()]
            texts = [t for t in texts if t]

            if texts:
                return texts, None

        return [], "Could not find any usable text column/rows (after skipping malformed lines)."
    except Exception as e:
        return [], f"Failed to parse CSV: {e}"

def normalize_sentiment_label(lbl: str) -> str:
    if not lbl:
        return "Neutral"
    low = lbl.lower()
    if low.startswith("label_"):
        # twitter-roberta legacy: LABEL_0,1,2 -> neg,neu,pos
        mapping = {"0": "Negative", "1": "Neutral", "2": "Positive"}
        return mapping.get(low.split("_")[-1], "Neutral")
    if "neg" in low:
        return "Negative"
    if "neu" in low:
        return "Neutral"
    if "pos" in low:
        return "Positive"
    return lbl.capitalize()

def stars_from_label(label: str) -> Optional[int]:
    m = re.search(r"([1-5])", str(label))
    return int(m.group(1)) if m else None

def choose_consistent_emotion(emo_dist: dict, sentiment: str, stars: Optional[int]):
    """
    Pick a final emotion consistent with overall polarity vote from:
      - sentiment (pos/neu/neg)
      - star rating (>=4 => pos, <=2 => neg)
    If the top raw emotion already agrees, keep it; else switch to the best in the agreed set.
    Returns: (final_label, final_prob, sorted_list)
    """
    # Sort raw emotions by prob
    sorted_items = sorted(emo_dist.items(), key=lambda kv: kv[1], reverse=True)
    raw_top, raw_p = sorted_items[0]

    # Vote polarity
    votes = 0
    if sentiment == "Positive":
        votes += 1
    elif sentiment == "Negative":
        votes -= 1
    if stars is not None:
        if stars >= 4:
            votes += 1
        elif stars <= 2:
            votes -= 1

    preferred: Optional[set] = None
    if votes > 0:
        preferred = POS_EMOS
    elif votes < 0:
        preferred = NEG_EMOS

    if preferred and raw_top not in preferred:
        for lbl, prob in sorted_items:
            if lbl in preferred:
                return lbl, prob, sorted_items  # switched for consistency
    return raw_top, raw_p, sorted_items  # keep raw top

def analyze_many(texts: List[str]) -> List[dict]:
    pipes = get_pipes()

    # Batch
    sent_out = pipes["sentiment"](texts)
    rating_out = pipes["rating"](texts)
    emo_out = pipes["emotion"](texts)

    results = []
    for i, t in enumerate(texts):
        # Sentiment
        s_label = normalize_sentiment_label(sent_out[i]["label"])
        s_score = float(sent_out[i]["score"])

        # Rating
        r_label = rating_out[i]["label"]
        stars = stars_from_label(r_label)

        # Emotions
        raw_list = emo_out[i]  # list[{'label':..,'score':..}]
        dist = {d["label"].lower(): float(d["score"]) for d in raw_list}
        # Ensure all six labels exist
        for k in EMOTION_LABELS:
            dist.setdefault(k, 0.0)

        final_lbl, final_p, sorted_items = choose_consistent_emotion(dist, s_label, stars)
        top3 = [(lbl, prob) for lbl, prob in sorted_items[:3]]

        results.append({
            "text": t,
            "sentiment": s_label,
            "sentiment_score": round(s_score, 4),
            "rating_stars": stars,
            "stars_display": ("★" * stars + "☆" * (5 - stars)) if stars else "—",
            "emotion": final_lbl,
            "emotion_label": final_lbl.capitalize(),
            "emotion_score": round(final_p, 4),
            "emotion_emoji": EMOTION_EMOJI.get(final_lbl, "🎯"),
            "emotion_top3": [(lbl, round(prob, 4)) for lbl, prob in top3],
            "emotion_dist": {k: round(dist[k], 4) for k in EMOTION_LABELS},
        })
    return results

# ----- Routes -----
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    results = []
    messages = []

    input_text = clean_text(request.form.get("text", ""))
    file = request.files.get("file")

    texts = []
    if input_text:
        texts.append(input_text)

    if file and file.filename:
        tlist, err = read_csv_flex(file)
        if err:
            messages.append(err)
        else:
            texts.extend(tlist)

    texts = [t for t in texts if t]
    if not texts:
        messages.append("Please enter text or upload a CSV containing a usable text column.")
        return render_template("index.html", results=[], messages=messages)

    try:
        results = analyze_many(texts)
    except Exception as e:
        messages.append(f"Failed to analyze: {e}")

    return render_template("index.html", results=results, messages=messages)

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json(silent=True) or {}
    try:
        if "texts" in data:
            texts = [clean_text(t) for t in data["texts"] if clean_text(t)]
            if not texts:
                return jsonify({"ok": False, "error": "No non-empty texts provided."}), 400
            out = analyze_many(texts)
            return jsonify({"ok": True, "count": len(out), "data": out})
        elif "text" in data:
            t = clean_text(data["text"])
            if not t:
                return jsonify({"ok": False, "error": "Text is empty."}), 400
            out = analyze_many([t])[0]
            return jsonify({"ok": True, "data": out})
        else:
            return jsonify({"ok": False, "error": "Provide 'text' or 'texts' in JSON."}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/json", methods=["GET"])
def json_hint():
    return jsonify({
        "message": "POST to /api/analyze with {'text': '...'} or {'texts': ['...','...']}"
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
