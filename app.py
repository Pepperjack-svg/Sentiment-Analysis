import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from flask import Flask, request, render_template, jsonify
from transformers import pipeline
import pandas as pd
import re

app = Flask(__name__)

# --------- Lazy model loaders (initialized on first use) ----------
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

EMOJI = {
    "anger":"😠","disgust":"🤢","fear":"😨","joy":"😊","sadness":"😢","surprise":"😲"
}

def clean_text(s):
    if not isinstance(s, str):
        s = str(s)
    return re.sub(r"\s+", " ", s).strip()

def run_models_on_text(text):
    pipes = get_pipes()
    t = clean_text(text)

    # Sentiment (pos/neu/neg)
    s_out = pipes["sentiment"](t)[0]  # {'label': 'positive', 'score': ...}
    sentiment = s_out["label"].capitalize()
    sentiment_score = round(float(s_out["score"]), 4)

    # Rating (1-5 stars)
    r_out = pipes["rating"](t)[0]["label"]  # e.g., "4 stars" or "1 star"
    stars = int(re.search(r"\d", r_out).group()) if re.search(r"\d", r_out) else None

    # Emotion (top + distribution)
    e_out = pipes["emotion"](t)[0]  # list of dicts with label/score
    e_sorted = sorted(e_out, key=lambda x: x["score"], reverse=True)
    top_emotion = e_sorted[0]["label"]
    top_emotion_score = round(float(e_sorted[0]["score"]), 4)
    # Compact distribution as label:prob%
    emos = {d["label"]: round(float(d["score"]), 4) for d in e_sorted}

    return {
        "text": t,
        "sentiment": sentiment,
        "sentiment_score": sentiment_score,
        "rating_stars": stars,
        "emotion": top_emotion,
        "emotion_score": top_emotion_score,
        "emotion_dist": emos
    }

# ------------------------ Routes ------------------------

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    """Handles form posts from UI. Text OR CSV upload."""
    results = []
    error = None

    input_text = request.form.get("text", "").strip()
    file = request.files.get("file")

    try:
        if file and file.filename:
            # Expect a CSV with a 'text' column; if not, use the first column
            df = pd.read_csv(file)
            if "text" not in df.columns:
                df.columns = [c.strip().lower() for c in df.columns]
            if "text" not in df.columns:
                # fallback: first column
                df = df.rename(columns={df.columns[0]: "text"})
            texts = df["text"].fillna("").astype(str).tolist()
            results = [run_models_on_text(t) for t in texts if t.strip()]
        elif input_text:
            results = [run_models_on_text(input_text)]
        else:
            error = "Please enter text or upload a CSV file."
    except Exception as e:
        error = f"Failed to analyze: {e}"

    return render_template("index.html", results=results, error=error, EMOJI=EMOJI)

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    JSON API
    Body:
      { "text": "..." }
      or
      { "texts": ["...","..."] }
    """
    data = request.get_json(silent=True) or {}
    try:
        if "texts" in data:
            texts = [t for t in data["texts"] if str(t).strip()]
            out = [run_models_on_text(t) for t in texts]
            return jsonify({"ok": True, "count": len(out), "data": out})
        elif "text" in data:
            out = run_models_on_text(data["text"])
            return jsonify({"ok": True, "data": out})
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
