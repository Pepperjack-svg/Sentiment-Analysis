# app.py
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from flask import Flask, request, render_template, jsonify, redirect, url_for, Response
from collections import deque
from datetime import datetime
from typing import List
import threading, json

# our modules
from modules.io_csv import read_csv_flex, clean_text
from modules.logic import analyze_many

app = Flask(__name__)

# ===== In-memory Recent Feed =====
_RECENT_LOCK = threading.Lock()
_RECENT = deque(maxlen=500)

def _truncate(s: str, n: int = 280) -> str:
    return s if len(s) <= n else s[:n-1] + "…"

def add_to_recent(items: List[dict]):
    now_iso = datetime.now().isoformat(timespec="seconds")
    with _RECENT_LOCK:
        for r in items:
            _RECENT.appendleft({
                "ts": now_iso,
                "text": _truncate(r["text"], 280),
                "sentiment": r["sentiment"],
                "sentiment_score": r["sentiment_score"],
                "rating_stars": r["rating_stars"],
                "stars_display": r["stars_display"],
                "emotion_label": r["emotion_label"],
                "emotion_score": r["emotion_score"],
                "emotion_emoji": r["emotion_emoji"],
            })

def snapshot_recent(limit: int = 50):
    with _RECENT_LOCK:
        return list(list(_RECENT)[:max(0, min(limit, len(_RECENT)))])

def clear_recent():
    with _RECENT_LOCK:
        _RECENT.clear()

# ===== Cache of last results shown (for download) =====
_LAST_LOCK = threading.Lock()
_LAST_RESULTS: List[dict] = []

def set_last_results(items: List[dict]):
    global _LAST_RESULTS
    with _LAST_LOCK:
        _LAST_RESULTS = list(items)

def get_last_results() -> List[dict]:
    with _LAST_LOCK:
        return list(_LAST_RESULTS)

# ===== Routes =====
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", results=None, messages=[], recent=snapshot_recent(500000))


@app.route("/analyze", methods=["POST"])
def analyze():
    messages = []
    texts = []

    input_text = clean_text(request.form.get("text", ""))
    if input_text:
        texts.append(input_text)

    file = request.files.get("file")
    if file and file.filename:
        tlist, err = read_csv_flex(file)
        if err: messages.append(err)
        else:   texts.extend(tlist)

    texts = [t for t in texts if t]
    if not texts:
        messages.append("Please enter text or upload a CSV containing a usable text column.")
        return render_template("index.html", results=[], messages=messages, recent=snapshot_recent(20))

    try:
        results = analyze_many(texts)
        set_last_results(results)     # <-- save current results for download
        add_to_recent(results)
    except Exception as e:
        messages.append(f"Failed to analyze: {e}")
        results = []

    return render_template("index.html", results=results, messages=messages, recent=snapshot_recent(500000))


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json(silent=True) or {}
    try:
        if "texts" in data:
            texts = [clean_text(t) for t in data["texts"] if clean_text(t)]
            if not texts:
                return jsonify({"ok": False, "error": "No non-empty texts provided."}), 400
            out = analyze_many(texts)
            set_last_results(out)      # make downloadable via UI too
            add_to_recent(out)
            return jsonify({"ok": True, "count": len(out), "data": out})
        elif "text" in data:
            t = clean_text(data["text"])
            if not t:
                return jsonify({"ok": False, "error": "Text is empty."}), 400
            out = analyze_many([t])[0]
            set_last_results([out])    # make downloadable via UI too
            add_to_recent([out])
            return jsonify({"ok": True, "data": out})
        else:
            return jsonify({"ok": False, "error": "Provide 'text' or 'texts' in JSON."}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/recent", methods=["GET"])
def api_recent():
    try:
        limit = int(request.args.get("limit", "50"))
    except ValueError:
        limit = 50
    return jsonify({"ok": True, "data": snapshot_recent(limit)})

@app.route("/recent/clear", methods=["POST"])
def recent_clear():
    clear_recent()
    if request.headers.get("Accept", "").startswith("text/html"):
        return redirect(url_for("index"))
    return jsonify({"ok": True, "message": "Recent feed cleared."})

# ----- Pretty JSON download endpoints -----
@app.route("/download/results.json", methods=["GET"])
def download_results_json():
    data = get_last_results() or []
    pretty = json.dumps(data, ensure_ascii=False, indent=2)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Response(
        pretty,
        mimetype="application/json; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="results-{ts}.json"'}
    )

@app.route("/download/recent.json", methods=["GET"])
def download_recent_json():
    try:
        limit = int(request.args.get("limit", "500"))
    except ValueError:
        limit = 500
    data = snapshot_recent(limit)
    pretty = json.dumps(data, ensure_ascii=False, indent=2)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Response(
        pretty,
        mimetype="application/json; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="recent-{ts}.json"'}
    )

@app.route("/json", methods=["GET"])
def json_hint():
    return jsonify({"message": "POST /api/analyze with {'text': '...'} or {'texts': ['...','...']}"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
