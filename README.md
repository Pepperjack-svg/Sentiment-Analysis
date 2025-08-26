# Sentiment Analysis (Flask + Transformers)

Analyze customer feedback with three pretrained NLP models—sentiment, star rating, and emotion—via a simple Flask web UI and a JSON API.

---

## Overview

This application runs three Hugging Face models in a single pass:

* Sentiment (Positive/Neutral/Negative): `cardiffnlp/twitter-roberta-base-sentiment-latest`
* Star rating (1–5): `nlptown/bert-base-multilingual-uncased-sentiment`
* Emotion (anger, fear, joy, love, sadness, surprise): `bhadresh-savani/distilbert-base-uncased-emotion`

Key features:

* Web UI for one-off text or CSV uploads
* Robust CSV ingestion (encoding and delimiter detection, skips malformed rows)
* Emotion “consistency gate” aligned with overall polarity (sentiment + stars)
* Batch inference for speed
* Programmatic access via `POST /api/analyze`

---

## Prerequisites

* Python 3.9+ (3.10 or newer recommended)
* pip
* Internet access on first run (to download model weights)

---

## Installation

```bash
git clone https://github.com/Pepperjack-svg/Sentiment-Analysis.git
cd Sentiment-Analysis

python -m venv .venv

# macOS / Linux
source .venv/bin/activate
# Windows PowerShell
# . .venv/Scripts/Activate.ps1

pip install -r requirements.txt
```

---

## Run

```bash
python app.py
# then open http://127.0.0.1:5000
```

Notes:

* The first run downloads model weights from Hugging Face and may take a few minutes.
* Subsequent runs use the local cache.

---

## Using the Web UI

1. Open `http://127.0.0.1:5000`.
2. Paste a single feedback in the textarea or upload a CSV.
3. Click Analyze to see:

   * Sentiment and confidence
   * Star rating (1–5)
   * Emotion (label) and top-3 emotions

### CSV Format

* Preferred column name: `text`
* If `text` is missing, the app tries to pick the first text-like column
* Supported delimiters: `, ; \t |` (auto-detected)
* Encodings tried: `utf-8`, `utf-8-sig`, `cp1252`, `latin-1`
* Malformed lines are skipped automatically

Example:

```csv
text
I love the new update! The UI is so smooth.
The package arrived late and support didn’t respond for two days.
```

---

## JSON API

Endpoint:

```
POST /api/analyze
Content-Type: application/json
```

Single text:

```json
{ "text": "Loved the service and delivery speed!" }
```

Multiple texts:

```json
{ "texts": ["Great value!", "Support never replied."] }
```

Example response (single):

```json
{
  "ok": true,
  "data": {
    "text": "Loved the service and delivery speed!",
    "sentiment": "Positive",
    "sentiment_score": 0.9873,
    "rating_stars": 5,
    "emotion": "joy",
    "emotion_label": "Joy",
    "emotion_score": 0.9421,
    "emotion_top3": [["joy",0.9421],["love",0.0331],["surprise",0.0127]],
    "emotion_dist": {
      "anger": 0.0003, "fear": 0.0011, "joy": 0.9421,
      "love": 0.0331, "sadness": 0.0107, "surprise": 0.0127
    }
  }
}
```

PowerShell example:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/api/analyze `
  -ContentType "application/json" `
  -Body (@{ texts = @("Fast delivery!", "Support never replied.") } | ConvertTo-Json)
```

cURL example:

```bash
curl -X POST http://127.0.0.1:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"texts":["Fast delivery!","Support never replied."]}'
```

---

## Emotion Consistency Gate

The raw emotion classifier can misclassify sarcastic or strongly polarized feedback. To improve interpretability:

* A simple vote combines:

  * Sentiment (Positive/Neutral/Negative)
  * Star rating (≥4 positive, ≤2 negative)
* If the raw top emotion conflicts with this vote, the app switches to the highest-probability emotion from the corresponding set:

  * Positive: joy, love, surprise
  * Negative: anger, fear, sadness

The top-3 raw emotions are still displayed for transparency.

---

## Troubleshooting

* CSV tokenizing errors: the parser auto-detects delimiter and encoding and skips malformed rows. Ensure the file is a real CSV (not XLSX) and that quotes are balanced.
* Emotion looks off: the consistency gate corrects many obvious mismatches; review the top-3 list to see model uncertainty.
* Slow first run: model weights are being downloaded and cached.

---
