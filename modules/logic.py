# modules/logic.py
import re
from typing import List, Optional, Tuple, Dict
from transformers import pipeline

# ----- Emotion config for bhadresh-savani/distilbert-base-uncased-emotion -----
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

# ----- Lazy pipelines -----
_pipes = {"sentiment": None, "rating": None, "emotion": None}

def _get_pipes():
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

# ----- Helpers used by analysis -----
def normalize_sentiment_label(lbl: str) -> str:
    if not lbl:
        return "Neutral"
    low = lbl.lower()
    if low.startswith("label_"):
        # legacy mapping: LABEL_0/1/2 -> neg/neu/pos
        return {"0": "Negative", "1": "Neutral", "2": "Positive"}.get(low.split("_")[-1], "Neutral")
    if "neg" in low: return "Negative"
    if "neu" in low: return "Neutral"
    if "pos" in low: return "Positive"
    return lbl.capitalize()

def stars_from_label(label: str) -> Optional[int]:
    m = re.search(r"([1-5])", str(label))
    return int(m.group(1)) if m else None

def _choose_consistent_emotion(emo_dist: Dict[str, float], sentiment: str, stars: Optional[int]) -> Tuple[str, float, list]:
    """Force top emotion to agree with overall polarity vote (sentiment + stars)."""
    sorted_items = sorted(emo_dist.items(), key=lambda kv: kv[1], reverse=True)
    raw_top, raw_p = sorted_items[0]

    votes = 0
    votes += 1 if sentiment == "Positive" else (-1 if sentiment == "Negative" else 0)
    if stars is not None:
        votes += 1 if stars >= 4 else (-1 if stars <= 2 else 0)

    preferred = POS_EMOS if votes > 0 else (NEG_EMOS if votes < 0 else None)
    if preferred and raw_top not in preferred:
        for lbl, prob in sorted_items:
            if lbl in preferred:
                return lbl, prob, sorted_items
    return raw_top, raw_p, sorted_items

# ----- Main batch analyzer -----
def analyze_many(texts: List[str]) -> List[dict]:
    pipes = _get_pipes()

    sent_out  = pipes["sentiment"](texts)
    rating_out = pipes["rating"](texts)
    emo_out   = pipes["emotion"](texts)

    results = []
    for i, t in enumerate(texts):
        s_label = normalize_sentiment_label(sent_out[i]["label"])
        s_score = float(sent_out[i]["score"])

        r_label = rating_out[i]["label"]
        stars   = stars_from_label(r_label)

        # emotions
        raw_list = emo_out[i]  # list[{label,score}]
        dist = {d["label"].lower(): float(d["score"]) for d in raw_list}
        for k in EMOTION_LABELS:
            dist.setdefault(k, 0.0)

        final_lbl, final_p, sorted_items = _choose_consistent_emotion(dist, s_label, stars)
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
            #"emotion_top3": [(lbl, round(prob, 4)) for lbl, prob in top3],
            #"emotion_dist": {k: round(dist[k], 4) for k in EMOTION_LABELS},
        })
    return results
