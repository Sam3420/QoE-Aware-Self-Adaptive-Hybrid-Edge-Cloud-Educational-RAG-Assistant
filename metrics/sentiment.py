"""Feedback sentiment from a small, transparent word lexicon (a labelled estimate, not a model).

Score in [−1, 1] = (positive − negative) / (positive + negative); a negator ("not", "never", "no",
"n't") within the two previous words flips a hit. None when the comment has no lexicon words.
English only.
"""
import re

POSITIVE = frozenset("""good great excellent helpful clear useful easy understand understood love liked like
nice fast quick amazing awesome perfect thanks thank interesting simple better best enjoyed enjoy fun""".split())
NEGATIVE = frozenset("""bad poor confusing confused unclear slow wrong boring hard difficult useless hate
terrible awful worse worst error broken irrelevant long annoying lost""".split())
NEGATORS = frozenset({"not", "never", "no", "hardly", "nothing"})
_TOKEN = re.compile(r"[a-z']+")


def lexicon_sentiment(text: str | None) -> float | None:
    """Sentiment in [−1, 1], or None when nothing in the text is in the lexicon."""
    if not text:
        return None
    tokens = _TOKEN.findall(text.lower())
    pos = neg = 0
    for i, tok in enumerate(tokens):
        polarity = 1 if tok in POSITIVE else -1 if tok in NEGATIVE else 0
        if not polarity:
            continue
        window = tokens[max(0, i - 2):i]
        if any(w in NEGATORS or w.endswith("n't") for w in window):
            polarity = -polarity
        pos, neg = pos + (polarity > 0), neg + (polarity < 0)
    return None if pos + neg == 0 else round((pos - neg) / (pos + neg), 4)
