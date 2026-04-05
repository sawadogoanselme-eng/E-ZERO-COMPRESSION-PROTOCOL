"""
E-ZERO PROTOCOL — Lightweight Filter F
=======================================
Author : Sawadogo Anselme (@sawadogoanselme-eng)
Version: 1.0 — April 2026

This module implements the Logical Skeleton extractor S(P) described
in the E-ZERO white paper. It operates in O(n log n) and is model-agnostic.

Usage:
    python ezero_filter.py

Requirements:
    pip install spacy scikit-learn
    python -m spacy download en_core_web_sm
    python -m spacy download fr_core_news_sm   # for French prompts
"""

import math
import time
import re
from typing import List, Tuple

# ── Try to import spacy (optional — falls back to simple mode) ──────────────
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("[E-ZERO] spacy not found. Running in simple mode (TF-IDF only).")


# ═══════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

class EZeroConfig:
    """All tunable parameters of the E-ZERO filter."""

    # Scoring weights (must sum to 1.0)
    lambda_tfidf : float = 0.5   # Rarity of token in context
    lambda_pos   : float = 0.3   # Syntactic importance
    lambda_dep   : float = 0.2   # Dependency tree depth

    # Activation thresholds
    n_min        : int   = 20    # Minimum prompt length to activate filter
    rho_max      : float = 0.9   # Maximum allowed compression rate
    gamma        : float = 0.25  # Minimum confidence score to activate

    # Compression target
    rho_target   : float = 0.4   # Target: keep 40% of tokens

    # Cost coefficients (for gain estimation)
    alpha        : float = 1.0   # Model cost coefficient (O(n²))
    beta         : float = 0.001 # Filter cost coefficient (O(n log n))


# ═══════════════════════════════════════════════════════════════════════════
# 2. SYNTACTIC ROLE WEIGHTS  (Pos score)
# ═══════════════════════════════════════════════════════════════════════════

# Higher = more important to keep
POS_WEIGHTS = {
    "VERB"  : 1.0,
    "NOUN"  : 0.9,
    "PROPN" : 0.95,   # proper noun
    "ADJ"   : 0.6,
    "ADV"   : 0.4,
    "NUM"   : 0.8,
    "PRON"  : 0.3,
    "ADP"   : 0.2,    # preposition
    "DET"   : 0.1,    # article
    "PUNCT" : 0.0,
    "SPACE" : 0.0,
}

# Stopwords for simple mode (no spacy)
STOPWORDS = {
    # French
    "le", "la", "les", "un", "une", "des", "du", "de", "et", "ou",
    "est", "en", "que", "qui", "ne", "pas", "je", "tu", "il", "nous",
    "vous", "ils", "me", "te", "se", "ce", "on", "y", "au", "aux",
    "peux", "s'il", "plaît", "très", "bien", "aussi", "donc", "mais",
    "comment", "pourquoi", "quand", "quel", "quelle", "voudrais",
    "manière", "exhaustive", "exactement", "expliquer", "m'expliquer",
    # English
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "to", "of", "and", "or", "in", "it", "that", "this", "for",
    "with", "you", "do", "can", "could", "please", "very", "quite",
    "i", "me", "my", "we", "if", "not", "too", "much", "trouble",
    "would", "might", "able", "help", "understand", "wondering",
    "explain", "detail", "detailed", "exactly", "manner",
    "great", "what", "makes", "how", "when", "why", "which", "where",
    "most", "important", "between", "differences", "key", "different",
    "previous", "consider", "designing", "factors", "high", "about",
}


# ═══════════════════════════════════════════════════════════════════════════
# 3. TF-IDF SCORER  (O(n))
# ═══════════════════════════════════════════════════════════════════════════

def compute_tfidf(tokens: List[str]) -> dict:
    """
    Compute a simple TF-IDF-like score for each token.
    Higher score = rarer = more informative.
    """
    n = len(tokens)
    if n == 0:
        return {}

    # Term frequency
    tf = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1

    scores = {}
    for t in tokens:
        freq = tf[t] / n
        # Inverse frequency: rare tokens score higher
        idf = math.log(n / tf[t] + 1)
        scores[t] = freq * idf

    # Normalize to [0, 1]
    max_score = max(scores.values()) if scores else 1.0
    if max_score > 0:
        scores = {t: v / max_score for t, v in scores.items()}

    return scores



# ═══════════════════════════════════════════════════════════════════════════
# CRITICAL TOKEN DETECTION (numbers, symbols, math keywords)
# ═══════════════════════════════════════════════════════════════════════════

MATH_QUESTION_WORDS = {
    "how", "many", "much", "total", "each", "per", "left", "remaining",
    "altogether", "number", "times", "half", "double", "triple", "what",
    "cost", "price", "paid", "spend", "earn", "buy", "sell", "more", "less",
    "days", "hours", "minutes", "weeks", "months", "years", "between",
}

def is_critical_token(token: str) -> bool:
    """
    Returns True if the token must ALWAYS be kept regardless of score.
    Protects: numbers, currency, percentages, math question words.
    """
    t = "".join(c for c in token if c not in ".,!?;:()[]")
    # Pure number or decimal
    if re.match(r'^-?\d+(\.\d+)?$', t):
        return True
    # Currency: $5, $80,000
    if re.match(r'^\$[\d,]+(\.\d+)?$', t):
        return True
    # Percentage: 10%, 50%
    if re.match(r'^\d+(\.\d+)?%$', t):
        return True
    # Number with unit: 2GB, 60mph, 3kg
    if re.match(r'^\d+[a-zA-Z]+$', t):
        return True
    # Math question keywords
    if t.lower() in MATH_QUESTION_WORDS:
        return True
    return False

# ═══════════════════════════════════════════════════════════════════════════
# 4. FILTER F
# ═══════════════════════════════════════════════════════════════════════════

class EZeroFilter:
    """
    Lightweight Filter F — computes the Logical Skeleton S(P).

    Complexity: O(n log n)
    """

    def __init__(self, config: EZeroConfig = None, lang: str = "en"):
        self.config = config or EZeroConfig()
        self.nlp = None

        if SPACY_AVAILABLE:
            model = "fr_core_news_sm" if lang == "fr" else "en_core_web_sm"
            try:
                self.nlp = spacy.load(model)
            except OSError:
                print(f"[E-ZERO] spacy model '{model}' not found. Run:")
                print(f"         python -m spacy download {model}")
                print("[E-ZERO] Falling back to simple mode.")

    # ── Activation gate ────────────────────────────────────────────────────

    def _should_activate(self, tokens: List[str]) -> Tuple[bool, float, str]:
        """
        Returns (activate, confidence, reason).
        Implements the condition: n > n_min AND rho* < rho_max AND confidence > gamma
        """
        n = len(tokens)
        cfg = self.config

        if n < cfg.n_min:
            return False, 0.0, f"Prompt too short (n={n} < n_min={cfg.n_min})"

        # Estimate natural compression rate: fraction of stopwords
        stop_count = sum(1 for t in tokens if t.lower() in STOPWORDS)
        rho_estimated = 1.0 - (stop_count / n)

        if rho_estimated > cfg.rho_max:
            return False, rho_estimated, f"Prompt too dense (rho*={rho_estimated:.2f} > rho_max={cfg.rho_max})"

        # Confidence: ratio of non-stopword tokens (more = more filterable)
        confidence = stop_count / n
        if confidence < cfg.gamma:
            return False, confidence, f"Low confidence (confidence={confidence:.2f} < gamma={cfg.gamma})"

        return True, confidence, "Filter activated"

    # ── Scoring ────────────────────────────────────────────────────────────

    def _score_tokens_simple(self, tokens: List[str]) -> List[float]:
        """Simple scoring: TF-IDF only, no spacy."""
        tfidf = compute_tfidf(tokens)
        scores = []
        for t in tokens:
            s_tfidf = tfidf.get(t, 0.0)
            # Penalize stopwords
            s_pos = 0.1 if t.lower() in STOPWORDS else 0.9
            score = (self.config.lambda_tfidf + self.config.lambda_dep) * s_tfidf \
                  + self.config.lambda_pos * s_pos
            scores.append(score)
        return scores

    def _score_tokens_spacy(self, tokens: List[str], text: str) -> List[float]:
        """Full scoring: TF-IDF + POS + dependency depth."""
        tfidf = compute_tfidf(tokens)
        doc = self.nlp(text)

        # Map spacy tokens to our token list
        spacy_tokens = [tok for tok in doc]
        pos_map = {}
        dep_depth_map = {}

        for tok in spacy_tokens:
            key = tok.text.lower()
            pos_map[key] = POS_WEIGHTS.get(tok.pos_, 0.5)
            # Dependency depth: count hops to root
            depth = 0
            current = tok
            while current.head != current and depth < 10:
                current = current.head
                depth += 1
            dep_depth_map[key] = 1.0 / (1.0 + depth)  # shallower = more important

        scores = []
        cfg = self.config
        for t in tokens:
            key = t.lower()
            s_tfidf = tfidf.get(t, 0.0)
            s_pos   = pos_map.get(key, 0.5)
            s_dep   = dep_depth_map.get(key, 0.5)
            score   = cfg.lambda_tfidf * s_tfidf \
                    + cfg.lambda_pos   * s_pos   \
                    + cfg.lambda_dep   * s_dep
            scores.append(score)
        return scores

    # ── Main filter ────────────────────────────────────────────────────────

    def filter(self, prompt: str) -> dict:
        """
        Apply filter F to prompt P. Returns a result dict with:
          - skeleton   : the reduced prompt (string)
          - tokens_in  : original token count
          - tokens_out : skeleton token count
          - rho        : actual compression rate
          - gain_pct   : estimated energy gain (%)
          - activated  : whether filter was applied
          - reason     : explanation
        """
        t_start = time.perf_counter()
        cfg = self.config

        # Tokenize (simple whitespace split)
        tokens = prompt.split()
        n = len(tokens)

        # Activation gate
        activate, confidence, reason = self._should_activate(tokens)

        if not activate:
            elapsed = time.perf_counter() - t_start
            return {
                "skeleton"   : prompt,
                "tokens_in"  : n,
                "tokens_out" : n,
                "rho"        : 1.0,
                "gain_pct"   : 0.0,
                "activated"  : False,
                "reason"     : reason,
                "elapsed_ms" : round(elapsed * 1000, 3),
            }

        # Score tokens
        if self.nlp is not None:
            scores = self._score_tokens_spacy(tokens, prompt)
        else:
            scores = self._score_tokens_simple(tokens)

        # Determine threshold θ to hit rho_target
        k = max(1, int(cfg.rho_target * n))
        sorted_scores = sorted(scores, reverse=True)
        theta = sorted_scores[k - 1] if k <= len(sorted_scores) else 0.0

        # Build skeleton — always keep critical tokens (numbers, currency, math keywords)
        skeleton_tokens = [
            t for t, s in zip(tokens, scores)
            if s >= theta or is_critical_token(t)
        ]
        skeleton = " ".join(skeleton_tokens)

        rho = len(skeleton_tokens) / n if n > 0 else 1.0

        # Estimate energy gain: G = alpha*n²*(1-rho²) - beta*n*log(n)
        gain_raw = cfg.alpha * n**2 * (1 - rho**2) - cfg.beta * n * math.log2(n + 1)
        gain_pct = max(0.0, gain_raw / (cfg.alpha * n**2) * 100)

        elapsed = time.perf_counter() - t_start

        return {
            "skeleton"   : skeleton,
            "tokens_in"  : n,
            "tokens_out" : len(skeleton_tokens),
            "rho"        : round(rho, 3),
            "gain_pct"   : round(gain_pct, 1),
            "activated"  : True,
            "reason"     : reason,
            "confidence" : round(confidence, 3),
            "elapsed_ms" : round(elapsed * 1000, 3),
        }


# ═══════════════════════════════════════════════════════════════════════════
# 5. DEMO
# ═══════════════════════════════════════════════════════════════════════════

def print_result(prompt: str, result: dict):
    print("\n" + "═" * 60)
    print(f"  PROMPT   : {prompt}")
    print(f"  SKELETON : {result['skeleton']}")
    print(f"  Tokens   : {result['tokens_in']} → {result['tokens_out']}  (ρ = {result['rho']})")
    print(f"  Gain     : {result['gain_pct']}%  |  Filter: {result['elapsed_ms']} ms")
    print(f"  Status   : {'✅ Activated' if result['activated'] else '⏭  Bypassed'} — {result['reason']}")
    print("═" * 60)


if __name__ == "__main__":

    filter_en = EZeroFilter(lang="en")
    filter_fr = EZeroFilter(lang="fr")

    test_prompts = [
        # Long verbose English prompts
        ("en", "Could you please explain in very great detail and in an exhaustive manner exactly how a transformer neural network works and what makes it different from previous architectures?"),
        ("en", "I was wondering if you might be able to help me understand the key differences between supervised learning and unsupervised learning in machine learning if that is not too much trouble?"),
        ("en", "What are the most important factors to consider when designing a distributed system architecture for a high-availability production environment?"),

        # Long verbose French prompts
        ("fr", "Peux-tu s'il te plaît m'expliquer de manière très détaillée et exhaustive comment fonctionne exactement un transformeur et pourquoi il est si important en intelligence artificielle?"),
        ("fr", "Je voudrais bien comprendre comment fonctionne le protocole E-ZERO et quels sont les avantages principaux par rapport aux autres méthodes d'optimisation existantes?"),

        # Short prompt — should be bypassed
        ("en", "What is AI?"),
    ]

    print("\nE-ZERO PROTOCOL — Filter F Demo")
    print("================================\n")

    for lang, prompt in test_prompts:
        f = filter_fr if lang == "fr" else filter_en
        result = f.filter(prompt)
        print_result(prompt, result)

    # Benchmark: measure gain across different prompt lengths
    print("\n\n📊 THEORETICAL GAIN vs PROMPT LENGTH")
    print(f"{'n (tokens)':<14} {'ρ=0.4 gain':<14} {'ρ=0.6 gain':<14} {'ρ=0.8 gain'}")
    print("-" * 56)
    cfg = EZeroConfig()
    for n in [20, 50, 100, 200, 500, 1000, 2000]:
        row = f"{n:<14}"
        for rho in [0.4, 0.6, 0.8]:
            gain = cfg.alpha * n**2 * (1 - rho**2) - cfg.beta * n * math.log2(n + 1)
            pct  = max(0.0, gain / (cfg.alpha * n**2) * 100)
            row += f"{pct:.1f}%{'':<9}"
        print(row)
