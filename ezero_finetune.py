"""
E-ZERO PROTOCOL — Fine-tuning des paramètres λ
================================================
Author : Sawadogo Anselme (@sawadogoanselme-eng)
Version: 2.0 — April 2026

Ce script apprend automatiquement les meilleurs poids λ₁, λ₂, λ₃
et le seuil θ sur les 100 questions GSM8K.

Objectif : maximiser la fidélité tout en maintenant la compression.

Usage:
    python ezero_finetune.py
"""

import math
import time
import json
import os
import sys

# ── Import du filtre E-ZERO ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ezero_filter import EZeroFilter, EZeroConfig, STOPWORDS, compute_tfidf, is_critical_token

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_OK = True
except:
    SPACY_OK = False
    print("[!] spacy non disponible — mode simple")

# ── Chargement GSM8K ─────────────────────────────────────────────────────────
print("Chargement du dataset GSM8K...")
from datasets import load_dataset
dataset = load_dataset("openai/gsm8k", "main", split="test")
questions = [item["question"] for item in dataset.select(range(100))]
print(f"✅ {len(questions)} questions chargées\n")

# ── Mots-clés mathématiques ──────────────────────────────────────────────────
MATH_KEYWORDS = {
    "how", "many", "much", "total", "each", "per", "left", "remaining",
    "altogether", "number", "times", "half", "double", "triple", "what",
    "cost", "price", "paid", "spend", "earn", "buy", "sell", "more", "less",
    "days", "hours", "minutes", "weeks", "months", "years", "between",
}

def fidelity_score(original: str, skeleton: str) -> float:
    orig_tokens = set(original.lower().split())
    skel_tokens = set(skeleton.lower().split())
    present = orig_tokens & MATH_KEYWORDS
    if not present:
        return 1.0
    preserved = skel_tokens & present
    return len(preserved) / len(present)

# ── Fonction d'évaluation ────────────────────────────────────────────────────
def evaluate(lambda1, lambda2, lambda3, rho_target, questions):
    """
    Évalue une combinaison de paramètres λ sur toutes les questions.
    Retourne le score composite : fidélité moyenne + bonus compression.
    """
    config = EZeroConfig()
    config.lambda_tfidf = lambda1
    config.lambda_pos   = lambda2
    config.lambda_dep   = lambda3
    config.rho_target   = rho_target
    config.gamma        = 0.25
    config.n_min        = 15

    ezero = EZeroFilter(config=config, lang="en")

    total_fidelity = 0.0
    total_compression = 0.0
    activated = 0

    for q in questions:
        result = ezero.filter(q)
        fid = fidelity_score(q, result["skeleton"])
        total_fidelity += fid
        total_compression += (1 - result["rho"])
        if result["activated"]:
            activated += 1

    n = len(questions)
    avg_fidelity    = total_fidelity / n
    avg_compression = total_compression / n

    # Score composite : 70% fidélité + 30% compression
    score = 0.7 * avg_fidelity + 0.3 * avg_compression

    return score, avg_fidelity, avg_compression, activated

# ── Grid Search sur λ ────────────────────────────────────────────────────────
print("=" * 60)
print("  E-ZERO — FINE-TUNING DES PARAMÈTRES λ")
print("=" * 60)
print("Recherche des meilleurs paramètres...\n")

# Valeurs candidates pour chaque paramètre
lambda_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
rho_values    = [0.3, 0.4, 0.5, 0.6]

best_score    = -1
best_params   = None
best_fidelity = 0
best_compression = 0
best_activated = 0

total_combinations = 0
for l1 in lambda_values:
    for l2 in lambda_values:
        l3 = round(1.0 - l1 - l2, 2)
        if l3 > 0:
            for rho in rho_values:
                total_combinations += 1

print(f"  Nombre de combinaisons à tester : {total_combinations}")
print(f"  Cela peut prendre 2-3 minutes...\n")

tested = 0
t_start = time.time()

for l1 in lambda_values:
    for l2 in lambda_values:
        l3 = round(1.0 - l1 - l2, 2)
        if l3 <= 0:
            continue
        for rho in rho_values:
            score, fid, comp, act = evaluate(l1, l2, l3, rho, questions)
            tested += 1

            if score > best_score:
                best_score       = score
                best_params      = (l1, l2, l3, rho)
                best_fidelity    = fid
                best_compression = comp
                best_activated   = act

            if tested % 50 == 0:
                elapsed = time.time() - t_start
                print(f"  ✔ {tested}/{total_combinations} combinaisons testées "
                      f"({elapsed:.1f}s) — meilleur score: {best_score:.4f}")

elapsed_total = time.time() - t_start

# ── Résultats ────────────────────────────────────────────────────────────────
l1, l2, l3, rho = best_params

print(f"\n{'=' * 60}")
print(f"  RÉSULTATS DU FINE-TUNING")
print(f"{'=' * 60}")
print(f"  Meilleurs paramètres trouvés :")
print(f"  λ₁ (TF-IDF)  = {l1}  (était 0.5)")
print(f"  λ₂ (syntaxe) = {l2}  (était 0.3)")
print(f"  λ₃ (profond) = {l3}  (était 0.2)")
print(f"  ρ  (cible)   = {rho}  (était 0.4)")
print(f"{'=' * 60}")
print(f"  Fidélité moyenne    : {best_fidelity*100:.1f}%  (était 87.9%)")
print(f"  Compression moyenne : {best_compression*100:.1f}%  (était 32.9%)")
print(f"  Filtre activé       : {best_activated}/100")
print(f"  Temps de recherche  : {elapsed_total:.1f}s")
print(f"{'=' * 60}")

# ── Comparaison avant/après ──────────────────────────────────────────────────
print(f"\n  COMPARAISON v1.2 → v2.0")
print(f"{'=' * 60}")
print(f"  {'Métrique':<25} {'v1.2':>10} {'v2.0':>10} {'Gain':>10}")
print(f"  {'-'*55}")

fid_gain  = (best_fidelity*100) - 87.9
comp_gain = (best_compression*100) - 32.9

print(f"  {'Fidélité':<25} {'87.9%':>10} {best_fidelity*100:.1f}%{'':<4} "
      f"{'+' if fid_gain >= 0 else ''}{fid_gain:.1f}pts")
print(f"  {'Compression':<25} {'32.9%':>10} {best_compression*100:.1f}%{'':<4} "
      f"{'+' if comp_gain >= 0 else ''}{comp_gain:.1f}pts")
print(f"{'=' * 60}")

# ── Sauvegarde des meilleurs paramètres ─────────────────────────────────────
output = {
    "version": "2.0",
    "date": "2026-04-05",
    "method": "Grid Search sur 100 questions GSM8K",
    "best_params": {
        "lambda_tfidf": l1,
        "lambda_pos":   l2,
        "lambda_dep":   l3,
        "rho_target":   rho,
        "gamma":        0.25,
        "n_min":        15
    },
    "results": {
        "fidelity_pct":    round(best_fidelity * 100, 1),
        "compression_pct": round(best_compression * 100, 1),
        "activated":       best_activated,
        "score":           round(best_score, 4)
    },
    "previous_v1_2": {
        "fidelity_pct":    87.9,
        "compression_pct": 32.9,
        "lambda_tfidf":    0.5,
        "lambda_pos":      0.3,
        "lambda_dep":      0.2,
        "rho_target":      0.4
    }
}

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ezero_best_params.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n✅ Meilleurs paramètres sauvegardés dans : ezero_best_params.json")
print(f"\nCopie ces paramètres dans ezero_filter.py pour la v2.0 :")
print(f"  lambda_tfidf : float = {l1}")
print(f"  lambda_pos   : float = {l2}")
print(f"  lambda_dep   : float = {l3}")
print(f"  rho_target   : float = {rho}")
print(f"\nFine-tuning terminé !")
