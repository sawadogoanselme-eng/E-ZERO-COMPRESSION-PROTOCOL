"""
E-ZERO PROTOCOL — BBH Benchmark
=================================
Author : Sawadogo Anselme (@sawadogoanselme-eng)
Version: 2.0 — April 2026

Ce script teste le filtre E-ZERO sur 100 questions du dataset BBH
(Big-Bench Hard) — dataset de raisonnement logique.

Objectif : prouver que E-ZERO fonctionne sur un type de prompt
différent des maths (GSM8K), renforçant sa généralisation.

Usage:
    python ezero_benchmark_bbh.py
"""

import math
import time
import json
import os
import sys

# ── Import du filtre E-ZERO ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ezero_filter import EZeroFilter, EZeroConfig

# ── Chargement BBH ───────────────────────────────────────────────────────────
print("Chargement du dataset BBH...")
try:
    from datasets import load_dataset
    # BBH contient plusieurs sous-tâches — on prend les plus connues
    subtasks = [
        "logical_deduction_five_objects",
        "causal_judgement",
        "date_understanding",
        "tracking_shuffled_objects_five_objects",
        "reasoning_about_colored_objects",
    ]
    
    all_questions = []
    for task in subtasks:
        try:
            ds = load_dataset("lukaemon/bbh", task, split="test")
            questions = [item["input"] for item in ds.select(range(20))]
            all_questions.extend(questions)
            print(f"  ✅ {task}: {len(questions)} questions")
        except Exception as e:
            print(f"  ⚠️  {task}: {e}")
    
    # Limiter à 100 questions
    questions = all_questions[:100]
    print(f"\n✅ {len(questions)} questions BBH chargées au total\n")

except Exception as e:
    print(f"❌ Erreur: {e}")
    exit(1)

# ── Mots-clés de raisonnement à préserver ────────────────────────────────────
REASONING_KEYWORDS = {
    # Logique
    "if", "then", "therefore", "because", "since", "unless",
    "not", "and", "or", "but", "however", "although",
    # Comparaison
    "before", "after", "left", "right", "between", "above", "below",
    "first", "second", "third", "last", "next",
    # Quantité
    "all", "some", "none", "every", "each", "any",
    "more", "less", "most", "least", "only",
    # Questions
    "what", "which", "who", "where", "when", "how", "why",
    "true", "false", "correct", "incorrect",
}

def fidelity_score(original: str, skeleton: str) -> float:
    orig_tokens = set(original.lower().split())
    skel_tokens = set(skeleton.lower().split())
    present = orig_tokens & REASONING_KEYWORDS
    if not present:
        return 1.0
    preserved = skel_tokens & present
    return len(preserved) / len(present)

# ── Benchmark ────────────────────────────────────────────────────────────────
print("=" * 65)
print("  E-ZERO PROTOCOL — BENCHMARK BBH (100 questions)")
print("=" * 65)

config = EZeroConfig()
ezero = EZeroFilter(config=config, lang="en")

results = []
activated_count = 0
total_tokens_in = 0
total_tokens_out = 0
total_gain = 0.0
total_fidelity = 0.0
total_time = 0.0

for i, question in enumerate(questions):
    result = ezero.filter(question)
    fid = fidelity_score(question, result["skeleton"])

    results.append({
        "id": i + 1,
        "question": question[:80] + "..." if len(question) > 80 else question,
        "skeleton": result["skeleton"][:80] + "..." if len(result["skeleton"]) > 80 else result["skeleton"],
        "tokens_in": result["tokens_in"],
        "tokens_out": result["tokens_out"],
        "rho": result["rho"],
        "gain_pct": result["gain_pct"],
        "fidelity": round(fid, 3),
        "activated": result["activated"],
        "elapsed_ms": result["elapsed_ms"],
    })

    if result["activated"]:
        activated_count += 1

    total_tokens_in  += result["tokens_in"]
    total_tokens_out += result["tokens_out"]
    total_gain       += result["gain_pct"]
    total_fidelity   += fid
    total_time       += result["elapsed_ms"]

    if (i + 1) % 10 == 0:
        print(f"  ✔ {i+1}/{len(questions)} questions traitées...")

# ── Résultats ────────────────────────────────────────────────────────────────
n = len(results)
avg_gain       = total_gain / n
avg_fidelity   = total_fidelity / n
avg_time       = total_time / n
avg_rho        = total_tokens_out / total_tokens_in
activated_pct  = activated_count / n * 100

print(f"\n{'=' * 65}")
print(f"  RÉSULTATS FINAUX — BBH")
print(f"{'=' * 65}")
print(f"  Questions testées      : {n}")
print(f"  Filtre activé          : {activated_count}/{n} ({activated_pct:.1f}%)")
print(f"  Tokens total (avant)   : {total_tokens_in}")
print(f"  Tokens total (après)   : {total_tokens_out}")
print(f"  Taux de compression ρ  : {avg_rho:.3f} ({(1-avg_rho)*100:.1f}% réduit)")
print(f"  Gain énergétique moyen : {avg_gain:.1f}%")
print(f"  Fidélité moyenne       : {avg_fidelity*100:.1f}%")
print(f"  Temps moyen par filtre : {avg_time:.3f} ms")
print(f"{'=' * 65}")

# ── Comparaison GSM8K vs BBH ─────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print(f"  COMPARAISON GSM8K vs BBH")
print(f"{'=' * 65}")
print(f"  {'Métrique':<25} {'GSM8K':>10} {'BBH':>10}")
print(f"  {'-'*45}")
print(f"  {'Gain énergétique':<25} {'48.0%':>10} {avg_gain:.1f}%{'':<4}")
print(f"  {'Fidélité':<25} {'87.6%':>10} {avg_fidelity*100:.1f}%{'':<4}")
print(f"  {'Compression':<25} {'35.4%':>10} {(1-avg_rho)*100:.1f}%{'':<4}")
print(f"  {'Vitesse':<25} {'14ms':>10} {avg_time:.1f}ms{'':<4}")
print(f"{'=' * 65}")

# ── Exemples ─────────────────────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print(f"  EXEMPLES DÉTAILLÉS (5 premières activations)")
print(f"{'=' * 65}")
shown = 0
for r in results:
    if r["activated"] and shown < 5:
        print(f"\n  [{r['id']}] QUESTION : {r['question']}")
        print(f"       SQUELETTE : {r['skeleton']}")
        print(f"       Tokens: {r['tokens_in']} → {r['tokens_out']}  |  Gain: {r['gain_pct']}%  |  Fidélité: {r['fidelity']*100:.0f}%")
        shown += 1

# ── Sauvegarde ───────────────────────────────────────────────────────────────
summary = {
    "date": "2026-04-06",
    "author": "Sawadogo Anselme",
    "dataset": "BBH Big-Bench Hard",
    "summary": {
        "questions_tested": n,
        "filter_activated_pct": round(activated_pct, 1),
        "avg_compression_rho": round(avg_rho, 3),
        "tokens_reduced_pct": round((1 - avg_rho) * 100, 1),
        "avg_energy_gain_pct": round(avg_gain, 1),
        "avg_fidelity_pct": round(avg_fidelity * 100, 1),
        "avg_filter_time_ms": round(avg_time, 3),
    },
    "comparison_gsm8k": {
        "gsm8k_gain_pct": 48.0,
        "gsm8k_fidelity_pct": 87.6,
        "gsm8k_compression_pct": 35.4,
        "bbh_gain_pct": round(avg_gain, 1),
        "bbh_fidelity_pct": round(avg_fidelity * 100, 1),
        "bbh_compression_pct": round((1 - avg_rho) * 100, 1),
    },
    "results": results
}

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ezero_bbh_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"\n✅ Résultats sauvegardés dans : ezero_bbh_results.json")
print("\nBenchmark BBH terminé !")
