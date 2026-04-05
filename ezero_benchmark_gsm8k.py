"""
E-ZERO PROTOCOL — GSM8K Benchmark
===================================
Author : Sawadogo Anselme (@sawadogoanselme-eng)
Version: 1.0 — April 2026

Ce script teste le filtre E-ZERO sur 100 questions du dataset GSM8K.
Il mesure :
  1. Le gain énergétique (tokens éliminés)
  2. La fidélité (mots-clés mathématiques préservés)
  3. La vitesse du filtre

Usage:
    python ezero_benchmark_gsm8k.py
"""

import math
import time
import json

# ── Chargement du dataset GSM8K ─────────────────────────────────────────────
print("Chargement du dataset GSM8K...")
try:
    from datasets import load_dataset
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    questions = [item["question"] for item in dataset.select(range(100))]
    print(f"✅ {len(questions)} questions chargées depuis GSM8K\n")
except Exception as e:
    print(f"❌ Erreur chargement dataset: {e}")
    exit(1)

# ── Import du filtre E-ZERO ──────────────────────────────────────────────────
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from ezero_filter import EZeroFilter, EZeroConfig
    print("✅ Filtre E-ZERO importé avec succès\n")
except ImportError:
    print("❌ ezero_filter.py introuvable.")
    print("   Assure-toi que ezero_filter.py est sur le Bureau aussi.")
    exit(1)

# ── Mots-clés mathématiques à préserver (test de fidélité) ──────────────────
MATH_KEYWORDS = {
    "how", "many", "much", "total", "each", "per",
    "more", "less", "left", "remaining", "altogether",
    "number", "times", "half", "double", "triple",
    "cost", "price", "paid", "spend", "earn", "buy", "sell",
    "days", "hours", "minutes", "weeks", "months",
    "if", "then", "after", "before", "between",
}

def fidelity_score(original: str, skeleton: str) -> float:
    """
    Mesure la fidélité : fraction des mots-clés mathématiques
    du prompt original qui sont préservés dans le squelette.
    """
    orig_tokens = set(original.lower().split())
    skel_tokens = set(skeleton.lower().split())
    
    # Mots-clés présents dans l'original
    present = orig_tokens & MATH_KEYWORDS
    if not present:
        return 1.0
    
    # Mots-clés préservés dans le squelette
    preserved = skel_tokens & present
    return len(preserved) / len(present)

# ── Lancement du benchmark ───────────────────────────────────────────────────
print("=" * 65)
print("  E-ZERO PROTOCOL — BENCHMARK GSM8K (100 questions)")
print("=" * 65)

config = EZeroConfig()
config.rho_target = 0.4
config.gamma = 0.25
config.n_min = 15

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
    
    total_tokens_in += result["tokens_in"]
    total_tokens_out += result["tokens_out"]
    total_gain += result["gain_pct"]
    total_fidelity += fid
    total_time += result["elapsed_ms"]
    
    # Afficher chaque 10 questions
    if (i + 1) % 10 == 0:
        print(f"  ✔ {i+1}/100 questions traitées...")

# ── Résultats agrégés ────────────────────────────────────────────────────────
n = len(results)
activated_pct = activated_count / n * 100
avg_gain = total_gain / n
avg_fidelity = total_fidelity / n
avg_time = total_time / n
avg_rho = total_tokens_out / total_tokens_in

print("\n")
print("=" * 65)
print("  RÉSULTATS FINAUX")
print("=" * 65)
print(f"  Questions testées      : {n}")
print(f"  Filtre activé          : {activated_count}/{n} ({activated_pct:.1f}%)")
print(f"  Tokens total (avant)   : {total_tokens_in}")
print(f"  Tokens total (après)   : {total_tokens_out}")
print(f"  Taux de compression ρ  : {avg_rho:.3f} ({(1-avg_rho)*100:.1f}% réduit)")
print(f"  Gain énergétique moyen : {avg_gain:.1f}%")
print(f"  Fidélité moyenne       : {avg_fidelity*100:.1f}%")
print(f"  Temps moyen par filtre : {avg_time:.3f} ms")
print(f"  Temps total            : {total_time:.1f} ms")
print("=" * 65)

# ── Comparaison avec LLMLingua ───────────────────────────────────────────────
print("\n")
print("=" * 65)
print("  COMPARAISON E-ZERO vs LLMLingua (GSM8K)")
print("=" * 65)
print(f"  {'Métrique':<30} {'E-ZERO':<15} {'LLMLingua'}")
print(f"  {'-'*60}")
print(f"  {'Gain énergétique moyen':<30} {avg_gain:.1f}%{'':<9} ~82% (5x compression)")
print(f"  {'Fidélité (mots-clés)':<30} {avg_fidelity*100:.1f}%{'':<9} ~98% (exact match)")
print(f"  {'Dépendance modèle':<30} {'Aucune':<15} LLaMA-7B requis")
print(f"  {'Complexité':<30} {'O(n log n)':<15} O(n²) via LLM")
print(f"  {'Théorème de gain':<30} {'✅ Prouvé':<15} ❌ Non formalisé")
print(f"  {'Temps moyen par prompt':<30} {avg_time:.1f} ms{'':<8} ~500ms (LLM requis)")
print("=" * 65)

# ── Exemples détaillés (10 premiers activés) ────────────────────────────────
print("\n")
print("=" * 65)
print("  EXEMPLES DÉTAILLÉS (10 premières activations)")
print("=" * 65)

shown = 0
for r in results:
    if r["activated"] and shown < 10:
        print(f"\n  [{r['id']}] QUESTION : {r['question']}")
        print(f"       SQUELETTE : {r['skeleton']}")
        print(f"       Tokens: {r['tokens_in']} → {r['tokens_out']}  |  Gain: {r['gain_pct']}%  |  Fidélité: {r['fidelity']*100:.0f}%")
        shown += 1

# ── Sauvegarde JSON ──────────────────────────────────────────────────────────
output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ezero_gsm8k_results.json")
summary = {
    "date": "2026-04-05",
    "author": "Sawadogo Anselme",
    "dataset": "GSM8K (100 questions test)",
    "summary": {
        "questions_tested": n,
        "filter_activated_pct": round(activated_pct, 1),
        "avg_compression_rho": round(avg_rho, 3),
        "tokens_reduced_pct": round((1 - avg_rho) * 100, 1),
        "avg_energy_gain_pct": round(avg_gain, 1),
        "avg_fidelity_pct": round(avg_fidelity * 100, 1),
        "avg_filter_time_ms": round(avg_time, 3),
    },
    "results": results
}

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"\n✅ Résultats sauvegardés dans : ezero_gsm8k_results.json")
print("\nBenchmark terminé. Tu peux copier ces résultats dans ton README GitHub.")
