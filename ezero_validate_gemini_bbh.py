"""
E-ZERO PROTOCOL — Validation Réelle de Fidélité BBH avec Gemini
================================================================
Author : Sawadogo Anselme (@sawadogoanselme-eng)
Version: 2.1 — April 2026

Ce script valide la fidélité réelle d'E-ZERO sur BBH en comparant
les réponses de Gemini sur le prompt original vs le squelette.

Usage:
    python ezero_validate_gemini_bbh.py

Important: Remplace YOUR_API_KEY_HERE par ta vraie clé API Gemini
"""

import os
import sys
import time
import json
import re

from dotenv import load_dotenv
import os
load_dotenv()

# --- Configuration API ---
import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ezero_filter import EZeroFilter, EZeroConfig

# ── Chargement BBH (20 questions) ────────────────────────────────────────────
print("Chargement du dataset BBH...")
from datasets import load_dataset

all_samples = []
subtasks = [
    ("causal_judgement", 5),
    ("date_understanding", 5),
    ("reasoning_about_colored_objects", 5),
    ("logical_deduction_five_objects", 5),
]

for task, n in subtasks:
    try:
        ds = load_dataset("lukaemon/bbh", task, split="test")
        samples = [{"input": item["input"], "target": item["target"], "task": task}
                   for item in ds.select(range(n))]
        all_samples.extend(samples)
        print(f"  ✅ {task}: {n} questions")
    except Exception as e:
        print(f"  ⚠️  {task}: {e}")

print(f"\n✅ {len(all_samples)} questions BBH chargées\n")

# ── Initialisation E-ZERO ────────────────────────────────────────────────────
config = EZeroConfig()
ezero = EZeroFilter(config=config, lang="en")

# ── Fonction d'appel Gemini ──────────────────────────────────────────────────
def ask_gemini(prompt: str, task: str) -> str:
    try:
        if "date" in task:
            system = "Answer this question. Give only the final answer, nothing else."
        elif "causal" in task:
            system = "Answer yes or no only."
        else:
            system = "Answer this reasoning question. Give only the final answer (a letter, word, or short phrase)."
        
        response = model.generate_content(f"{system}\n\n{prompt}")
        return response.text.strip()
    except Exception as e:
        return f"ERROR: {e}"

def normalize(text: str) -> str:
    """Normalise une réponse pour comparaison."""
    t = text.lower().strip()
    t = re.sub(r'[^\w\s]', '', t)
    t = ' '.join(t.split())
    return t

# ── Validation ───────────────────────────────────────────────────────────────
print("=" * 65)
print("  E-ZERO — VALIDATION RÉELLE BBH (Gemini 1.5 Flash)")
print("=" * 65)
print("  Envoi de chaque question 2 fois à Gemini...\n")

results = []
same_answer = 0
filter_activated = 0

for i, sample in enumerate(all_samples):
    question = sample["input"]
    target = sample["target"]
    task = sample["task"]

    # Appliquer E-ZERO
    filter_result = ezero.filter(question)
    skeleton = filter_result["skeleton"]
    activated = filter_result["activated"]

    if activated:
        filter_activated += 1

    print(f"  [{i+1}/{len(all_samples)}] Tâche: {task[:30]}")
    print(f"           Question : {question[:60]}...")
    print(f"           Squelette: {skeleton[:60]}...")
    print(f"           Tokens: {filter_result['tokens_in']} → {filter_result['tokens_out']} | Filtre: {'✅' if activated else '⏭'}")

    # Appel Gemini — prompt original
    response_original = ask_gemini(question, task)
    time.sleep(1.5)

    # Appel Gemini — squelette E-ZERO
    response_skeleton = ask_gemini(skeleton, task)
    time.sleep(1.5)

    # Comparaison
    norm_orig = normalize(response_original)
    norm_skel = normalize(response_skeleton)
    norm_target = normalize(target)

    answers_match = norm_orig == norm_skel
    orig_correct = norm_orig == norm_target or norm_target in norm_orig
    skel_correct = norm_skel == norm_target or norm_target in norm_skel

    if answers_match:
        same_answer += 1

    status = "✅ MATCH" if answers_match else "❌ DIFF"
    print(f"           Cible: {target} | Original: {response_original[:30]} | Squelette: {response_skeleton[:30]} | {status}\n")

    results.append({
        "id": i + 1,
        "task": task,
        "question": question[:100],
        "skeleton": skeleton[:100],
        "tokens_in": filter_result["tokens_in"],
        "tokens_out": filter_result["tokens_out"],
        "filter_activated": activated,
        "target": target,
        "response_original": response_original[:100],
        "response_skeleton": response_skeleton[:100],
        "orig_correct": orig_correct,
        "skel_correct": skel_correct,
        "answers_match": answers_match,
    })

# ── Résultats ────────────────────────────────────────────────────────────────
n = len(results)
real_fidelity = same_answer / n * 100
orig_accuracy = sum(1 for r in results if r["orig_correct"]) / n * 100
skel_accuracy = sum(1 for r in results if r["skel_correct"]) / n * 100

print("\n" + "=" * 65)
print("  RÉSULTATS — VALIDATION RÉELLE BBH")
print("=" * 65)
print(f"  Questions testées          : {n}")
print(f"  Filtre activé              : {filter_activated}/{n}")
print(f"  Précision (original)       : {orig_accuracy:.1f}%")
print(f"  Précision (squelette)      : {skel_accuracy:.1f}%")
print(f"  *** FIDÉLITÉ RÉELLE ***    : {real_fidelity:.1f}%")
print("=" * 65)

# ── Comparaison GSM8K vs BBH ─────────────────────────────────────────────────
print(f"\n  COMPARAISON FIDÉLITÉ RÉELLE")
print(f"{'=' * 65}")
print(f"  GSM8K fidélité réelle : 100%  (20 questions)")
print(f"  BBH   fidélité réelle : {real_fidelity:.1f}%  ({n} questions)")
print(f"  LLMLingua             : ~98%  (exact match)")
print(f"{'=' * 65}")

# ── Interprétation ───────────────────────────────────────────────────────────
print("\n  INTERPRÉTATION :")
if real_fidelity >= 90:
    print(f"  ✅ EXCELLENTE fidélité ({real_fidelity:.1f}%) — E-ZERO généralise")
    print(f"     bien au-delà des maths. La Lacune #2 est RÉSOLUE.")
elif real_fidelity >= 75:
    print(f"  ⚠️  BONNE fidélité ({real_fidelity:.1f}%) — E-ZERO fonctionne sur BBH")
    print(f"     mais des améliorations sont encore possibles.")
else:
    print(f"  ❌ Fidélité insuffisante ({real_fidelity:.1f}%) sur BBH.")
    print(f"     Le filtre perd trop d'information logique.")

# ── Sauvegarde ───────────────────────────────────────────────────────────────
output = {
    "date": "2026-04-06",
    "author": "Sawadogo Anselme",
    "model_used": "gemini-1.5-flash",
    "dataset": "BBH Big-Bench Hard",
    "summary": {
        "questions_tested": n,
        "filter_activated": filter_activated,
        "accuracy_original_pct": round(orig_accuracy, 1),
        "accuracy_skeleton_pct": round(skel_accuracy, 1),
        "real_fidelity_pct": round(real_fidelity, 1),
    },
    "comparison": {
        "gsm8k_real_fidelity": 100.0,
        "bbh_real_fidelity": round(real_fidelity, 1),
        "llmlingua_fidelity": 98.0,
    },
    "results": results
}

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ezero_bbh_real_fidelity.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n✅ Résultats sauvegardés dans : ezero_bbh_real_fidelity.json")
print("\nValidation BBH terminée !")
