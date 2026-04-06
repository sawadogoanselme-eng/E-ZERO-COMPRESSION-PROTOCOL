"""
E-ZERO PROTOCOL — Validation Réelle de Fidélité avec Gemini
=============================================================
Author : Sawadogo Anselme (@sawadogoanselme-eng)
Version: 2.0 — April 2026

Ce script résout la Lacune #1 : prouver que le squelette E-ZERO
produit la MÊME réponse que le prompt original sur un vrai LLM.

Méthode :
  1. Prendre 20 questions GSM8K
  2. Envoyer le prompt original à Gemini → réponse A
  3. Envoyer le squelette E-ZERO à Gemini → réponse B
  4. Comparer A et B (exact match sur le nombre final)

Usage:
    python ezero_validate_gemini.py
    
Important: Remplace YOUR_API_KEY_HERE par ta vraie clé API Gemini
"""

import os
import sys
import time
import json
import re

# ── Clé API Gemini ───────────────────────────────────────────────────────────
# IMPORTANT: Remplace cette ligne par ta vraie clé API
GEMINI_API_KEY = "AIzaSyBU7aHaZKV_WT7YkOq_uien9B8ne75Mhkw"

if GEMINI_API_KEY == "YOUR_API_KEY_HERE":
    print("❌ ERREUR: Tu dois remplacer YOUR_API_KEY_HERE par ta vraie clé API Gemini")
    print("   Ouvre ce fichier avec le Bloc-notes et remplace YOUR_API_KEY_HERE")
    exit(1)

# ── Import des bibliothèques ─────────────────────────────────────────────────
import google.generativeai as genai
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ezero_filter import EZeroFilter, EZeroConfig

# ── Chargement GSM8K (20 questions) ─────────────────────────────────────────
print("Chargement du dataset GSM8K...")
from datasets import load_dataset
dataset = load_dataset("openai/gsm8k", "main", split="test")
# On prend 20 questions pour limiter les appels API
samples = [item for item in dataset.select(range(20))]
print(f"✅ {len(samples)} questions chargées\n")

# ── Initialisation du filtre E-ZERO ─────────────────────────────────────────
config = EZeroConfig()
ezero = EZeroFilter(config=config, lang="en")

# ── Fonction pour extraire le nombre final d'une réponse ────────────────────
def extract_final_number(text: str):
    """
    Extrait le nombre final d'une réponse GSM8K.
    GSM8K utilise #### comme marqueur de réponse finale.
    """
    # Cherche #### suivi d'un nombre
    match = re.search(r'####\s*(-?\d+(?:,\d+)?(?:\.\d+)?)', text)
    if match:
        num = match.group(1).replace(',', '')
        return float(num)
    
    # Sinon cherche le dernier nombre dans le texte
    numbers = re.findall(r'-?\d+(?:,\d+)?(?:\.\d+)?', text)
    if numbers:
        return float(numbers[-1].replace(',', ''))
    return None

# ── Fonction d'appel Gemini ──────────────────────────────────────────────────
def ask_gemini(prompt: str) -> str:
    """Envoie un prompt à Gemini et retourne la réponse."""
    try:
        system = "Solve this math problem step by step. End your answer with #### followed by the final number only."
        response = model.generate_content(f"{system}\n\n{prompt}")
        return response.text
    except Exception as e:
        return f"ERROR: {e}"

# ── Validation principale ────────────────────────────────────────────────────
print("=" * 65)
print("  E-ZERO — VALIDATION RÉELLE DE FIDÉLITÉ (Gemini)")
print("=" * 65)
print("  Envoi de chaque question 2 fois à Gemini...")
print("  (prompt original + squelette E-ZERO)\n")

results = []
correct_original = 0
correct_skeleton = 0
same_answer = 0
filter_activated = 0

for i, sample in enumerate(samples):
    question = sample["question"]
    ground_truth_text = sample["answer"]
    
    # Extraire la vraie réponse
    ground_truth = extract_final_number(ground_truth_text)
    
    # Appliquer le filtre E-ZERO
    filter_result = ezero.filter(question)
    skeleton = filter_result["skeleton"]
    activated = filter_result["activated"]
    
    if activated:
        filter_activated += 1
    
    print(f"  [{i+1}/20] Question : {question[:60]}...")
    print(f"           Squelette : {skeleton[:60]}...")
    print(f"           Tokens: {filter_result['tokens_in']} → {filter_result['tokens_out']} | Filtre: {'✅' if activated else '⏭'}")
    
    # Appel 1 : prompt original
    response_original = ask_gemini(question)
    answer_original = extract_final_number(response_original)
    time.sleep(1)  # Éviter le rate limit
    
    # Appel 2 : squelette E-ZERO
    response_skeleton = ask_gemini(skeleton)
    answer_skeleton = extract_final_number(response_skeleton)
    time.sleep(1)  # Éviter le rate limit
    
    # Comparaisons
    orig_correct = (answer_original is not None and 
                   ground_truth is not None and 
                   abs(answer_original - ground_truth) < 0.01)
    
    skel_correct = (answer_skeleton is not None and 
                   ground_truth is not None and 
                   abs(answer_skeleton - ground_truth) < 0.01)
    
    answers_match = (answer_original is not None and 
                    answer_skeleton is not None and 
                    abs(answer_original - answer_skeleton) < 0.01)
    
    if orig_correct:
        correct_original += 1
    if skel_correct:
        correct_skeleton += 1
    if answers_match:
        same_answer += 1
    
    status = "✅ MATCH" if answers_match else "❌ DIFF"
    print(f"           Réponse correcte: {ground_truth} | Original: {answer_original} | Squelette: {answer_skeleton} | {status}\n")
    
    results.append({
        "id": i + 1,
        "question": question[:100],
        "skeleton": skeleton[:100],
        "tokens_in": filter_result["tokens_in"],
        "tokens_out": filter_result["tokens_out"],
        "filter_activated": activated,
        "ground_truth": ground_truth,
        "answer_original": answer_original,
        "answer_skeleton": answer_skeleton,
        "orig_correct": orig_correct,
        "skel_correct": skel_correct,
        "answers_match": answers_match,
    })

# ── Résultats finaux ─────────────────────────────────────────────────────────
n = len(results)
fidelity_real = same_answer / n * 100
accuracy_original = correct_original / n * 100
accuracy_skeleton = correct_skeleton / n * 100

print("\n" + "=" * 65)
print("  RÉSULTATS DE VALIDATION RÉELLE")
print("=" * 65)
print(f"  Questions testées          : {n}")
print(f"  Filtre activé              : {filter_activated}/{n}")
print(f"  Précision (prompt original): {accuracy_original:.1f}%")
print(f"  Précision (squelette)      : {accuracy_skeleton:.1f}%")
print(f"  *** FIDÉLITÉ RÉELLE ***    : {fidelity_real:.1f}%")
print(f"      (même réponse original vs squelette)")
print("=" * 65)

# ── Interprétation ───────────────────────────────────────────────────────────
print("\n  INTERPRÉTATION :")
if fidelity_real >= 90:
    print(f"  ✅ EXCELLENTE fidélité ({fidelity_real:.1f}%) — E-ZERO préserve")
    print(f"     le sens mathématique dans {fidelity_real:.1f}% des cas.")
    print(f"     La Lacune #1 est RÉSOLUE.")
elif fidelity_real >= 75:
    print(f"  ⚠️  BONNE fidélité ({fidelity_real:.1f}%) — E-ZERO fonctionne bien")
    print(f"     mais quelques améliorations sont nécessaires.")
else:
    print(f"  ❌ Fidélité insuffisante ({fidelity_real:.1f}%) — le squelette")
    print(f"     perd trop d'information mathématique.")

print(f"\n  Comparaison avec LLMLingua :")
print(f"  LLMLingua fidélité réelle  : ~98% (exact match GSM8K)")
print(f"  E-ZERO fidélité réelle     : {fidelity_real:.1f}%")

# ── Sauvegarde ───────────────────────────────────────────────────────────────
output = {
    "date": "2026-04-05",
    "author": "Sawadogo Anselme",
    "model_used": "gemini-1.5-flash",
    "dataset": "GSM8K (20 questions)",
    "summary": {
        "questions_tested": n,
        "filter_activated": filter_activated,
        "accuracy_original_pct": round(accuracy_original, 1),
        "accuracy_skeleton_pct": round(accuracy_skeleton, 1),
        "real_fidelity_pct": round(fidelity_real, 1),
    },
    "results": results
}

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ezero_real_fidelity.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n✅ Résultats sauvegardés dans : ezero_real_fidelity.json")
print("\nValidation terminée !")
