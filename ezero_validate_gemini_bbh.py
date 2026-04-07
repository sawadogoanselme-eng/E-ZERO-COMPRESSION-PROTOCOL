import os
import sys
import time
import json
from dotenv import load_dotenv
from google import genai

# 1. Configuration
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

# Import E-ZERO
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ezero_filter import EZeroFilter, EZeroConfig

# 2. Dataset BBH
from datasets import load_dataset
all_samples = []
subtasks = [("causal_judgement", 5), ("date_understanding", 5), 
            ("reasoning_about_colored_objects", 5), ("logical_deduction_five_objects", 5)]

for task, n in subtasks:
    ds = load_dataset("lukaemon/bbh", task, split="test")
    samples = [{"input": item["input"], "target": item["target"], "task": task} for item in ds.select(range(n))]
    all_samples.extend(samples)

# 3. Initialisation E-ZERO
ezero = EZeroFilter(config=EZeroConfig(), lang="en")

# 4. Fonctions
def ask_gemini(prompt, task):
    if "date" in task:
        system = "Answer this question. Give only the final answer."
    elif "causal" in task:
        system = "Answer yes or no only."
    else:
        system = "Answer this reasoning question. Give only the final answer."
    
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=f"{system}\n\n{prompt}"
        )
        return response.text.strip()
    except Exception as e:
        return f"ERROR: {e}"

def normalize(text):
    return text.lower().strip()

# 5. Lancement de la Validation
print("\n--- DÉMARRAGE DE LA VALIDATION E-ZERO (VERSION 2026) ---\n")
results = []
same_answer = 0

for i, sample in enumerate(all_samples):
    q, target, task = sample["input"], sample["target"], sample["task"]
    
    # Filtrage
    f_res = ezero.filter(q)
    skel = f_res["skeleton"]
    
    # Appels
    res_orig = ask_gemini(q, task)
    time.sleep(1)
    res_skel = ask_gemini(skel, task)
    time.sleep(1)
    
    # Comparaison
    match = normalize(res_orig) == normalize(res_skel)
    if match: same_answer += 1
    
    print(f"[{i+1}/20] {task[:15]}... | Match: {'✅' if match else '❌'}")
    
    results.append({"id": i+1, "match": match, "orig": res_orig, "skel": res_skel})

# 6. Score Final
fidelity = (same_answer / len(all_samples)) * 100
print(f"\n======================================")
print(f"FIDÉLITÉ RÉELLE E-ZERO : {fidelity:.1f}%")
print(f"======================================\n")

# Sauvegarde
with open("ezero_bbh_results.json", "w") as f:
    json.dump({"fidelity": fidelity, "details": results}, f, indent=2)