import re
import time
import json
import math
from collections import Counter

# ── RÉFÉRENTIELS BIOLOGIQUES ─────────────────────────────────────────────────
SACRED_WORDS = {
    "not", "if", "then", "false", "true", "each", "half", "except", 
    "how", "what", "calculate", "total", "many", "left", "result",
    "before", "after", "between", "first", "last", "only",
    "address", "uint256", "external", "public", "require", "contract" # Ajouts Blockchain
}

# ── CONFIGURATION V5.0 ───────────────────────────────────────────────────────
class EZeroConfig:
    def __init__(self, **kwargs):
        self.n_min = kwargs.get('n_min', 5)
        self.rho_target = kwargs.get('rho_target', 0.3)
        self.memory_path = kwargs.get('memory_path', 'ezero_memory.json')

# ── ORGANISME E-ZERO V5 ──────────────────────────────────────────────────────
class EZeroFilter:
    def __init__(self, config=None):
        self.config = config or EZeroConfig()
        self.m_logic = SACRED_WORDS
        self.m_spec  = {"iron", "copper", "aluminum", "gold", "waste", "recycling", "battery"}
        self.synaptic_weights = {} 
        self.immune_memory = set()
        self.load_memories()

    # ── GESTION DE LA MÉMOIRE (Mise à jour v5.0) ────────────────────────────
    def load_memories(self):
        """Charge le cerveau de 556 Ko au démarrage."""
        try:
            with open(self.config.memory_path, "r", encoding='utf-8') as f:
                data = json.load(f)
                self.synaptic_weights = data.get("weights", {})
                self.immune_memory = set(data.get("immune", []))
                print(f"🧠 Cerveau chargé : {len(self.synaptic_weights)} synapses prêtes.")
        except (FileNotFoundError, json.JSONDecodeError):
            print("⚠️ Aucun cerveau trouvé. Démarrage en mode apprentissage (froid).")

    def save_memories(self):
        """Sauvegarde les synapses pour la persistance."""
        with open(self.config.memory_path, "w", encoding='utf-8') as f:
            json.dump({
                "weights": self.synaptic_weights,
                "immune": list(self.immune_memory)
            }, f, indent=2)

    # ── MOTEURS DE DÉTECTION ─────────────────────────────────────────────────
    def _is_numeric(self, token):
        """Membrane Métabolique (Chiffres, Adresses Hexa, Unités)."""
        # Capture les adresses 0x... et les versions 0.8.20
        return bool(re.search(r'\d|0x[a-fA-F0-9]{40}', token))

    def _get_dynamic_weight(self, token):
        clean = token.lower().strip(",.?!:;()")
        return self.synaptic_weights.get(clean, 1.0)

    # ── FILTRAGE MEMBRANAIRE ─────────────────────────────────────────────────
    def filter(self, prompt: str, mode="general") -> dict:
        t_start = time.perf_counter()
        tokens = prompt.split()
        n = len(tokens)

        if n < self.config.n_min:
            return {"skeleton": prompt, "activated": False, "gain_pct": 0.0}

        keep = []
        for t in tokens:
            clean = t.lower().strip(",.?!:;()\"'")
            
            # CONDITIONS DE SURVIE (Théorème E-ZERO)
            is_sacred = clean in self.m_logic or clean in self.m_spec
            is_immune = clean in self.immune_memory
            is_strong = self._get_dynamic_weight(t) > 1.3
            
            # Si c'est un chiffre, une adresse Blockchain ou un mot sacré, on garde.
            if (self._is_numeric(t) or is_sacred or is_immune or is_strong):
                keep.append(t)

        skeleton = " ".join(keep)
        rho = len(keep) / n if n > 0 else 1.0
        # Calcul du gain basé sur ton théorème G = alpha * n^2 * (1 - rho^2)
        gain_pct = (1 - (rho ** 2)) * 100

        return {
            "skeleton": skeleton,
            "tokens_in": n,
            "tokens_out": len(keep),
            "gain_pct": round(gain_pct, 1),
            "ms": round((time.perf_counter() - t_start) * 1000, 3),
            "plasticity": {
                "synapses": len(self.synaptic_weights),
                "antibodies": len(self.immune_memory)
            }
        }