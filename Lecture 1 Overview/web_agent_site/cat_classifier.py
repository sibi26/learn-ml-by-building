from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import torch

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError as e:
    raise ImportError("transformers is required for CatClassifier. Please install it.") from e

# PEFT is optional; if not installed, we just skip LoRA loading
try:
    from peft import PeftModel  # type: ignore
    _PEFT_AVAILABLE = True
except Exception:
    PeftModel = None  # type: ignore
    _PEFT_AVAILABLE = False


# Exported category config for front-end styling
CAT_CATEGORIES: Dict[str, Dict[str, str]] = {
    'NAP_SURFACE': {
        'category': 'Nap Surface',
        'emoji': '😴',
        'color': '#6c5ce7'
    },
    'HUNT_PLAY': {
        'category': 'Hunt/Play',
        'emoji': '🧶',
        'color': '#00b894'
    },
    'TERRITORY': {
        'category': 'Territory',
        'emoji': '🚩',
        'color': '#e17055'
    },
    'GROOMING': {
        'category': 'Grooming',
        'emoji': '🛁',
        'color': '#0984e3'
    },
    'CONSUMPTION': {
        'category': 'Eating/Drinking',
        'emoji': '🍣',
        'color': '#fdcb6e'
    },
    'DANGER': {
        'category': 'Danger',
        'emoji': '⚠️',
        'color': '#d63031'
    },
    'IRRELEVANT': {
        'category': 'Irrelevant',
        'emoji': '😼',
        'color': '#636e72'
    },
}


class CatClassifier:
    """Lightweight classifier + chat on top of a small Gemma model.

    - Model path uses a relative default that goes up one level to `models/`.
    - Base model set to google/gemma-270m as requested.
    - If a LoRA folder exists at model_path and PEFT is available, it will be applied.
    """

    def __init__(self, model_path: str = "../models/gemma-cat-lora"):
        # This goes UP one level from web_agent_site/ to find models/
        self.model_path = Path(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Token mapping for classification (space-prefixed for tokenization stability)
        self.cat_tokens: Dict[str, str] = {
            'NAP_SURFACE': ' napping',
            'HUNT_PLAY': ' hunting',
            'TERRITORY': ' territory',
            'GROOMING': ' grooming',
            'CONSUMPTION': ' eating',
            'DANGER': ' danger',
            'IRRELEVANT': ' boring',
        }

        self.model = None
        self.tokenizer = None
        self.category_token_ids: Dict[str, int] = {}
        self._loaded = False

    def load_model(self) -> None:
        if self._loaded:
            return
        base_model = "google/gemma-270m"  # Match the notebook
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

        # Load base
        base_model_obj = AutoModelForCausalLM.from_pretrained(base_model)

        # If LoRA exists and PEFT available, apply it; otherwise use base
        if self.model_path.exists() and _PEFT_AVAILABLE:
            self.model = PeftModel.from_pretrained(base_model_obj, str(self.model_path))
        else:
            self.model = base_model_obj

        self.model.to(self.device)
        self.model.eval()

        # Pre-compute token IDs for efficiency
        for cat, token in self.cat_tokens.items():
            ids = self.tokenizer.encode(token, add_special_tokens=False)
            if ids:
                self.category_token_ids[cat] = ids[0]
        self._loaded = True

    @torch.no_grad()
    def classify_product(self, title: str, description: str) -> Dict[str, Any]:
        """Classify a product into the cat categories using next-token scores.
        Heuristic: score predefined tokens given the product title+desc.
        Returns dict with keys: category, emoji, color, confidence, description
        """
        self.load_model()
        assert self.model is not None and self.tokenizer is not None

        text = (title or "").strip()
        if description:
            text += "\n" + description.strip()
        if not text:
            # Fallback to irrelevant if we have no content
            best_key = 'IRRELEVANT'
            return self._build_result(best_key, 0.2, title)

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        # Take the last hidden token logits
        logits = outputs.logits[:, -1, :].squeeze(0)  # [vocab]

        # Gather scores for our category tokens
        scores = []
        keys = []
        for key, tok_id in self.category_token_ids.items():
            if tok_id < logits.shape[-1]:
                scores.append(logits[tok_id].item())
                keys.append(key)
        if not scores:
            best_key = 'IRRELEVANT'
            return self._build_result(best_key, 0.2, title)

        # Softmax over our scores to form a pseudo-confidence
        t = torch.tensor(scores)
        probs = torch.softmax(t, dim=0)
        best_idx = int(torch.argmax(probs).item())
        best_key = keys[best_idx]
        confidence = float(probs[best_idx].item())
        return self._build_result(best_key, confidence, title)

    def _build_result(self, key: str, confidence: float, title: str) -> Dict[str, Any]:
        meta = CAT_CATEGORIES.get(key, CAT_CATEGORIES['IRRELEVANT'])
        # Simple descriptive string
        desc_map = {
            'NAP_SURFACE': 'Looks comfy for a long nap.',
            'HUNT_PLAY': 'Seems great for chasing and playtime.',
            'TERRITORY': 'Useful for marking and owning the space.',
            'GROOMING': 'Good for keeping fur pristine.',
            'CONSUMPTION': 'Smells like treats or food-related fun.',
            'DANGER': 'Hmm, could be risky. Approach with caution.',
            'IRRELEVANT': "Doesn't catch my feline interest.",
        }
        return {
            'category': meta['category'],
            'emoji': meta['emoji'],
            'color': meta['color'],
            'confidence': confidence,
            'description': desc_map.get(self._invert_key(meta['category']), desc_map['IRRELEVANT'])
        }

    def _invert_key(self, category_name: str) -> str:
        for k, v in CAT_CATEGORIES.items():
            if v['category'] == category_name:
                return k
        return 'IRRELEVANT'

    def chat_about_product(self, product_name: str, question: str) -> str:
        """Lightweight, templated chat response.
        For full generation, you could prompt the model; here we keep it fast.
        """
        product_name = product_name or "this product"
        question = (question or "").strip()
        if not question:
            return f"Meow! I love {product_name}. What would you like to know?"
        # Heuristic hints
        q_lower = question.lower()
        if any(w in q_lower for w in ["safe", "danger", "hazard", "toxic"]):
            return f"I sense potential danger around {product_name}. Please check materials and safety notes before buying."
        if any(w in q_lower for w in ["eat", "food", "treat", "bowl", "snack"]):
            return f"If it involves snacks, I’m in! {product_name} looks suitable for eating-related fun."
        if any(w in q_lower for w in ["sleep", "nap", "bed", "comfy", "soft"]):
            return f"Mmm… {product_name} seems purrfect for a cozy nap."
        if any(w in q_lower for w in ["play", "toy", "chase", "hunt"]):
            return f"I’d love to pounce on {product_name}! Looks great for play and hunting instincts."
        return f"I’m curious about {product_name}. From a cat’s view, it could be fun or comfy—what matters most to you?"


# Simple singleton to avoid re-loading
_classifier_singleton: CatClassifier | None = None

def get_cat_classifier() -> CatClassifier:
    global _classifier_singleton
    if _classifier_singleton is None:
        _classifier_singleton = CatClassifier()
        # Lazy model load; optional: force load here
        try:
            _classifier_singleton.load_model()
        except Exception:
            # Keep usable even if model can't load; classify falls back to IRRELEVANT gracefully
            pass
    return _classifier_singleton
