#!/usr/bin/env python3
"""Builds LLM chat prompts for conversation and explanation datasets.
Outputs prompt files into the same directory as this script.
- Reads: ../data/processed/cat_products.json (relative to this file)
- Writes: conversation_prompt_*.txt and explanation_prompt_*.txt
"""
from pathlib import Path
import json
import math
import argparse
from typing import Optional

# ----- Config -----
DATA_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = DATA_DIR / "processed"
PRODUCTS_PATH = PROCESSED_DIR / "cat_products.json"

CONV_SYSTEM = (
    "You are an expert assistant generating compact, high-quality conversation\n"
    "examples about what a house cat would think of household products.\n"
    "Rules:\n"
    "- Output must be valid JSON only (no prose outside JSON).\n"
    "- Generate one prompt (user query) and one completion (assistant reply) per item.\n"
    "- Completion: 1–2 sentences, first-person cat inner monologue, playful, concise.\n"
    "- Do not repeat/echo the prompt in the completion.\n"
    "- Be concrete and specific to the item; avoid filler.\n"
    "- Keep consistency with the provided category.\n"
    "- Categories: NAP_SURFACE, HUNT_PLAY, TERRITORY, GROOMING, CONSUMPTION, DANGER, IRRELEVANT.\n"
)

CONV_FEWSHOT = [
    {
        "product_name": "Velvet Throw Pillow",
        "category": "NAP_SURFACE",
        "conversation": {
            "prompt": "Human: I bought a Velvet Throw Pillow. What will my cat think?\nAssistant:",
            "completion": " Ooh, a plush cloud for my royal loaf—don’t mind me while I shed a little love on it."
        }
    },
    {
        "product_name": "Laser Pointer",
        "category": "HUNT_PLAY",
        "conversation": {
            "prompt": "Human: How would a cat react to a Laser Pointer?\nAssistant:",
            "completion": " The red prey returns—must stalk, pounce, and pretend I meant to miss."
        }
    }
]

EXPL_SYSTEM = (
    "You are an expert assistant generating brief, concrete explanations of why\n"
    "a house cat would categorize a product into a given category.\n"
    "Rules:\n"
    "- Output must be valid JSON only (no prose outside JSON).\n"
    "- One-sentence explanation per item.\n"
    "- Reference 1–2 concrete cues that fit the category (e.g., soft/warm/elevated for NAP_SURFACE;\n"
    "  motion/chase for HUNT_PLAY; height/perches/ownership for TERRITORY; cleaning/grooming/odor for GROOMING;\n"
    "  edible/food/drink/feeding for CONSUMPTION; loud/hot/sharp/toxic/wet for DANGER; unrelated/office/technical for IRRELEVANT).\n"
    "- Do not repeat the product name verbatim unless needed; keep it concise and specific.\n"
    "- Categories: NAP_SURFACE, HUNT_PLAY, TERRITORY, GROOMING, CONSUMPTION, DANGER, IRRELEVANT.\n"
)

EXPL_FEWSHOT = [
    {
        "product_name": "Velvet Throw Pillow",
        "category": "NAP_SURFACE",
        "explanation": {
            "prompt": "Explain why a cat would categorize Velvet Throw Pillow as nap surface:",
            "completion": " Its soft, warm cushion offers a comfy, elevated loafing spot ideal for long naps."
        }
    },
    {
        "product_name": "Laser Pointer",
        "category": "HUNT_PLAY",
        "explanation": {
            "prompt": "Explain why a cat would categorize Laser Pointer as hunt play:",
            "completion": " It triggers chase instincts with fast, darting motion that mimics elusive prey."
        }
    }
]

CONV_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["product_name", "category", "conversation"],
        "properties": {
            "product_name": {"type": "string"},
            "category": {"type": "string"},
            "conversation": {
                "type": "object",
                "required": ["prompt", "completion"],
                "properties": {
                    "prompt": {"type": "string"},
                    "completion": {"type": "string"}
                }
            }
        }
    }
}

EXPL_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["product_name", "category", "explanation"],
        "properties": {
            "product_name": {"type": "string"},
            "category": {"type": "string"},
            "explanation": {
                "type": "object",
                "required": ["prompt", "completion"],
                "properties": {
                    "prompt": {"type": "string"},
                    "completion": {"type": "string"}
                }
            }
        }
    }
}

CONV_INSTRUCTIONS = (
    "Task: For each item, produce one conversation example (prompt + completion) that follows the rules and styles above.\n"
    "Output schema (return JSON ONLY):\n" + json.dumps(CONV_SCHEMA, indent=2) + "\n"
    "Items (JSON array) — use exactly these and generate your own suitable prompt texts:"
)

EXPL_INSTRUCTIONS = (
    "Task: For each item, produce an explanation (prompt + one-sentence completion) that follows the rules and styles above.\n"
    "Output schema (return JSON ONLY):\n" + json.dumps(EXPL_SCHEMA, indent=2) + "\n"
    "Items (JSON array) — use exactly these and generate your own suitable prompt texts:"
)


def load_items(limit: Optional[int] = None):
    data = json.loads(PRODUCTS_PATH.read_text())
    items = [{
        "product_name": d.get("name", "").strip(),
        "category": d.get("cat_category", "").strip()
    } for d in data if d.get("name") and d.get("cat_category")]
    if limit is not None:
        items = items[:limit]
    return items


def write_batch(files_prefix: str, header: str, items_batch: list, suffix: str):
    # Compose the full prompt as a single text block
    body = header + "\n" + json.dumps(items_batch, ensure_ascii=False, indent=2)
    out_path = DATA_DIR / f"{files_prefix}{suffix}.txt"
    out_path.write_text(body)
    return out_path


def clean_old_files():
    removed = 0
    for p in DATA_DIR.iterdir():
        if not p.is_file():
            continue
        name = p.name
        if (
            name == "conversation_prompt.txt"
            or name == "explanation_prompt.txt"
            or name.startswith("conversation_prompt_")
            or name.startswith("explanation_prompt_")
        ):
            p.unlink()
            removed += 1
    return removed


def main():
    parser = argparse.ArgumentParser(description="Build LLM chat prompts from cat_products.json")
    parser.add_argument("--batch-size", type=int, default=50, help="Items per prompt file; if >= items count, a single file is produced")
    parser.add_argument("--limit", type=int, default=None, help="Limit total items (e.g., 500)")
    parser.add_argument("--clean", action="store_true", help="Remove previous prompt files before generating")
    args = parser.parse_args()

    items = load_items(limit=args.limit)
    n = len(items)
    if n == 0:
        raise SystemExit(f"No items found in {PRODUCTS_PATH}")

    if args.clean:
        removed = clean_old_files()
        print(f"Removed {removed} existing prompt file(s)")

    batch_size = max(1, args.batch_size)
    if batch_size >= n:
        # Single file mode
        conv_header = (
            "System:\n" + CONV_SYSTEM + "\n\n" +
            "Few-shot examples:\n" + json.dumps(CONV_FEWSHOT, indent=2) + "\n\n" +
            "User:\n" + CONV_INSTRUCTIONS + "\n"
        )
        expl_header = (
            "System:\n" + EXPL_SYSTEM + "\n\n" +
            "Few-shot examples:\n" + json.dumps(EXPL_FEWSHOT, indent=2) + "\n\n" +
            "User:\n" + EXPL_INSTRUCTIONS + "\n"
        )
        conv_path = write_batch("conversation_prompt", conv_header, items, "")
        expl_path = write_batch("explanation_prompt", expl_header, items, "")
        print(f"Wrote: {conv_path.name} and {expl_path.name}  (items 1-{n} of {n})")
        return

    # Batched mode
    num_batches = math.ceil(n / batch_size)
    for bi in range(num_batches):
        start = bi * batch_size
        end = min((bi + 1) * batch_size, n)
        batch_items = items[start:end]
        tag = f"_{bi+1:03d}" if num_batches > 1 else ""

        conv_header = (
            "System:\n" + CONV_SYSTEM + "\n\n" +
            "Few-shot examples:\n" + json.dumps(CONV_FEWSHOT, indent=2) + "\n\n" +
            "User:\n" + CONV_INSTRUCTIONS + "\n"
        )
        conv_path = write_batch("conversation_prompt", conv_header, batch_items, tag)

        expl_header = (
            "System:\n" + EXPL_SYSTEM + "\n\n" +
            "Few-shot examples:\n" + json.dumps(EXPL_FEWSHOT, indent=2) + "\n\n" +
            "User:\n" + EXPL_INSTRUCTIONS + "\n"
        )
        expl_path = write_batch("explanation_prompt", expl_header, batch_items, tag)

        print(f"Wrote: {conv_path.name} and {expl_path.name}  (items {start+1}-{end} of {n})")


if __name__ == "__main__":
    main()
