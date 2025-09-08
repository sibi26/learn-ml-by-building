# extract_webshop_categories.py
import json
from pathlib import Path

# Resolve dataset path whether you run from repo root or from "Lecture 1 Overview/"
candidates = [
    Path("data/items_shuffle_1000.json"),
    Path("Lecture 1 Overview/data/items_shuffle_1000.json"),
]
src = next((p for p in candidates if p.exists()), None)
if not src:
    raise FileNotFoundError("items_shuffle_1000.json not found. Run from repo root or Lecture 1 Overview/")

print(f"Reading: {src}")
data = json.loads(src.read_text(encoding="utf-8"))
items = data if isinstance(data, list) else list(data.values())

# Collect short categories
short_cats = set()
# Collect top-level breadcrumb categories from product_category
top_level_cats = set()

def top_level_from_breadcrumb(breadcrumb: str):
    if not isinstance(breadcrumb, str):
        return None
    # Split on common separators: '›', '>', '/'
    for sep in ["›", ">", "/"]:
        if sep in breadcrumb:
            parts = [p.strip() for p in breadcrumb.split(sep)]
            # first non-empty
            for p in parts:
                if p:
                    return p
    # fallback: entire breadcrumb if it's just a single label
    return breadcrumb.strip() or None

for it in items:
    if not isinstance(it, dict):
        continue
    # short tag
    cat = it.get("category")
    if isinstance(cat, str) and cat.strip():
        short_cats.add(cat.strip())

    # breadcrumb top-level
    tl = top_level_from_breadcrumb(it.get("product_category", ""))
    if tl:
        top_level_cats.add(tl)

# Sort
short_cats = sorted(short_cats)
top_level_cats = sorted(top_level_cats)

print(f"\nUnique short categories ({len(short_cats)}):")
for c in short_cats:
    print(" -", c)

print(f"\nUnique top-level product_category ({len(top_level_cats)}):")
for c in top_level_cats:
    print(" -", c)

# Save for prompting an LLM
out_dir = Path("data/processed") if Path("data").exists() else Path("Lecture 1 Overview/data/processed")
out_dir.mkdir(parents=True, exist_ok=True)

(out_dir / "webshop_categories_short.json").write_text(json.dumps(short_cats, indent=2), encoding="utf-8")
(out_dir / "webshop_categories_top_level.json").write_text(json.dumps(top_level_cats, indent=2), encoding="utf-8")

print(f"\nSaved:")
print(f" - {out_dir / 'webshop_categories_short.json'}")
print(f" - {out_dir / 'webshop_categories_top_level.json'}")