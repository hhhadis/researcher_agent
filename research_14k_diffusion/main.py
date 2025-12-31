import argparse
import json
import os
from typing import Dict
from .utils import load_split, iter_split, get_year, get_title, get_arxiv
from .llm_glm import classify_and_extract
from typing import Dict, Tuple, List
from .embedding import aggregate

def process_item(item: Dict) -> Tuple[Dict, List[str]]:
    title = get_title(item)
    abstract = item.get("abstract") or ""
    sections = item.get("sections") or []
    res = classify_and_extract(title, abstract, sections)
    if not res.get("is_diffusion"):
        return {}, []
    paragraphs = res.get("paragraphs") or []
    vec = aggregate(paragraphs)
    out = {
        "title": title,
        "year": get_year(item),
        "arxiv": get_arxiv(item),
        "vector": vec,
    }
    return out, paragraphs

def run(split: str, out_path: str, limit: int | None = None, show_progress: bool = True) -> int:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    data = load_split(split)
    total = len(data)
    n = 0
    with open(out_path, "w", encoding="utf-8") as w:
        for i, item in enumerate(data):
            if limit is not None and i >= limit:
                break
            out, paragraphs = process_item(item)
            if out:
                if n < 3:
                    print(f"\n--- Sample {n+1} ---")
                    print(f"Title: {out['title']}")
                    print(f"Paragraphs for embedding ({len(paragraphs)}):")
                    for i, p in enumerate(paragraphs):
                        print(f"  [{i+1}] {p[:100]}...") # Print first 100 chars
                    print("---------------------")
                w.write(json.dumps(out, ensure_ascii=False) + "\n")
                n += 1
            if show_progress:
                done = i + 1
                bar_len = 30
                ratio = done / (total if total else 1)
                filled = int(bar_len * ratio)
                bar = "#" * filled + "-" * (bar_len - filled)
                print(f"\r{split} {done}/{total} |{bar}| matched {n}", end="", flush=True)
    if show_progress:
        print()
    return n

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="train")
    p.add_argument("--out", default=os.path.join("outputs", "diffusion_train.jsonl"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-progress", action="store_true")
    args = p.parse_args()
    c = run(args.split, args.out, args.limit, show_progress=not args.no_progress)
    print(f"wrote {c} items to {args.out}")

if __name__ == "__main__":
    main()

