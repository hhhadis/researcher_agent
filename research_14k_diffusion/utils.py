import os
import json
from typing import Dict, Iterable, List

def load_split(split: str) -> List[Dict]:
    root = os.path.join(os.getcwd(), "Research-14K", "data")
    path = os.path.join(root, f"{split}.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def iter_split(split: str) -> Iterable[Dict]:
    for item in load_split(split):
        yield item

def get_year(item: Dict) -> int:
    y = item.get("year")
    try:
        return int(y) if y is not None else -1
    except Exception:
        return -1

def get_title(item: Dict) -> str:
    return str(item.get("title") or "")

def get_arxiv(item: Dict) -> str:
    return str(item.get("arxiv") or "")

