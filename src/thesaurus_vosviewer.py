from pathlib import Path
import argparse, csv, json
from collections import defaultdict, deque

def load_synonyms(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    norm = {}
    for k, vs in data.items():
        k = (k or "").strip()
        if not k:
            continue
        seen = set()
        clean_vs = []
        for v in (vs or []):
            v = (v or "").strip()
            if v and v.lower() not in seen and v.lower() != k.lower():
                seen.add(v.lower())
                clean_vs.append(v)
        norm[k] = clean_vs
    return norm

def build_canonical_components(syn: dict) -> dict:
    canon_set = {c.lower().strip() for c in syn.keys()}
    g = defaultdict(set)
    for c, vs in syn.items():
        c_l = c.lower().strip()
        for v in (vs or []):
            v_l = v.lower().strip()
            if v_l in canon_set:
                g[c_l].add(v_l); g[v_l].add(c_l)
    root_map = {}
    visited = set()
    for c in canon_set:
        if c in visited: continue
        q = deque([c]); comp = []; visited.add(c)
        while q:
            u = q.popleft(); comp.append(u)
            for w in g[u]:
                if w not in visited:
                    visited.add(w); q.append(w)
        root = min(comp)  # deterministik
        for node in comp: root_map[node] = root
    for c in canon_set:
        root_map.setdefault(c, c)
    return root_map

def write_thesaurus(syn: dict, out_path: Path) -> int:
    canon_set = {k.lower().strip() for k in syn.keys()}
    comp_root = build_canonical_components(syn)
    rows = [("label","replace by")]
    for canon, variants in syn.items():
        canon_root = comp_root.get(canon.lower().strip(), canon.lower().strip())
        for v in (variants or []):
            v_norm = (v or "").strip()
            if not v_norm: continue
            if v_norm.lower() in canon_set:  # skip varian yang juga kanonik
                continue
            rows.append((v_norm, canon_root))
    seen = set(); out = []
    for a,b in rows:
        key = (a.lower().strip(), b.lower().strip())
        if key not in seen:
            seen.add(key); out.append((a,b))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(out)
    return len(out)

def main():
    ap = argparse.ArgumentParser(description="Generate VOSviewer thesaurus from synonyms.json (cycle-safe)")
    ap.add_argument("--syn", required=True, help="Path to config/synonyms.json")
    ap.add_argument("--out", required=True, help="Output thesaurus file path")
    args = ap.parse_args()
    syn = load_synonyms(Path(args.syn))
    n = write_thesaurus(syn, Path(args.out))
    print(f"[thesaurus] wrote {args.out} ({n} rows)")

if __name__ == "__main__":
    main()
