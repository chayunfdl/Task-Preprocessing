# src/build_thesaurus_final.py
# Build VOSviewer thesaurus TANPA konflik:
# - IGNORE terms generik (termasuk "nan")
# - MERGE plural/singular, hyphen/underscore/spasi, UK↔US + special merges
# - Union-Find (path-compression) ke akar kanonik
# - Jika akar (root) termasuk ignore, semua varian dijadikan IGNORE juga
#
# Usage:
#   python -m src.build_thesaurus_final \
#     --in  data/output/preprocessed.csv \
#     --col tokens_regex_wordpunct \
#     --syn config/synonyms.json \
#     --out config/thesaurus_vosviewer.txt

import argparse, csv, json, re
from pathlib import Path
import pandas as pd

# ---------- Normalisasi UK↔US ----------
UK_US = {
    "organisation": "organization", "organisations": "organizations",
    "modelling": "modeling", "modelled": "modeled",
    "behaviour": "behavior", "behaviours": "behaviors",
    "colour": "color", "colours": "colors",
}

# ---------- Istilah generik (diabaikan/IGNORE) ----------
IGNORE_TERMS = {
    "nan","study","paper","result","results","experimental result","experimental results",
    "effect","effects","impact","impacts","role","year","years","progress","evidence",
    "content","platform","management","experience","theory","insight","analysis","analyses",
    "participant","participants","interview","interviews","questionnaire","questionnaires",
    "education","awareness","software","site","dataset","data set","data-set",
    "short term","short-term","first stage","pair","agent","reward"
}

# ---------- Bulk merges umum (kanonik = bentuk singular) ----------
BULK_MERGES = {
    "representation": ["representations"],
    "relation": ["relations","dependency relation","dependency relations"],
    "entity": ["entities"],
    "network": ["networks"],
    "layer": ["layers"],
    "sentence": ["sentences"],
    "modality": ["modalities"],
    "dependency": ["dependencies"],
    "attack": ["attacks"],
    "benchmark": ["benchmarks"],
    "triple": ["triples"],
    "video": ["videos"],
    "caption": ["captions","closed caption","closed captions"],
    "graph neural network": ["graph neural networks","gnns"],
    "knowledge graph": ["knowledge graphs"],
    "large language model": ["large language models","llms"]
}

# ---------- Special merges (domain-spesifik) ----------
SPECIAL_MERGES = {
    "social media": [
        "social medium","social-media","social_media",
        "social media platform","social-media-platform","social_media_platform",
        "social media platforms","social mediums"
    ],
    "natural language processing": [
        "language processing","nlp (natural language processing)","nlp/ natural language processing"
    ],
    "hate speech detection": [
        "hate-speech detection","hate speech","toxic speech detection","toxic speech"
    ],
    "sentiment analysis": [
        "sentiment polarity","sentiment classification","opinion mining","sentiment-analysis"
    ],
    "cross-sectional analysis": [
        "cross sectional analysis","cross-sectional analyses","cross sectional analyses"
    ],
    "short-term memory": ["short term memory"],
    "tiktok": ["tiktok video","tiktok videos","tiktok content","tiktok posts"],
    "question answering": ["question-answering","qa","q&a"],
    "live streaming": ["live stream","livestream","livestreaming","live-streaming","live_streaming"],
    "knowledge graph completion": ["kg completion","knowledge-graph completion"]
}

def normalize_variants(term: str) -> str:
    """lower, ganti _/- ke spasi, collapse spasi, UK->US, hapus duplikasi bertetangga"""
    s = (term or "").lower().strip()
    s = s.replace("_"," ").replace("-"," ")
    s = re.sub(r"\s+"," ", s)
    toks = [UK_US.get(w, w) for w in s.split()]
    dedup = []
    for w in toks:
        if not dedup or dedup[-1] != w:
            dedup.append(w)
    return " ".join(dedup)

def to_label(term: str) -> str:
    # PENTING: gunakan SPASI agar cocok dengan surface form di VOSviewer
    return term

def read_synonyms(syn_path: Path) -> dict:
    """synonyms.json → {canon(norm): set(variant(norm))}"""
    if not syn_path.exists(): return {}
    raw = json.loads(syn_path.read_text(encoding="utf-8"))
    syn: dict[str, set[str]] = {}
    for k, vs in raw.items():
        ck = normalize_variants(k)
        syn.setdefault(ck, set())
        for v in (vs or []):
            cv = normalize_variants(str(v))
            if cv and cv != ck:
                syn[ck].add(cv)
    return syn

def collect_terms(csv_path: Path, col: str) -> set[str]:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if col not in df.columns:
        raise SystemExit(f"Column {col} not found in {csv_path}")
    terms = set()
    for line in df[col].fillna(""):
        for tok in str(line).split():
            terms.add(tok)
    return terms

# ---------- Union-Find (path-compression) ----------
class CanonMap:
    def __init__(self):
        self.parent: dict[str, str] = {}   # label_norm -> parent_norm (canon)
        self.ignore: set[str] = set()      # label_norm yang di-ignore

    def find(self, x: str) -> str:
        p = self.parent.get(x, x)
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent.get(x, x)

    def union(self, a: str, b: str):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        # pilih root stabil: yang lebih pendek / lebih kecil alfabet
        root = min(ra, rb, key=lambda s: (len(s), s))
        other = rb if root == ra else ra
        self.parent[other] = root

    def set_ignore(self, x: str):
        self.ignore.add(x)

def build_canon_map(terms: set[str], syn: dict) -> CanonMap:
    cm = CanonMap()

    # Tandai IGNORE
    for t in list(terms):
        base = normalize_variants(t)
        if base in IGNORE_TERMS:
            cm.set_ignore(base)

    # Manual merges
    for canon, vars_ in syn.items():
        for v in vars_:
            cm.union(normalize_variants(v), normalize_variants(canon))

    # Special merges
    for canon, vars_ in SPECIAL_MERGES.items():
        for v in vars_:
            cm.union(normalize_variants(v), normalize_variants(canon))

    # Bulk merges
    for canon, vars_ in BULK_MERGES.items():
        for v in vars_:
            cm.union(normalize_variants(v), normalize_variants(canon))

    # Auto merges: varian spasi/underscore/hyphen + singularisasi kasar token terakhir
    for t in list(terms):
        base = normalize_variants(t)
        if not base: continue
        cm.union(base.replace(" ", "_"), base)
        cm.union(base.replace(" ", "-"), base)
        words = base.split()
        if words:
            last = words[-1]
            cand = None
            if last.endswith("es") and len(last) > 3:
                cand = last[:-2]
            elif last.endswith("s") and len(last) > 3:
                cand = last[:-1]
            if cand:
                cm.union(base, " ".join(words[:-1] + [cand]))

    return cm

def write_thesaurus(cm: CanonMap, terms: set[str], out_path: Path):
    rows = [("label","replace by")]
    seen = set()

    # IGNORE final (root yang di-ignore juga termasuk)
    ignore_set = set(cm.ignore)

    # (A) Baris IGNORE
    for t in sorted({normalize_variants(x) for x in terms} | ignore_set):
        if t in ignore_set:
            key = (to_label(t), "")
            if key not in seen:
                seen.add(key); rows.append(key)

    # (B) Baris MERGE → ke akar (root); jika root di-ignore, jadikan IGNORE juga
    all_terms_norm = {normalize_variants(x) for x in terms}
    for t in sorted(all_terms_norm):
        if t in ignore_set:
            continue
        root = cm.find(t)
        if root in ignore_set:
            key = (to_label(t), "")
            if key not in seen:
                seen.add(key); rows.append(key)
        elif root != t:
            key = (to_label(t), to_label(root))
            if key not in seen:
                seen.add(key); rows.append(key)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"[build_thesaurus_final] wrote {out_path} with {len(rows)-1} rules")

def main():
    ap = argparse.ArgumentParser(description="Build conflict-free VOSviewer thesaurus (labels use spaces)")
    ap.add_argument("--in",  dest="in_csv",  required=True)
    ap.add_argument("--col", dest="tokens_col", required=True)
    ap.add_argument("--syn", dest="syn_json", required=True)
    ap.add_argument("--out", dest="out_txt",  required=True)
    args = ap.parse_args()

    syn = read_synonyms(Path(args.syn_json))
    terms = collect_terms(Path(args.in_csv), args.tokens_col)
    cm = build_canon_map(terms, syn)
    write_thesaurus(cm, terms, Path(args.out_txt))

if __name__ == "__main__":
    main()
