from pathlib import Path
import bibtexparser
import pandas as pd

def read_bib(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        db = bibtexparser.load(f)
    rows = []
    for e in db.entries:
        rows.append({
            "id": e.get("ID") or e.get("id"),
            "type": e.get("ENTRYTYPE"),
            "title": e.get("title","").strip("{}"),
            "author": e.get("author",""),
            "year": e.get("year",""),
            "journal": e.get("journal") or e.get("booktitle") or "",
            "abstract": e.get("abstract",""),
            "doi": e.get("doi",""),
        })
    return pd.DataFrame(rows)

def run(in_dir: str, out_csv: str):
    p = Path(in_dir)
    dfs = []
    for bib in p.glob("*.bib"):
        dfs.append(read_bib(bib))
    if not dfs:
        raise SystemExit(f"No .bib files found in {p}")
    df = pd.concat(dfs, ignore_index=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[merge_bibtex] merged: {len(df)} rows -> {out_csv}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument("--out", dest="out_csv", required=True)
    args = ap.parse_args()
    run(args.in_dir, args.out_csv)
