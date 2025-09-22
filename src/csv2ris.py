import pandas as pd
from pathlib import Path
import rispy

FIELD_MAP = {
    "type": "type_of_reference",
    "title": "primary_title",
    "author": "authors",
    "year": "year",
    "journal": "secondary_title",
    "doi": "doi",
    "abstract": "abstract"
}

def row_to_ris(row):
    rec = {}
    for k, v in FIELD_MAP.items():
        val = row.get(k, "")
        if isinstance(val, float):
            val = str(val)
        rec[v] = val
    typ = str(row.get("type","article")).lower()
    rec["type_of_reference"] = "JOUR" if "article" in typ else "CONF"
    if rec.get("authors"):
        rec["authors"] = [a.strip() for a in str(rec["authors"]).replace(" and ", ";").split(";") if a.strip()]
    return rec

def run(in_csv: str, out_ris: str):
    df = pd.read_csv(in_csv)
    records = [row_to_ris(row) for _, row in df.iterrows()]
    with open(out_ris, "w", encoding="utf-8") as f:
        rispy.dump(records, f)
    print(f"[csv2ris] wrote {out_ris} ({len(records)} records)")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", required=True)
    ap.add_argument("--out", dest="out_ris", required=True)
    args = ap.parse_args()
    run(args.in_csv, args.out_ris)
