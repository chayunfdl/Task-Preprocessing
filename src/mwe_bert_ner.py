# src/mwe_bert_ner.py
import argparse, re
from pathlib import Path
from transformers import pipeline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", required=True)
    ap.add_argument("--out", dest="out_txt", required=True)
    ap.add_argument("--text-col", default="abstract")
    ap.add_argument("--bert-model", default="Davlan/xlm-roberta-base-ner-hrl")
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_csv(args.in_csv, dtype=str, keep_default_na=False)
    if args.text_col not in df.columns:
        raise SystemExit(f"Kolom {args.text_col} tidak ada")
    ner = pipeline("token-classification", model=args.bert_model,
                   aggregation_strategy="simple")
    phrases = set()
    for t in df[args.text_col].fillna(""):
        for ent in ner(t):
            span = ent["word"].strip()
            if len(span.split()) > 1:
                phrases.add(re.sub(r"\s+","_", span.lower()))
    Path(args.out_txt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_txt).write_text("\n".join(sorted(phrases)), encoding="utf-8")
    print(f"[bert-ner] wrote {args.out_txt} ({len(phrases)} phrases)")

if __name__ == "__main__":
    main()
