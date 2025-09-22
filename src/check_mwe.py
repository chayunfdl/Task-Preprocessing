# src/check_mwe.py
# Cek apakah MWE (token ber-underscore) sudah masuk ke preprocessed.csv
# Usage:
#   python -m src.check_mwe --in data/output/preprocessed.csv --col tokens_regex_wordpunct --top 30

import argparse, pandas as pd
from collections import Counter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", required=True)
    ap.add_argument("--col", dest="tokens_col", required=True,
                    help="Nama kolom tokens_* di preprocessed.csv (lihat tokenizer_stats.csv)")
    ap.add_argument("--top", dest="topn", type=int, default=20)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv, dtype=str, keep_default_na=False)
    if args.tokens_col not in df.columns:
        raise SystemExit(f"Kolom {args.tokens_col} tidak ditemukan. Cek nama kolom tokens_* di preprocessed.csv")

    total_docs = len(df)
    docs_with_mwe = df[df[args.tokens_col].str.contains("_", na=False)]
    frac_docs = len(docs_with_mwe) / max(total_docs, 1)

    cnt = Counter()
    for row in df[args.tokens_col].fillna(""):
        for tok in row.split():
            if "_" in tok:
                cnt[tok] += 1

    print(f"[check_mwe] Dokumen total: {total_docs}")
    print(f"[check_mwe] Dokumen mengandung MWE (underscore): {len(docs_with_mwe)} ({frac_docs:.1%})")
    print(f"[check_mwe] Top {args.topn} MWE:")
    for tok, c in cnt.most_common(args.topn):
        print(f"  {tok:35s}  {c}")

    print("\n[check_mwe] Contoh dokumen dengan MWE:")
    for i, (_, r) in enumerate(docs_with_mwe.head(3).iterrows(), 1):
        mwes = [t for t in r[args.tokens_col].split() if "_" in t]
        print(f"  {i}. {', '.join(mwes[:12])}")

if __name__ == "__main__":
    main()
