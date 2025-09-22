import pandas as pd
def run(in_csv: str, out_csv: str):
    df = pd.read_csv(in_csv)
    cols = ["id","type","title","author","year","journal","abstract","doi"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = df[cols]
    df.to_csv(out_csv, index=False)
    print(f"[bib2csv] wrote {out_csv}")
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", required=True)
    ap.add_argument("--out", dest="out_csv", required=True)
    args = ap.parse_args()
    run(args.in_csv, args.out_csv)
