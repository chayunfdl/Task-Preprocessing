# src/check_thesaurus_conflict.py
# Cek apakah ada istilah yang muncul baik di kolom LABEL maupun REPLACE BY.
# Usage:
#   python -m src.check_thesaurus_conflict --in config/thesaurus_vosviewer.txt

import argparse, csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_txt", required=True)
    args = ap.parse_args()

    labels, replaces = set(), set()
    rows = []
    with open(args.in_txt, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for i, row in enumerate(reader, start=2):  # header=1
            if not row: continue
            lab = (row[0] or "").strip()
            rep = (row[1] or "").strip() if len(row) > 1 else ""
            rows.append((i, lab, rep))
            if lab: labels.add(lab)
            if rep: replaces.add(rep)

    conflicts = labels & replaces
    if not conflicts:
        print("[ok] No conflicts: no term appears both as LABEL and REPLACE BY.")
        return

    print(f"[conflict] {len(conflicts)} term(s) appear in both columns:")
    for term in sorted(conflicts):
        print("  -", term)

    print("\n[lines]")
    for i, lab, rep in rows:
        if lab in conflicts or rep in conflicts:
            print(f"  line {i}: LABEL='{lab}'  REPLACE BY='{rep}'")

if __name__ == "__main__":
    main()
