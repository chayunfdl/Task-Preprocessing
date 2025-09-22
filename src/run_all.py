from pathlib import Path
import subprocess, sys

def run_cmd(args):
    print("+", " ".join(map(str, args)))
    subprocess.run([sys.executable, *map(str, args)], check=True)

def main(indir, outdir, syn, dstop):
    root = Path(__file__).resolve().parent.parent
    tmp = root/"data/tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    merged = tmp/"merged.csv"
    dedup = tmp/"merged_dedup.csv"
    csv_norm = (root/outdir)/"merged_dedup.csv"
    preproc = (root/outdir)/"preprocessed.csv"
    ris_out = (root/outdir)/"export_vosviewer.ris"

    run_cmd([root/"src/merge_bibtex.py", "--in", (root/indir), "--out", merged])
    run_cmd([root/"src/dedupe.py", "--in", merged, "--out", dedup])
    run_cmd([root/"src/bib2csv.py", "--in", dedup, "--out", csv_norm])
    run_cmd([root/"src/preprocess.py", "--in", csv_norm, "--out", preproc,
             "--syn", (root/syn), "--domain-stop", (root/dstop), "--figdir", (root/outdir)])
    run_cmd([root/"src/csv2ris.py", "--in", preproc, "--out", ris_out])

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--out", dest="outdir", required=True)
    ap.add_argument("--syn", dest="syn", required=True)
    ap.add_argument("--domain-stop", dest="dstop", required=True)
    args = ap.parse_args()
    main(args.indir, args.outdir, args.syn, args.dstop)
