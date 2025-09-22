from pathlib import Path
import re, pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def read_phrases(p):
    p = Path(p)
    if not p.exists():
        print(f"[skip] {p} tidak ditemukan")
        return set()
    s = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip().lower()
        if line:
            s.add(re.sub(r"\s+","_", line))
    print(f"[ok]  {p} -> {len(s)} frasa")
    return s

def count_phrase_freq(phrases, series):
    cnt = Counter()
    for doc in series.fillna(""):
        bag = " " + doc.lower() + " "
        for p in phrases:
            if " " + p + " " in bag:
                cnt[p] += 1
    return cnt

def bar_top(counter, title, outpath, top=15):
    if not counter:
        print(f"[warn] {title}: kosong")
        return
    items = counter.most_common(top)
    labels = [k.replace("_"," ") for k,_ in items][::-1]
    values = [v for _,v in items][::-1]
    plt.figure(figsize=(9,6)); plt.barh(range(len(labels)), values)
    plt.yticks(range(len(labels)), labels); plt.xlabel("Dokumen")
    plt.title(title); plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=180); plt.close()
    print(f"[save] {outpath}")

def main():
    outdir = Path("data/output/figures")
    sets = {
        "NLTK":   read_phrases("data/output/mwe/mwe_nltk.txt"),
        "Gensim": read_phrases("data/output/mwe/mwe_gensim.txt"),
        "spaCy":  read_phrases("data/output/mwe/mwe_spacy.txt"),
        "BERT":   read_phrases("data/output/mwe/mwe_bert.txt"),
    }
    sets = {k:v for k,v in sets.items() if v}
    if not sets:
        print("[error] Tidak ada daftar frasa (mwe_*.txt). Minimal isi mwe_gensim.txt.")
        return

    # after MWT
    pp = Path("data/output/preprocessed.csv")
    if not pp.exists():
        print("[error] preprocessed.csv tidak ditemukan")
        return
    df_after = pd.read_csv(pp, dtype=str, keep_default_na=False)
    if "tokens_regex_wordpunct" not in df_after.columns:
        print("[error] Kolom tokens_regex_wordpunct tidak ada di preprocessed.csv")
        return

    print(f"[info] Dokumen sesudah MWT: {len(df_after)}")
    for m,phr in sets.items():
        freq = count_phrase_freq(phr, df_after["tokens_regex_wordpunct"])
        bar_top(freq, f"Top multi-word — {m}", outdir/f"top_mwe_{m.lower()}.png", top=15)

    # before vs after (opsional)
    tb = Path("data/output/tokenized_base.csv")
    if tb.exists():
        df_before = pd.read_csv(tb, dtype=str, keep_default_na=False)
        col_before = "tokens_gensim" if "tokens_gensim" in df_before.columns else (
                     "tokens_nltk" if "tokens_nltk" in df_before.columns else None)
        if col_before:
            union = set().union(*sets.values())
            before = count_phrase_freq(union, df_before[col_before])
            after  = count_phrase_freq(union, df_after["tokens_regex_wordpunct"])
            top = [k for k,_ in after.most_common(10)]
            if top:
                import matplotlib.pyplot as plt
                x = range(len(top)); bw=0.4
                plt.figure(figsize=(10,5))
                plt.bar([i-bw/2 for i in x], [before[k] for k in top], bw, label="Sebelum")
                plt.bar([i+bw/2 for i in x], [after[k] for k in top],  bw, label="Sesudah")
                plt.xticks(x, [t.replace("_"," ") for t in top], rotation=25, ha="right")
                plt.ylabel("Dokumen"); plt.title("Dampak MWT"); plt.legend(); plt.tight_layout()
                out = outdir/"impact_before_after.png"
                outdir.mkdir(parents=True, exist_ok=True)
                plt.savefig(out, dpi=180); plt.close()
                print(f"[save] {out}")
        else:
            print("[warn] tokenized_base.csv ada, tapi tidak ada kolom tokens_gensim/tokens_nltk")

    print(f"[done] Gambar di {outdir.resolve()}")

if __name__ == "__main__":
    main()
