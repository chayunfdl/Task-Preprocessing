import pandas as pd
import hashlib
import re

def norm_str(x):
    if x is None:
        return ""
    # cast to str, handle NaN
    s = str(x)
    if s.lower() == "nan":
        return ""
    return s

def clean_title(t):
    s = norm_str(t)
    # hapus kurung kurawal latex {}, trim spasi
    s = s.replace("{", "").replace("}", "")
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def clean_author(a):
    s = norm_str(a)
    s = s.strip()
    # ambil author pertama untuk stabilitas kunci
    if " and " in s:
        s = s.split(" and ", 1)[0]
    elif ";" in s:
        s = s.split(";", 1)[0]
    return s.lower()

def clean_year(y):
    s = norm_str(y).strip()
    # ambil 4 digit jika ada
    m = re.search(r"\b(19|20)\d{2}\b", s)
    return m.group(0) if m else s

def clean_doi(d):
    s = norm_str(d).strip().lower()
    s = s.replace("https://doi.org/", "").replace("http://doi.org/", "")
    s = s.replace("doi:", "").strip()
    return s

def make_key(row):
    doi = clean_doi(row.get("doi", ""))
    if doi:
        return "doi:" + doi
    title = clean_title(row.get("title", ""))
    author = clean_author(row.get("author", ""))
    year = clean_year(row.get("year", ""))
    payload = f"{title}|{author}|{year}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()

def run(in_csv: str, out_csv: str):
    # baca semua kolom sebagai string agar aman dari .strip()
    df = pd.read_csv(in_csv, dtype=str, keep_default_na=False)
    before = len(df)
    df["__key"] = df.apply(make_key, axis=1)
    df = df.drop_duplicates("__key").drop(columns="__key")
    df.to_csv(out_csv, index=False)
    print(f"[dedupe] {before} -> {len(df)} unique")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", required=True)
    ap.add_argument("--out", dest="out_csv", required=True)
    args = ap.parse_args()
    run(args.in_csv, args.out_csv)
