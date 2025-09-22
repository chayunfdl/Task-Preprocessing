# src/clean_mwe.py
# Bersihkan mwe_list.txt dari frasa generik/kurang informatif.
# Usage:
#   python -m src.clean_mwe --in data/output/mwe/mwe_list.txt --out data/output/mwe/mwe_list.clean.txt

import argparse, re
from pathlib import Path

# stop kata umum (en + id), boleh tambah sendiri
STOP = {
    "a","an","the","and","or","but","of","for","to","in","on","with","without","by","as",
    "is","are","was","were","be","been","being","this","that","these","those",
    "we","i","you","they","he","she","it","our","your","their",
    "can","could","should","would","may","might","must","not",
    "di","ke","dari","yang","dan","atau","pada","dengan","tanpa"
}

# pola frasa generik/fungsional yang sering tidak informatif
BAD_PATTERNS = [
    r"^this_", r"^we_", r"^it_", r"^to_", r"^in_", r"^on_", r"^by_",
    r"_of$", r"_to$", r"_in$", r"_for$", r"_and$", r"_the$",
    r"^based_on$", r"^such_as$", r"^due_to$", r"^in_addition$",
    r"^real_world$", r"^can_be$", r"^aims_to$", r"^compared_to$",
    r"^a_novel$", r"^this_study$", r"^this_paper$", r"^experimental_results$"
]

WHITELIST = {
    # frasa domain yang harus dipertahankan meski mengandung stopword
    "natural_language_processing",
    "knowledge_graph", "knowledge_graphs",
    "machine_learning", "deep_learning",
    "large_language", "large_language_models", "language_model", "language_models",
    "sentiment_analysis", "hate_speech",
    "live_streaming", "on_tiktok", "tiktok_shop", "virtual_gift",
    "graph_neural_network", "graph_neural_networks"
}

def is_bad(phrase: str) -> bool:
    # singkirkan jika semua token terlalu pendek
    toks = phrase.split("_")
    if all(len(t) <= 2 for t in toks):
        return True
    # singkirkan jika >50% token stopword
    if sum(t in STOP for t in toks) / max(len(toks),1) > 0.5:
        return True
    # cocokkan pola buruk
    for pat in BAD_PATTERNS:
        if re.search(pat, phrase):
            return True
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_txt", required=True)
    ap.add_argument("--out", dest="out_txt", required=True)
    args = ap.parse_args()
    inp = Path(args.in_txt)
    phrases = [l.strip() for l in inp.read_text(encoding="utf-8").splitlines() if l.strip()]
    kept, dropped = [], []
    for p in phrases:
        if p in WHITELIST:
            kept.append(p); continue
        (dropped if is_bad(p) else kept).append(p)
    Path(args.out_txt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_txt).write_text("\n".join(sorted(set(kept))), encoding="utf-8")
    print(f"[clean_mwe] kept={len(set(kept))} dropped={len(set(dropped))} -> {args.out_txt}")

if __name__ == "__main__":
    main()
