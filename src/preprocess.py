import time, json, re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ===== optional heavy libs with graceful fallback =====
def try_import_spacy():
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except Exception:
        return None

def try_import_nltk():
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
    except Exception:
        try:
            import nltk
            nltk.download('punkt', quiet=True)
        except Exception:
            pass
    try:
        import nltk
        from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
        try:
            nltk.data.find('corpora/wordnet')
        except Exception:
            nltk.download('wordnet', quiet=True)
        return nltk
    except Exception:
        return None

def try_import_gensim():
    try:
        from gensim.utils import simple_preprocess
        return simple_preprocess
    except Exception:
        return None

def regex_wordpunct(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?", text)

def load_stopwords(domain_path):
    base = set('''a an the and or but if while of for to in on with without from by as is are was were be been being 
    i you he she it we they that this these those not can could should would may might must do does did done having have has
    our your their its'''.split())
    if domain_path and Path(domain_path).exists():
        extra = [w.strip().lower() for w in Path(domain_path).read_text(encoding='utf-8').splitlines() if w.strip()]
        base.update(extra)
    # Protect domain terms
    for protect in ["nlp", "natural language processing"]:
        if protect in base: base.remove(protect)
    return base

def apply_synonyms(tokens, syn_map):
    if not syn_map:
        return tokens
    rep = {}
    for k, vs in syn_map.items():
        k_l = k.lower().strip()
        for v in vs:
            v_l = str(v).lower().strip()
            if v_l and v_l != k_l:
                rep[v_l] = k_l
    return [rep.get(t.lower(), t.lower()) for t in tokens]

# ===== Multi-Word Expressions (MWE) =====
def load_mwe_list(path_str: str):
    if not path_str: return []
    p = Path(path_str)
    if not p.exists(): return []
    phrases = []
    for line in p.read_text(encoding="utf-8").splitlines():
        t = line.strip()
        if t and "_" in t:
            phrases.append(t.split("_"))
    phrases.sort(key=lambda x: -len(x))  # longest first
    return phrases

def replace_multiword(tokens, phrases):
    if not phrases: return tokens
    out = []; i = 0; L = len(tokens)
    while i < L:
        matched = False
        for ph in phrases:
            k = len(ph)
            if i + k <= L and tokens[i:i+k] == ph:
                out.append("_".join(ph)); i += k; matched = True; break
        if not matched:
            out.append(tokens[i]); i += 1
    return out

def run(in_csv: str, out_csv: str, domain_stop: str, synonyms_json: str, figures_dir: str, mwe_list: str=None):
    df = pd.read_csv(in_csv, dtype=str, keep_default_na=False)
    texts = df["abstract"].fillna("").astype(str).tolist()

    nlp = try_import_spacy()
    nltk = try_import_nltk()
    gensim_tok = try_import_gensim()

    tokenizers = {"regex_wordpunct": (regex_wordpunct, "Regex word/punct tokenizer")}
    if nltk:
        from nltk.tokenize import word_tokenize
        tokenizers["nltk_word_tokenize"] = (word_tokenize, "NLTK Punkt tokenizer")
    if gensim_tok:
        tokenizers["gensim_simple_preprocess"] = (gensim_tok, "Gensim simple_preprocess")
    if nlp:
        tokenizers["spacy_tokenizer"] = (lambda s: [t.text for t in nlp(s)], "spaCy tokenizer")

    stopwords = load_stopwords(domain_stop)
    syn_map = json.loads(Path(synonyms_json).read_text(encoding="utf-8")) if synonyms_json and Path(synonyms_json).exists() else {}
    mwe_phrases = load_mwe_list(mwe_list)

    stats = []; tokenized_store = {}

    for name, (fn, desc) in tokenizers.items():
        t0 = time.time()
        toks = [[tok.lower() for tok in fn(txt)] for txt in texts]
        # MWE replacement BEFORE stopwords/synonyms
        if mwe_phrases:
            toks = [replace_multiword(doc, mwe_phrases) for doc in toks]
        toks_sw = [[w for w in doc if w not in stopwords] for doc in toks]
        toks_syn = [apply_synonyms(doc, syn_map) for doc in toks_sw]
        total_tokens = sum(len(doc) for doc in toks_syn)
        vocab = set([w for doc in toks_syn for w in doc])
        avg_len = (total_tokens / max(len(toks_syn), 1)) if toks_syn else 0
        dur = time.time() - t0
        stats.append({"tokenizer": name, "desc": desc, "docs": len(toks_syn), "avg_tokens_per_doc": avg_len, "vocab_size": len(vocab), "time_sec": dur})
        tokenized_store[name] = [" ".join(doc) for doc in toks_syn]

    best = sorted(stats, key=lambda r: (r["time_sec"], -r["vocab_size"]))[0]["tokenizer"]
    df["tokens_"+best] = tokenized_store[best]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[preprocess] wrote {out_csv}")

    # ===== figures =====
    fig_dir = Path(figures_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    labels = [s["tokenizer"] for s in stats]

    plt.figure(); plt.bar(labels, [s["time_sec"] for s in stats]); plt.ylabel("Seconds (lower is better)"); plt.title("Tokenizer Speed"); plt.xticks(rotation=20, ha='right'); plt.tight_layout(); plt.savefig(fig_dir/"fig_tokenizers.png"); plt.close()
    plt.figure(); plt.bar(labels, [s["avg_tokens_per_doc"] for s in stats]); plt.ylabel("Avg tokens/doc"); plt.title("Tokenizer Output Length"); plt.xticks(rotation=20, ha='right'); plt.tight_layout(); plt.savefig(fig_dir/"fig_token_len.png"); plt.close()
    plt.figure(); plt.bar(labels, [s["vocab_size"] for s in stats]); plt.ylabel("Vocab size"); plt.title("Tokenizer Vocabulary Size"); plt.xticks(rotation=20, ha='right'); plt.tight_layout(); plt.savefig(fig_dir/"fig_token_vocab.png"); plt.close()

    # Stemming vs Lemma chart (only if nltk available)
    vocab_sizes = {}
    if nltk:
        from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
        ps = PorterStemmer(); sn = SnowballStemmer("english"); wnl = WordNetLemmatizer()
        best_docs = [d.split() for d in tokenized_store[best]]
        def apply_proc(proc, docs): return [[proc(w) for w in doc] for doc in docs]
        variants = {
            "porter_stem": lambda w: ps.stem(w),
            "snowball_stem": lambda w: sn.stem(w),
            "wordnet_lemma": lambda w: wnl.lemmatize(w)
        }
        for name2, fn2 in variants.items():
            proc_docs = apply_proc(lambda w: fn2(w), best_docs)
            vocab_sizes[name2] = len(set([w for doc in proc_docs for w in doc]))
        plt.figure(); plt.bar(list(vocab_sizes.keys()), list(vocab_sizes.values())); plt.ylabel("Vocab size"); plt.title("Stemming vs Lemmatization"); plt.xticks(rotation=20, ha='right'); plt.tight_layout(); plt.savefig(fig_dir/"fig_norm.png"); plt.close()

    pd.DataFrame(stats).to_csv(fig_dir/"tokenizer_stats.csv", index=False)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", required=True)
    ap.add_argument("--out", dest="out_csv", required=True)
    ap.add_argument("--domain-stop", dest="domain_stop", required=False, default=None)
    ap.add_argument("--syn", dest="synonyms_json", required=False, default=None)
    ap.add_argument("--figdir", dest="figures_dir", required=True)
    ap.add_argument("--mwe", dest="mwe_list", required=False, default=None, help="Path to MWE list (underscore)")
    args = ap.parse_args()
    run(args.in_csv, args.out_csv, args.domain_stop, args.synonyms_json, args.figures_dir, args.mwe_list)
