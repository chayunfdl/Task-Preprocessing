from pathlib import Path
import argparse, re
import pandas as pd
from collections import Counter

def word_tokenize_fallback(text: str):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?", text)

def sent_tokenize_fallback(text: str):
    return re.split(r'(?<=[\.\!\?])\s+', text)

def load_texts(in_csv: str) -> list[str]:
    df = pd.read_csv(in_csv, dtype=str, keep_default_na=False)
    if "abstract" not in df.columns:
        raise SystemExit("Input CSV must contain an 'abstract' column")
    return df["abstract"].fillna("").astype(str).tolist()

def normalize_basic(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def bigram_trigram_freq(tokens: list[str], topn_bigram=50, topn_trigram=50):
    bigrams = list(zip(tokens, tokens[1:]))
    trigrams = list(zip(tokens, tokens[1:], tokens[2:]))
    bfreq = Counter(["_".join(b) for b in bigrams])
    tfreq = Counter(["_".join(t) for t in trigrams])
    return bfreq.most_common(topn_bigram), tfreq.most_common(topn_trigram)

def detect_freq(texts: list[str], topn_bigram=50, topn_trigram=50):
    tokens = []
    for t in texts:
        t = normalize_basic(t)
        tokens.extend(word_tokenize_fallback(t))
    b_top, t_top = bigram_trigram_freq(tokens, topn_bigram, topn_trigram)
    phrases = [p for p,_ in b_top] + [p for p,_ in t_top]
    return sorted(set(phrases))

def detect_pmi(texts: list[str], topn=100):
    try:
        import nltk
        from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
        from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures
        from nltk.tokenize import word_tokenize
        try:
            nltk.data.find("tokenizers/punkt")
        except Exception:
            nltk.download("punkt", quiet=True)
        tokens = []
        for t in texts:
            tokens.extend(word_tokenize(t.lower()))
        b = BigramCollocationFinder.from_words(tokens)
        t = TrigramCollocationFinder.from_words(tokens)
        b_pmi = b.nbest(BigramAssocMeasures.pmi, topn)
        t_pmi = t.nbest(TrigramAssocMeasures.pmi, topn)
        return sorted(set(["_".join(x) for x in b_pmi] + ["_".join(x) for x in t_pmi]))
    except Exception as e:
        print("[warn] NLTK PMI unavailable, fallback to freq:", e)
        return detect_freq(texts, topn, topn)

def detect_gensim(texts: list[str], min_count=2, threshold=8.0):
    try:
        from gensim.models import Phrases
        from gensim.models.phrases import Phraser
        try:
            import nltk
            from nltk.tokenize import word_tokenize, sent_tokenize
            try:
                nltk.data.find("tokenizers/punkt")
            except Exception:
                nltk.download("punkt", quiet=True)
            sent_tok = sent_tokenize
            word_tok = word_tokenize
        except Exception:
            sent_tok = sent_tokenize_fallback
            word_tok = word_tokenize_fallback
        sents = []
        for t in texts:
            for s in sent_tok(t.lower()):
                sents.append([w for w in word_tok(s) if w.strip()])
        bigram = Phrases(sents, min_count=min_count, threshold=threshold)
        bigram_phraser = Phraser(bigram)
        phrased = [bigram_phraser[s] for s in sents]
        phrases = {tok for sent in phrased for tok in sent if "_" in tok}
        return sorted(phrases)
    except Exception as e:
        print("[warn] gensim Phrases unavailable, fallback to freq:", e)
        return detect_freq(texts, 100, 100)

def detect_ner(texts: list[str]):
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        print("[warn] spaCy NER unavailable:", e)
        return []
    phrases = set()
    for t in texts:
        doc = nlp(t)
        for ent in doc.ents:
            if len(ent.text.split()) > 1:
                phrases.add(ent.text.lower().strip().replace(" ", "_"))
    return sorted(phrases)

def demo_bert_subword(text: str):
    try:
        from transformers import BertTokenizer
        tok = BertTokenizer.from_pretrained("bert-base-uncased")
        tokens = tok.tokenize(text)
        ids = tok.encode(text)
        return tokens, ids
    except Exception:
        return ["(transformers not installed)"], []

def main():
    ap = argparse.ArgumentParser(description="Detect MWE from abstracts")
    ap.add_argument("--in", dest="in_csv", required=True)
    ap.add_argument("--out", dest="out_txt", required=True)
    ap.add_argument("--mode", choices=["freq","pmi","gensim","ner","bert_subword"], default="gensim")
    ap.add_argument("--topn", type=int, default=100)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=8.0)
    ap.add_argument("--demo-text", type=str, default="I love NLP, it's so interesting!")
    args = ap.parse_args()

    texts = load_texts(args.in_csv)
    outp = Path(args.out_txt); outp.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "freq":
        phrases = detect_freq(texts, args.topn, args.topn)
    elif args.mode == "pmi":
        phrases = detect_pmi(texts, args.topn)
    elif args.mode == "gensim":
        phrases = detect_gensim(texts, args.min_count, args.threshold)
    elif args.mode == "ner":
        phrases = detect_ner(texts)
    elif args.mode == "bert_subword":
        toks, ids = demo_bert_subword(args.demo_text)
        print("Tokens :", toks); print("Token IDs :", ids)
        phrases = []
    else:
        phrases = []

    with open(outp, "w", encoding="utf-8") as f:
        for p in phrases:
            f.write(p + "\n")
    print(f"[mwe_detect] wrote {outp} ({len(phrases)} phrases)")

if __name__ == "__main__":
    main()
