# -*- coding: utf-8 -*-
"""
Detect Multi-Word Expressions (MWE) dari kolom 'abstract' CSV.

Modes:
  - freq         : bigram/trigram frequency
  - pmi          : NLTK PMI collocation
  - gensim       : Gensim Phrases (bigram/trigram)
  - ner          : spaCy NER  (--spacy-model)
  - bert         : BERT-based NER (--bert-model, use_fast=False)
  - bert_subword : DEMO tokenisasi subword BERT (bukan NER)
"""

from pathlib import Path
import argparse, re
import pandas as pd
from collections import Counter

# =============== util umum ===============

def word_tokenize_fallback(text: str):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?", text)

def sent_split_simple(text: str):
    # pemisah kalimat ringan
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

def normalize_phrase(span: str) -> str:
    s = span.strip().lower()
    s = re.sub(r"\s+", "_", s)
    return re.sub(r"_+", "_", s)

# =============== detektor statistik ===============

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
            sent_tok = sent_split_simple
            word_tok = word_tokenize_fallback
        sents = []
        for t in texts:
            for s in sent_tok(t.lower()):
                sents.append([w for w in word_tok(s) if w.strip()])
        bigram = Phrases(sents, min_count=min_count, threshold=threshold)
        bigram_phraser = Phraser(bigram)
        phrased = [bigram_phraser[s] for s in sents]
        trigram = Phrases(phrased, min_count=min_count, threshold=threshold)
        trigram_phraser = Phraser(trigram)
        phrased2 = [trigram_phraser[s] for s in phrased]
        phrases = {tok for sent in phrased2 for tok in sent if "_" in tok}
        return sorted(phrases)
    except Exception as e:
        print("[warn] gensim Phrases unavailable, fallback to freq:", e)
        return detect_freq(texts, 100, 100)

# =============== spaCy NER (FIX) ===============

def detect_ner(texts: list[str], spacy_model: str = "xx_ent_wiki_sm", batch_size: int = 64):
    """
    - support --spacy-model
    - fallback berantai: requested -> en_core_web_sm -> xx_ent_wiki_sm
    - nlp.pipe untuk batching
    - chunk teks panjang (~250 kata) agar aman
    """
    try:
        import spacy
        try:
            nlp = spacy.load(spacy_model, disable=["parser","tagger","textcat"])
        except Exception:
            try:
                print(f"[warn] '{spacy_model}' tidak tersedia, fallback ke en_core_web_sm")
                nlp = spacy.load("en_core_web_sm", disable=["parser","tagger","textcat"])
            except Exception:
                print("[warn] en_core_web_sm tidak ada, fallback ke xx_ent_wiki_sm")
                nlp = spacy.load("xx_ent_wiki_sm")
    except Exception as e:
        print("[warn] spaCy tidak tersedia:", e)
        return []

    nlp.max_length = max(2_000_000, nlp.max_length)

    # chunk teks
    chunks = []
    for t in texts:
        ss = sent_split_simple(t)
        block, cnt = [], 0
        for s in ss:
            wc = len(s.split())
            if cnt + wc > 250 and block:
                chunks.append(" ".join(block)); block, cnt = [s], wc
            else:
                block.append(s); cnt += wc
        if block:
            chunks.append(" ".join(block))

    phrases = set()
    for doc in nlp.pipe(chunks, batch_size=batch_size):
        for ent in doc.ents:
            if len(ent.text.split()) > 1:
                phrases.add(normalize_phrase(ent.text))

    return sorted(phrases)

# =============== BERT NER (FIX: use_fast=False) ===============

def detect_bert_ner(texts: list[str], bert_model: str, batch_size: int = 8):
    """
    - gunakan AutoTokenizer(use_fast=False) supaya tidak perlu tiktoken
    - pipeline token-classification, aggregation_strategy="simple"
    - chunk teks panjang (~250 kata) + batching
    """
    try:
        from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    except Exception as e:
        print("[warn] transformers tidak tersedia:", e)
        return []

    print(f"[info] Memuat BERT NER (slow tokenizer): {bert_model}")
    tokenizer = AutoTokenizer.from_pretrained(bert_model, use_fast=False)
    model = AutoModelForTokenClassification.from_pretrained(bert_model)
    ner = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=-1,  # CPU
    )

    # chunk
    chunks = []
    for t in texts:
        ss = sent_split_simple(t)
        block, cnt = [], 0
        for s in ss:
            wc = len(s.split())
            if cnt + wc > 250 and block:
                chunks.append(" ".join(block)); block, cnt = [s], wc
            else:
                block.append(s); cnt += wc
        if block:
            chunks.append(" ".join(block))

    phrases = set()
    for i in range(0, len(chunks), batch_size):
        outs = ner(chunks[i:i+batch_size])  # list-of-list
        for ents in outs:
            for ent in ents:
                span = (ent.get("word") or "").strip()
                if span and len(span.split()) > 1:
                    phrases.add(normalize_phrase(span))

    return sorted(phrases)

# =============== DEMO subword BERT ===============

def demo_bert_subword(text: str):
    try:
        from transformers import BertTokenizer
        tok = BertTokenizer.from_pretrained("bert-base-uncased")
        tokens = tok.tokenize(text); ids = tok.encode(text)
        return tokens, ids
    except Exception:
        return ["(transformers not installed)"], []

# =============== CLI ===============

def main():
    ap = argparse.ArgumentParser(description="Detect MWE from abstracts")
    ap.add_argument("--in", dest="in_csv", required=True)
    ap.add_argument("--out", dest="out_txt", required=True)
    ap.add_argument("--mode",
                    choices=["freq","pmi","gensim","ner","bert","bert_subword"],
                    default="gensim")
    ap.add_argument("--topn", type=int, default=100)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=8.0)
    ap.add_argument("--spacy-model", type=str, default="xx_ent_wiki_sm")
    ap.add_argument("--bert-model",  type=str, default="Davlan/xlm-roberta-base-ner-hrl")
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
        phrases = detect_ner(texts, spacy_model=args.spacy_model)
    elif args.mode == "bert":
        phrases = detect_bert_ner(texts, bert_model=args.bert_model)
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
