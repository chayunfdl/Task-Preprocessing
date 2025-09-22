# Task-Preprocessing

Pipeline: merge .bib → dedupe → CSV → preprocess (tokenize, stopwords, stemming/lemmatization, synonyms) → export .ris for VOSviewer.

## Install
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download xx_ent_wiki_sm
pip install spacy transformers torch --index-url https://download.pytorch.org/whl/cpu
pip install sentencepiece tokenizers "protobuf<5"

```
## Run
```bash
python -m src.run_all --in data/input --out data/output --syn config/synonyms.json --domain-stop config/domain_stopwords.txt


python -m src.build_thesaurus_final `
  --in  "data/output/preprocessed.csv" `
  --col "tokens_regex_wordpunct" `
  --syn "config/synonyms.json" `
  --out "config/thesaurus_vosviewer.txt"

# NLTK PMI
python -m src.mwe_detect --in data/output/preprocessed.csv --out data/output/mwe/mwe_nltk.txt --mode pmi --topn 1500
# Gensim
python -m src.mwe_detect --in data/output/preprocessed.csv --out data/output/mwe/mwe_gensim.txt --mode gensim --min-count 2 --threshold 8
# spaCy NER
python -m src.mwe_detect --in data/output/preprocessed.csv --out data/output/mwe/mwe_spacy.txt --mode ner --spacy-model xx_ent_wiki_sm
# BERT NER
python -m src.mwe_detect --in data/output/preprocessed.csv --out data/output/mwe/mwe_bert.txt --mode bert --bert-model Davlan/xlm-roberta-base-ner-hrl

python -m src.viz_mwe_figures
```
