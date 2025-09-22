# Task-Preprocessing

Pipeline: merge .bib → dedupe → CSV → preprocess (tokenize, stopwords, stemming/lemmatization, synonyms) → export .ris for VOSviewer.

## Install
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
## Run
```bash
python -m src.run_all --in data/input --out data/output --syn config/synonyms.json --domain-stop config/domain_stopwords.txt


python -m src.build_thesaurus_final `
  --in  "data/output/preprocessed.csv" `
  --col "tokens_regex_wordpunct" `
  --syn "config/synonyms.json" `
  --out "config/thesaurus_vosviewer.txt"
```
