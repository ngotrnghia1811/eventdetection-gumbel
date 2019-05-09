# Data

## ACE 2005

The ACE 2005 dataset (LDC2006T06) requires an LDC license. Request access at:
https://catalog.ldc.upenn.edu/LDC2006T06

Once obtained, preprocess it with:

```bash
python scripts/preprocess_ace2005.py \
    --ace_dir /path/to/LDC2006T06/data/English \
    --output_dir data/ace2005/
```

This produces `train.json`, `dev.json`, and `test.json` under `data/ace2005/`.

The train/dev/test split follows the standard used in Nguyen & Grishman (2015):
- **Test**: 40 newswire documents
- **Dev**: 30 documents
- **Train**: 529 remaining documents

## TAC KBP 2015

The TAC KBP 2015 Event Nugget Detection dataset is available through NIST:
https://tac.nist.gov/2015/KBP/

Once obtained, run:

```bash
python scripts/preprocess_tac2015.py \
    --tac_dir /path/to/tac2015 \
    --output_dir data/tac2015/
```

## Word Embeddings

The model uses 300-dimensional word2vec embeddings (Google News vectors):
https://code.google.com/archive/p/word2vec/

Download `GoogleNews-vectors-negative300.bin.gz`, decompress it, and place at:
```
data/embeddings/GoogleNews-vectors-negative300.bin
```

## Expected Directory Structure

```
data/
├── ace2005/
│   ├── train.json
│   ├── dev.json
│   └── test.json
├── tac2015/
│   ├── train.json
│   └── test.json
└── embeddings/
    └── GoogleNews-vectors-negative300.bin
```

## JSON Format

Each `.json` file contains one document per line. Each document has the form:

```json
{
  "id": "APW_ENG_20030322.0119-1",
  "tokens": ["The", "police", "fired", "tear", "gas", "."],
  "entity_mentions": [
    {"start": 1, "end": 2, "type": "ORG"}
  ],
  "event_mentions": [
    {
      "event_type": "Conflict:Attack",
      "trigger": {"start": 2, "end": 3}
    }
  ]
}
```

Token indices are 0-based; `start` is inclusive, `end` is exclusive.
