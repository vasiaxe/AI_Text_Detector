# Data

Raw data files are not committed to this repository.

Expected structure:

```text
data/raw/train.jsonl
data/raw/val.jsonl
data/raw/test.jsonl
```

Each file should contain one JSON object per line with at least:

```json
{"text": "Example text...", "label": 0}
```

Labels:

```text
0 = human-written
1 = AI-generated
```

The dataset is excluded from GitHub because of file size, availability possible through messaging me.