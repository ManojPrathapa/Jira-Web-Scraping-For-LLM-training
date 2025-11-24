import json
from pathlib import Path
from collections import Counter

p = Path("data/processed")
if not p.exists():
    print("No processed data found. Run transform first.")
    exit(0)

counts = 0
lengths = []
labels = Counter()
for f in p.glob("*.jsonl"):
    for line in f.open(encoding="utf8"):
        obj = json.loads(line)
        counts += 1
        lengths.append(len(obj.get("text","")))
        for l in obj.get("labels",[]):
            labels[l] += 1

print(f"Total records: {counts}")
if lengths:
    print(f"Avg text length: {sum(lengths)/len(lengths):.1f}")
    print(f"Max text length: {max(lengths)}")
    print(f"Min text length: {min(lengths)}")

print("Top labels:")
for k,v in labels.most_common(10):
    print(k, v)
