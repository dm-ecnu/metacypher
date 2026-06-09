import json
import os
from pathlib import Path

# Data dir is env-overridable (METACYPHER_SUBGRAPH_DIR / PROJECT_ROOT); defaults
# to <repo>/data/subgraph. See ../config.py.
_SUBGRAPH = Path(os.environ.get("METACYPHER_SUBGRAPH_DIR") or
                 (Path(os.environ.get("PROJECT_ROOT") or os.environ.get("METACYPHER_DATA_DIR") or
                       (Path(__file__).resolve().parents[2] / "data")) / "subgraph"))
infile = str(_SUBGRAPH / 'final' / 'mtq_correction.filtered.jsonl')
outfile = str(_SUBGRAPH / 'final' / 'mtq_schema.jsonl')

err = 0
with open(infile, 'r') as f:
    data = [json.loads(line) for line in f]
    # Process the data as needed

for item in data:
    ana = item.get('analysis',{})
    rs = ana.get('related_schema', {})
    if rs == {}:
        err+=1
        data.remove(item)

with open(outfile, 'w') as f:
    for item in data:
        f.write(json.dumps(item) + '\n')

print(err)