"""Dataset inventory and SHA-256 provenance; never modifies source data."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-dir',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); result={}
    for path in sorted(a.data_dir.rglob('*.tsv')):
        countries=Counter(); missing=Counter(); count=singletons=links=0
        with path.open(encoding='utf8',newline='') as f:
            for row in csv.DictReader(f,delimiter='\t'):
                count+=1
                for k,v in row.items():
                    if not v.strip(): missing[k]+=1
                if 'country' in row: countries[row['country']]+=1
                if 'matched_entity_ids' in row:
                    singletons+=int(not row['matched_entity_ids'])
                    links+=len(row['matched_entity_ids'].split(',')) if row['matched_entity_ids'] else 0
        with path.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
        entry={'rows':count,'countries':dict(countries),'empty_fields':dict(missing),'bytes':path.stat().st_size,'sha256':digest}
        if 'ground_truth' in path.name: entry.update(singletons=singletons,positive_links=links)
        result[str(path.relative_to(a.data_dir))]=entry
        print(path.name,json.dumps(entry),flush=True)
    a.output.write_text(json.dumps(result,indent=2))

if __name__=='__main__': main()
