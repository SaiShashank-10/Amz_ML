"""Post-assessment error inspection; does not change the model or predictions."""
import argparse
import json
from collections import Counter
from pathlib import Path
from pipeline import records, features

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-dir',type=Path,required=True); p.add_argument('--work-dir',type=Path,required=True)
    a=p.parse_args(); errors=json.loads((a.work_dir/'validation_errors.json').read_text())
    refs={r[0]:tuple(r) for r in json.loads((a.work_dir/'sample.json').read_text())}
    needed={e['target'] for e in errors}; targets={}
    for source in (2,3):
        for row in records(a.data_dir/'train'/f'train_source{source}.tsv'):
            if row[0] in needed: targets[row[0]]=row
    assert needed<=targets.keys()
    summary={'false_positive':Counter(),'scored_false_negative':Counter()}; examples={'false_positive':[],'scored_false_negative':[]}
    for e in errors:
        l=refs[e['source1_entity_id']]; r=targets[e['target']]; x=features(l,r)
        kind='scored_false_negative' if e['label'] else 'false_positive'
        conditions={'missing_target_address':bool(x[45]),'high_name_similarity':x[10]>=.9,
            'high_address_similarity':x[17]>=.9,'numeric_disagreement':x[39]+x[40]>0,
            'weak_name_similarity':x[10]<.5}
        summary[kind].update(k for k,v in conditions.items() if v); summary[kind]['total']+=1
        if len(examples[kind])<12:
            examples[kind].append({**e,'source1_name':l[1],'source1_address':l[2],
                'target_name':r[1],'target_address':r[2],'country':l[3],
                'core_name_similarity':x[10],'address_similarity':x[17]})
    result={'note':'Categories overlap; these are post-assessment descriptions, not definitive error causes. Blocking misses are not included among scored errors.',
        'counts':{k:dict(v) for k,v in summary.items()},'examples':examples}
    (a.work_dir/'error_analysis.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result['counts'],indent=2))

if __name__=='__main__': main()
