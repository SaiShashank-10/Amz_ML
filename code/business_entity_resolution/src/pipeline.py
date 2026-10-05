"""Offline, bounded-memory business entity resolution. No network access."""
import argparse
import csv
import hashlib
import itertools
import json
import mmap
import os
import re
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from collections import Counter
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
import xxhash
from rapidfuzz import fuzz
from sklearn.model_selection import train_test_split

DT = np.dtype([('key', '<u8'), ('offset', '<u8')])
SEED = 20260926
LEGAL = set('inc incorporated corporation corp llc llp ltd limited private pvt company co plc pc the and of sa sas sarl eurl'.split())
ABBR = dict(zip('road street avenue boulevard drive lane court apartment suite floor building highway private limited corporation incorporated centre'.split(),
                'rd st ave blvd dr ln ct apt ste fl bldg hwy pvt ltd corp inc center'.split()))
FEATURE_NAMES = [f'{field}_{metric}' for field in ('name','core_name','address')
    for metric in ('ratio','partial','token_sort','token_set','weighted_ratio','exact','length_ratio')]
FEATURE_NAMES += [f'{field}_{metric}' for field in ('name_tokens','address_tokens','numbers')
    for metric in ('jaccard','overlap_min','coverage_left','coverage_right')]
FEATURE_NAMES += ['name_tokens_left','name_tokens_right','address_tokens_left','address_tokens_right',
    'numbers_left','numbers_right','numbers_only_left','numbers_only_right','same_country',
    'missing_left_name','missing_right_name','missing_left_address','missing_right_address',
    'left_token_alignment_min','left_token_alignment_mean','right_token_alignment_min','right_token_alignment_mean']

def log(s):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), s, flush=True)

@lru_cache(maxsize=100000)
def norm(s):
    s = s.casefold()
    if not s.isascii():
        s = ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
    return ' '.join(ABBR.get(t, t) for t in re.findall(r'[^\W_]+', s.replace('&', ' and ')))

@lru_cache(maxsize=80000)
def prep(row):
    ident, name, addr, country = row
    n, a = norm(name), norm(addr)
    tokens = tuple(sorted(set(t for t in n.split() if t not in LEGAL)))
    if not tokens:
        tokens = tuple(sorted(set(n.split())))
    at = frozenset(a.split())
    nums = tuple(sorted(set(re.findall(r'\d+', a))))
    return n, a, tokens, at, nums, norm(country)

def keys(row):
    n, a, nt, at, nums, country = prep(row)
    k = set()
    def add(kind, value):
        if value:
            k.add(xxhash.xxh3_64_intdigest(country + '|' + kind + '|' + value))
    add('name', n)
    add('core', ' '.join(nt))
    add('prefix', ' '.join(sorted(t[:4] for t in nt)))
    add('addr', ' '.join(sorted(at)))
    # Token combinations survive word reordering and extra legal/trade words.
    for x, y in itertools.combinations(sorted(nt, key=lambda t: (-len(t), t))[:5], 2):
        add('pair', ' '.join(sorted((x[:5], y[:5]))))
    strong = sorted((t for t in at if t.isalpha() and len(t) >= 4), key=lambda t: (-len(t), t))[:2]
    for t in sorted(nt, key=lambda t: (-len(t), t))[:4]:
        if len(t) < 3:
            continue
        for z in nums[:4]:
            add('num', t[:4] + '|' + z)
        for z in strong:
            add('local', t[:4] + '|' + z[:5])
    return k

def extra_keys(row):
    n, a, nt, at, nums, country = prep(row)
    k = set()
    def add(kind, value):
        if value:
            k.add(xxhash.xxh3_64_intdigest(country+'|extra|'+kind+'|'+value))
    add('compact', ''.join(nt))
    add('compact', ''.join(t for t in n.split() if t not in LEGAL))
    long_tokens = sorted((t for t in at if t.isalpha() and len(t)>=4), key=lambda t:(-len(t),t))[:4]
    for t in long_tokens[:3]:
        for number in nums[:3]:
            add('addrnum', t[:6]+'|'+number)
    for t,u in itertools.combinations(long_tokens,2):
        add('addrpair', '|'.join(sorted((t[:6],u[:6]))))
    return k

def build_extra_index(dest):
    extra=dest/'extra'; extra.mkdir(exist_ok=True)
    done=extra/'complete.json'
    if done.exists(): return
    files=[open(extra/f'{i:02x}.bin','wb') for i in range(256)]
    buffers=[[] for _ in range(256)]; count=0
    def flush():
        for f,b in zip(files,buffers):
            if b: np.asarray(b,dtype=DT).tofile(f); b.clear()
    with open(dest/'records.jsonl','rb') as stream:
        while True:
            offset=stream.tell(); line=stream.readline()
            if not line: break
            row=tuple(json.loads(line))
            for k in extra_keys(row): buffers[k&255].append((k,offset))
            count+=1
            if count%20000==0: flush()
            if count%1000000==0: log(f'{dest.name} supplementary indexing: {count:,}')
    flush()
    for f in files: f.close()
    prep.cache_clear()
    for i in range(256):
        p=extra/f'{i:02x}.bin'; arr=np.fromfile(p,dtype=DT); arr.sort(order=['key','offset']); arr.tofile(p)
    done.write_text(json.dumps({'records':count,'version':2}))
    log(f'{dest.name} supplementary index complete')

def records(path):
    with open(path, encoding='utf-8', newline='') as f:
        r = csv.reader(f, delimiter='\t')
        header = next(r)
        assert header == ['entity_id', 'business_name', 'business_address', 'country'], header
        for row in r:
            assert len(row) == 4
            yield tuple(row)

def fingerprint(paths):
    return [{'path': str(p.resolve()), 'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in paths]

def build_index(data, split, work):
    paths = [data / split / f'{split}_source{s}.tsv' for s in (2, 3)]
    dest = work / f'{split}_index'
    dest.mkdir(parents=True, exist_ok=True)
    fp = fingerprint(paths)
    done = dest / 'complete.json'
    if done.exists():
        meta = json.loads(done.read_text())
        assert meta['inputs'] == fp, 'Input data changed; use a fresh work directory.'
        build_extra_index(dest)
        return dest
    # Canonical line store permits random access even if original TSV has quoted fields.
    files = [open(dest / f'{i:02x}.bin', 'wb') for i in range(256)]
    buffers = [[] for _ in range(256)]
    count = 0
    def flush():
        for h, b in zip(files, buffers):
            if b:
                np.asarray(b, dtype=DT).tofile(h)
                b.clear()
    with open(dest / 'records.jsonl', 'wb') as store:
        for path in paths:
            for row in records(path):
                offset = store.tell()
                store.write((json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8'))
                for k in keys(row):
                    buffers[k & 255].append((k, offset))
                count += 1
                if count % 20000 == 0:
                    flush()
                if count % 250000 == 0:
                    log(f'{split} indexing: {count:,} records')
        flush()
    for h in files:
        h.close()
    prep.cache_clear()
    for i in range(256):
        p = dest / f'{i:02x}.bin'
        arr = np.fromfile(p, dtype=DT)
        arr.sort(order=['key', 'offset'])
        arr.tofile(p)
        if i % 64 == 0:
            log(f'{split} sorting index shard {i}/256')
    done.write_text(json.dumps({'inputs': fp, 'records': count}))
    build_extra_index(dest)
    return dest

class Index:
    def __init__(self, dest, max_block=100, max_candidates=240):
        self.shards = [np.memmap(dest / f'{i:02x}.bin', dtype=DT, mode='r') if (dest / f'{i:02x}.bin').stat().st_size else np.empty(0, dtype=DT) for i in range(256)]
        self.extra_shards = [np.memmap(dest/'extra'/f'{i:02x}.bin',dtype=DT,mode='r') if (dest/'extra'/f'{i:02x}.bin').stat().st_size else np.empty(0,dtype=DT) for i in range(256)]
        self.file = open(dest / 'records.jsonl', 'rb')
        self.store = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self.max_block, self.max_candidates = max_block, max_candidates

    def get(self, offset):
        offset = int(offset)
        end = self.store.find(b'\n', offset)
        return tuple(json.loads(self.store[offset:end]))

    def candidates(self, rows):
        requests = [[] for _ in range(512)]
        counts = [Counter() for _ in rows]
        for i, row in enumerate(rows):
            for k in keys(row):
                requests[k & 255].append((k, i))
            for k in extra_keys(row):
                requests[256+(k&255)].append((k,i))
        for shard, req in zip(self.shards+self.extra_shards, requests):
            if not req:
                continue
            ks = np.array([x[0] for x in req], dtype=np.uint64)
            lo = np.searchsorted(shard['key'], ks, side='left')
            hi = np.searchsorted(shard['key'], ks, side='right')
            for (_, qi), start, end in zip(req, lo, hi):
                if 0 < end - start <= self.max_block:
                    counts[qi].update(int(x) for x in shard['offset'][start:end])
        return [sorted(c, key=lambda x: (-c[x], x))[:self.max_candidates] for c in counts]

def overlap(a, b):
    a, b = set(a), set(b)
    inter = len(a & b)
    return [inter / max(1, len(a | b)), inter / max(1, min(len(a), len(b))), inter / max(1, len(a)), inter / max(1, len(b))]

def features(left, right):
    n, a, nt, at, nums, c = prep(left)
    m, b, mt, bt, digs, d = prep(right)
    nc, mc = ' '.join(nt), ' '.join(mt)
    vals = []
    for x, y in ((n, m), (nc, mc), (a, b)):
        vals.extend(f(x, y) / 100 for f in (fuzz.ratio, fuzz.partial_ratio, fuzz.token_sort_ratio, fuzz.token_set_ratio, fuzz.WRatio))
        vals += [float(x == y and bool(x)), min(len(x), len(y)) / max(1, len(x), len(y))]
    vals += overlap(nt, mt) + overlap(at, bt) + overlap(nums, digs)
    vals += [len(nt), len(mt), len(at), len(bt), len(nums), len(digs), len(set(nums) - set(digs)), len(set(digs) - set(nums))]
    vals += [float(c == d), float(not n), float(not m), float(not a), float(not b)]
    # Best token edit alignments, useful for localized spelling damage.
    for x, y in ((nt, mt), (mt, nt)):
        sim = [max((fuzz.ratio(t, u) / 100 for u in y), default=0) for t in x]
        vals += [min(sim, default=0), sum(sim) / max(1, len(sim))]
    return vals

def candidate_keep(name_similarity, address_similarity, missing_address):
    return ((name_similarity >= .9 - 1e-7) | (address_similarity >= .85 - 1e-7)
        | ((name_similarity >= .35 - 1e-7) & (address_similarity >= .5 - 1e-7))
        | (missing_address & (name_similarity >= .5 - 1e-7)))

def cheap_filter(left, right):
    _, a, nt, *_ = prep(left)
    _, b, mt, *_ = prep(right)
    return candidate_keep(fuzz.token_set_ratio(' '.join(nt), ' '.join(mt))/100,
        fuzz.token_set_ratio(a,b)/100, not a or not b)

def batch_pairs(rows, index):
    cs = index.candidates(rows)
    feats, qi, targets = [], [], []
    for i, (row, offsets) in enumerate(zip(rows, cs)):
        for off in offsets:
            target = index.get(off)
            if not cheap_filter(row, target):
                continue
            feats.append(features(row, target))
            qi.append(i)
            targets.append(target[0])
    return np.asarray(feats, dtype=np.float32).reshape(-1, 50), np.asarray(qi, dtype=np.int32), targets

def load_truth(data, ids):
    truth = {}
    with open(data / 'train/train_ground_truth.tsv', encoding='utf8', newline='') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            if r['source1_entity_id'] in ids:
                truth[r['source1_entity_id']] = set(filter(None, r['matched_entity_ids'].split(',')))
    assert len(truth) == len(ids)
    return truth

def init_feature_worker(index_path):
    global WORKER_INDEX
    WORKER_INDEX = Index(Path(index_path))

def extract_training_chunk(task):
    rows,start,truth=task
    x,q,t=batch_pairs(rows,WORKER_INDEX)
    y=np.array([tid in truth[rows[int(i)][0]] for i,tid in zip(q,t)],dtype=np.uint8)
    return x,y,q+start,t

def prepare_training(data, work, sample_size):
    path = work / 'training_v2.npz'
    if path.exists():
        return
    # Reservoir sampling is uniform across the entire reference file, never ID digits as features.
    rng = np.random.default_rng(SEED)
    sample = []
    for i, row in enumerate(records(data / 'train/train_source1.tsv')):
        if i < sample_size:
            sample.append(row)
        else:
            j = int(rng.integers(i + 1))
            if j < sample_size:
                sample[j] = row
    truth = load_truth(data, {r[0] for r in sample})
    (work / 'sample.json').write_text(json.dumps(sample))
    (work / 'truth.json').write_text(json.dumps({k: sorted(v) for k, v in truth.items()}))
    index_path = build_index(data, 'train', work)
    xs, ys, groups, tids = [], [], [], []
    tasks=[(sample[start:start+200],start,{r[0]:truth[r[0]] for r in sample[start:start+200]}) for start in range(0,len(sample),200)]
    with ProcessPoolExecutor(max_workers=2,initializer=init_feature_worker,initargs=(str(index_path),)) as pool:
        for chunk,(x,y,q,t) in enumerate(pool.map(extract_training_chunk,tasks)):
            xs.append(x); ys.append(y); groups.append(q); tids.extend(t)
            if chunk%10==0:
                log(f'Training features {min((chunk+1)*200,len(sample)):,}/{len(sample):,}; pairs {sum(len(a) for a in ys):,}')
    np.savez(path, X=np.concatenate(xs), y=np.concatenate(ys), group=np.concatenate(groups), target=np.array(tids))
    log('Training features saved')

def score(groups, targets, prob, threshold, selected, truth_sets):
    mask = prob >= threshold
    pred = [set() for _ in truth_sets]
    for g, t in zip(groups[mask], targets[mask]):
        pred[int(g)].add(str(t))
    scores = []; tp = fp = fn = single = correct_single = 0
    for i in selected:
        a, b = truth_sets[i], pred[i]
        good = len(a & b)
        tp += good; fp += len(b - a); fn += len(a - b)
        scores.append(1.0 if not a and not b else 1.25 * good / (.25 * len(a) + len(b)) if a or b else 0)
        if not a:
            single += 1; correct_single += int(not b)
    mean=float(np.mean(scores)); se=float(np.std(scores,ddof=1)/np.sqrt(len(scores))) if len(scores)>1 else 0.
    return {'macro_f0.5': mean, 'approximate_95pct_ci': [max(0.,mean-1.96*se),min(1.,mean+1.96*se)], 'micro_precision': tp / max(1, tp + fp), 'micro_recall': tp / max(1, tp + fn), 'tp': tp, 'fp': fp, 'fn': fn, 'entities': len(selected), 'singletons': single, 'correct_singletons': correct_single}

def new_model(trees=450):
    return lgb.LGBMClassifier(n_estimators=trees, learning_rate=.045, num_leaves=31, max_depth=-1,
        min_child_samples=60, colsample_bytree=.9, reg_lambda=3, n_jobs=8,
        random_state=SEED, deterministic=True, force_col_wise=True, verbosity=-1)

def train(work):
    z = np.load(work / 'training_v2.npz')
    x, y, group, target = (z[k] for k in ('X', 'y', 'group', 'target'))
    # Also accepts caches produced before the final cheap blocking filter.
    keep = candidate_keep(x[:,10], x[:,17], (x[:,44] + x[:,45]) > 0)
    x, y, group, target = x[keep], y[keep], group[keep], target[keep]
    sample = json.loads((work / 'sample.json').read_text())
    truth = json.loads((work / 'truth.json').read_text())
    truth_sets = [set(truth[r[0]]) for r in sample]
    strata = [r[3] + ('|single' if not t else '|matched') for r, t in zip(sample, truth_sets)]
    fit, rest = train_test_split(np.arange(len(sample)), test_size=.4, random_state=SEED, stratify=strata)
    tune, audit = train_test_split(rest, test_size=.5, random_state=SEED, stratify=np.array(strata)[rest])
    masks = [np.isin(group, a) for a in (fit, tune, audit)]
    model = new_model()
    log(f'Actual supervised training: {masks[0].sum():,} pairs; {int(y[masks[0]].sum()):,} positive pairs')
    model.fit(x[masks[0]], y[masks[0]], feature_name=FEATURE_NAMES, eval_set=[(x[masks[1]], y[masks[1]])], callbacks=[lgb.log_evaluation(50)])
    p = model.predict_proba(x[masks[1]])[:, 1]
    trials = []
    for threshold in np.arange(.10, .951, .025):
        s = score(group[masks[1]], target[masks[1]], p, threshold, tune, truth_sets)
        trials.append((s['macro_f0.5'], float(threshold), s))
    _, threshold, tuning = max(trials, key=lambda v: (v[0], v[1]))
    ap = model.predict_proba(x[masks[2]])[:, 1]
    audit_metrics = score(group[masks[2]], target[masks[2]], ap, threshold, audit, truth_sets)
    metrics = {'seed': SEED, 'sample_entities': len(sample), 'training_pairs': len(y), 'features': x.shape[1],
        'threshold': threshold, 'tuning': tuning, 'held_out': audit_metrics,
        'candidate_recall': int(y.sum()) / sum(map(len, truth_sets)),
        'threshold_trials': [{'threshold': t, **s} for _, t, s in trials]}
    metrics['held_out_blocking_ceiling'] = score(group[masks[2]], target[masks[2]], y[masks[2]], .5, audit, truth_sets)
    metrics['held_out_empty_baseline'] = sum(not truth_sets[i] for i in audit)/len(audit)
    metrics['held_out_by_country'] = {c: score(group[masks[2]], target[masks[2]], ap, threshold,
        [i for i in audit if sample[i][3] == c], truth_sets) for c in sorted({r[3] for r in sample})}
    errors = []
    for g, t, prob in zip(group[masks[2]], target[masks[2]], ap):
        actual = str(t) in truth_sets[int(g)]
        if (prob >= threshold) != actual:
            errors.append({'source1_entity_id': sample[int(g)][0], 'target': str(t), 'probability': float(prob), 'label': actual})
    (work / 'validation_errors.json').write_text(json.dumps(errors, indent=2))
    (work / 'splits.json').write_text(json.dumps({k: [sample[i][0] for i in ids] for k, ids in [('fit', fit), ('tune', tune), ('audit', audit)]}))
    log('Held-out assessment: ' + json.dumps(audit_metrics))
    # Diagnostic only: train on one country and assess another without exposing
    # that country's labels to the classifier. This is not a France score.
    metrics['country_transfer_diagnostics'] = {}
    for held_country in sorted({r[3] for r in sample}):
        transfer_fit = [i for i in fit if sample[i][3] != held_country]
        transfer_tune = [i for i in tune if sample[i][3] != held_country]
        transfer_eval = [i for i in audit if sample[i][3] == held_country]
        if not transfer_fit or not transfer_eval:
            continue
        fm, em = np.isin(group, transfer_fit), np.isin(group, transfer_eval)
        transfer_model = new_model()
        log(f'Country-transfer diagnostic: withholding {held_country} from model fitting')
        transfer_model.fit(x[fm], y[fm], feature_name=FEATURE_NAMES)
        tm = np.isin(group, transfer_tune)
        tune_prob = transfer_model.predict_proba(x[tm])[:, 1]
        transfer_threshold = max(np.arange(.10, .951, .025), key=lambda t:
            (score(group[tm], target[tm], tune_prob, t, transfer_tune, truth_sets)['macro_f0.5'], t))
        transfer_prob = transfer_model.predict_proba(x[em])[:, 1]
        metrics['country_transfer_diagnostics'][held_country] = score(group[em], target[em], transfer_prob, transfer_threshold, transfer_eval, truth_sets)
        metrics['country_transfer_diagnostics'][held_country]['threshold'] = float(transfer_threshold)
        log('Country-transfer result: ' + json.dumps(metrics['country_transfer_diagnostics'][held_country]))
    log('Refitting final model on all sampled training entities')
    final = new_model()
    final.fit(x, y, feature_name=FEATURE_NAMES, callbacks=[lgb.log_evaluation(50)])
    final.booster_.save_model(str(work / 'model.txt'))
    metrics['tree_count'] = final.booster_.num_trees()
    metrics['total_leaves'] = sum(t['num_leaves'] for t in final.booster_.dump_model()['tree_info'])
    metrics['feature_importance_gain'] = dict(zip(FEATURE_NAMES, map(float, final.booster_.feature_importance(importance_type='gain'))))
    (work / 'metrics.json').write_text(json.dumps(metrics, indent=2))

def init_worker(index_path, model_path, threshold):
    global WORKER_INDEX, WORKER_MODEL, WORKER_THRESHOLD, norm, prep
    # Smaller per-process caches leave RAM available for shared memory-mapped indexes.
    norm = lru_cache(maxsize=30000)(norm.__wrapped__)
    prep = lru_cache(maxsize=20000)(prep.__wrapped__)
    WORKER_INDEX = Index(Path(index_path))
    WORKER_MODEL = lgb.Booster(model_file=model_path)
    WORKER_THRESHOLD = threshold

def predict_chunk(rows, filename):
    x, q, targets = batch_pairs(rows, WORKER_INDEX)
    probs = WORKER_MODEL.predict(x, num_threads=1) if len(x) else []
    candidates, matches = [[] for _ in rows], [[] for _ in rows]
    for i, tid, prob in zip(q, targets, probs):
        candidates[int(i)].append(tid)
        if prob >= WORKER_THRESHOLD:
            matches[int(i)].append(tid)
    payload = [(r[0], sorted(set(c)), sorted(set(m))) for r, c, m in zip(rows, candidates, matches)]
    file = Path(filename); temp = file.with_suffix('.tmp')
    temp.write_text(json.dumps(payload)); temp.replace(file)
    return len(rows)

def predict(data, work, output, workers=4):
    output.mkdir(parents=True, exist_ok=True)
    index_path = build_index(data, 'test', work)
    threshold = json.loads((work / 'metrics.json').read_text())['threshold']
    # Separate atomic batch files make long inference safely resumable.
    chunks = work / 'prediction_chunks'; chunks.mkdir(exist_ok=True)
    signature = {'model_sha256': hashlib.sha256((work/'model.txt').read_bytes()).hexdigest(),
        'threshold': threshold, 'source1': fingerprint([data/'test/test_source1.tsv']),
        'index': json.loads((work/'test_index/complete.json').read_text()),
        'max_block': 100, 'max_candidates': 240, 'cheap_filter_version': 1, 'index_version': 2}
    manifest = chunks/'manifest.json'
    if manifest.exists():
        assert json.loads(manifest.read_text()) == signature, 'Prediction configuration changed; use a fresh prediction_chunks directory.'
    else:
        manifest.write_text(json.dumps(signature, indent=2))
    iterator = records(data / 'test/test_source1.tsv')
    batches = 0; total = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker,
            initargs=(str(index_path), str(work/'model.txt'), threshold)) as pool:
        pending = set()
        while rows := list(itertools.islice(iterator, 500)):
            file = chunks / f'{batches:06d}.json'
            if file.exists():
                total += len(rows)
            else:
                pending.add(pool.submit(predict_chunk, rows, str(file)))
            batches += 1
            if len(pending) >= workers * 2:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    total += future.result()
                if total % 5000 == 0:
                    log(f'Test inference: {total:,} reference entities complete')
        for future in pending:
            total += future.result()
    stats = {'entities': 0, 'candidates': 0, 'matches': 0, 'singletons': 0}
    with open(output / 'matching_results.tsv', 'w', encoding='utf8', newline='') as fm, open(output / 'candidate_pairs.tsv', 'w', encoding='utf8', newline='') as fc:
        wm, wc = csv.writer(fm, delimiter='\t', lineterminator='\n'), csv.writer(fc, delimiter='\t', lineterminator='\n')
        wm.writerow(['source1_entity_id', 'matched_entity_ids']); wc.writerow(['source1_entity_id', 'candidate_entity_ids'])
        for i in range(batches):
            for sid, candidates, matches in json.loads((chunks / f'{i:06d}.json').read_text()):
                assert set(matches) <= set(candidates)
                wm.writerow([sid, ','.join(matches)]); wc.writerow([sid, ','.join(candidates)])
                stats['entities'] += 1; stats['candidates'] += len(candidates); stats['matches'] += len(matches); stats['singletons'] += int(not matches)
    (work / 'prediction_stats.json').write_text(json.dumps(stats, indent=2))
    log('Inference complete: ' + json.dumps(stats))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--work-dir', type=Path, default=Path('artifacts'))
    p.add_argument('--output-dir', type=Path, default=Path('output'))
    p.add_argument('--sample-size', type=int, default=30000)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--stage', choices=['all', 'prepare', 'train', 'predict'], default='all')
    a = p.parse_args(); a.work_dir.mkdir(parents=True, exist_ok=True)
    if a.stage in ('all', 'prepare'):
        prepare_training(a.data_dir, a.work_dir, a.sample_size)
    if a.stage in ('all', 'train'):
        train(a.work_dir)
    if a.stage in ('all', 'predict'):
        predict(a.data_dir, a.work_dir, a.output_dir, a.workers)

if __name__ == '__main__':
    main()
