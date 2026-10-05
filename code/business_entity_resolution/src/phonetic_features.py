"""Deterministic Unicode/phonetic features; no network or identity dictionary."""
import re
import unicodedata as ud
from functools import lru_cache
from itertools import combinations
import xxhash
from rapidfuzz import fuzz
import pipeline as base

INDIC = ('DEVANAGARI', 'BENGALI', 'GURMUKHI', 'GUJARATI', 'ORIYA', 'TAMIL', 'TELUGU', 'KANNADA', 'MALAYALAM')

@lru_cache(maxsize=60000)
def romanize(text):
    if text.isascii(): return ' '.join(re.findall(r'[^\W_]+', text.casefold()))
    out = []
    for ch in ud.normalize('NFC', text.casefold()):
        if ch.isascii():
            out.append(ch)
            continue
        label = ud.name(ch, '')
        if any(label.startswith(s + ' ') for s in INDIC):
            if 'VIRAMA' in label:
                if out and out[-1].endswith('a'): out[-1] = out[-1][:-1]
            elif 'VOWEL SIGN ' in label:
                if out and out[-1].endswith('a'): out[-1] = out[-1][:-1]
                out.append(re.sub(r'(vocalic|candra|short|long) ', '', label.split('VOWEL SIGN ')[-1].lower()))
            elif 'LETTER ' in label:
                out.append(re.sub(r'(vocalic|candra|short|long) ', '', label.split('LETTER ')[-1].lower()))
            elif 'ANUSVARA' in label or 'CANDRABINDU' in label: out.append('n')
            elif 'VISARGA' in label: out.append('h')
            elif ch.isdecimal(): out.append(str(ud.decimal(ch)))
        elif ud.category(ch) == 'Cf':
            continue
        else:
            out.append(''.join(c for c in ud.normalize('NFKD', ch) if not ud.combining(c)))
    return ' '.join(re.findall(r'[^\W_]+', ''.join(out)))

@lru_cache(maxsize=60000)
def phonetic(word):
    word = romanize(word)
    word = re.sub(r'c(?=[eiy])', 's', word)
    for a,b in [('sch','s'),('sh','s'),('ch','s'),('ph','f'),('kh','k'),('gh','g'),('th','t'),('dh','d'),('bh','b'),('c','k'),('q','k'),('v','w'),('z','j'),('x','ks')]:
        word = word.replace(a,b)
    word = re.sub('[aeiouy]', '', word)
    return re.sub(r'(.)\1+', r'\1', word)

LEGAL_PHON = {phonetic(t) for t in ('private','limited','corporation','incorporated','company','public')}

@lru_cache(maxsize=30000)
def forms(row):
    name = romanize(row[1]); addr = romanize(row[2])
    nt = tuple(t for t in name.split() if t not in base.LEGAL)
    pt = tuple(p for t in nt if (p:=phonetic(t)) and p not in LEGAL_PHON)
    return ' '.join(nt), ' '.join(pt), ''.join(pt), addr

def keys(row):
    n,p,compact,a = forms(row); country = base.norm(row[3]); out=set()
    def add(kind,value):
        if value: out.add(xxhash.xxh3_64_intdigest(country+'|v4|'+kind+'|'+value))
    add('pn',compact);add('ps',''.join(sorted(p.split())))
    tokens = sorted(set(p.split()),key=lambda t:(-len(t),t))[:5]
    for t in tokens:
        if len(t)>=3: add('pt',t)
    for t,u in combinations(tokens,2):add('pp','|'.join(sorted((t[:4],u[:4]))))
    if len(compact)>=8:
        add('pf',compact[:8]);add('pl',compact[-8:])
    # Rare romanized address-word pairs also survive a completely changed name.
    at=sorted(set(t for t in a.split() if len(t)>=5 and t.isalpha()),key=lambda t:(-len(t),t))[:4]
    for t,u in combinations(at,2):add('ap','|'.join(sorted((t,u))))
    return out

def features(left,right):
    lf,rf=forms(left),forms(right); vals=[]
    for x,y in zip(lf,rf):
        vals += [f(x,y)/100 for f in (fuzz.ratio,fuzz.partial_ratio,fuzz.token_sort_ratio,fuzz.token_set_ratio)]
        vals += [float(bool(x) and x==y)]
    # Preserve numeric order and alphanumeric units, unlike a bag of numbers.
    for raw_l,raw_r in [(left[2],right[2]),(left[1],right[1])]:
        a=re.findall(r'\d+[a-z]?(?:[-/]\d+[a-z]?)*',romanize(raw_l))
        b=re.findall(r'\d+[a-z]?(?:[-/]\d+[a-z]?)*',romanize(raw_r))
        vals += [float(bool(a) and bool(b) and a[0]==b[0]),float(bool(a) and bool(b) and a[0]!=b[0]),
                 float(bool(a) and a[0] in b),float(bool(b) and b[0] in a),len(set(a)-set(b)),len(set(b)-set(a))]
        vals += [fuzz.ratio(a[0],b[0])/100 if a and b else 0.]
    return vals

WIDTH=34
