import requests
import json


# FIX: sometimes, synonyms come with spelling variations\ (comma-separated) -> pull apart
def diamant_synonyms(lemma, lower=True):
    r = requests.get(f"https://rest.diamant.ivdnt.org/query?word={lemma}&timelineStart=1200&timelineEnd=2022")
    if not r.ok:
        raise ValueError("Request failed!")
    j = json.loads(r.text)
    if "@graph" not in j:
        raise ValueError("No results for this query.")
    to_lower = lambda s: s.lower() if lower else s
    return [to_lower(d["writtenRep"]) for d in j["@graph"] if "writtenRep" in d]



import xml.etree.ElementTree as ET

def INT_wordforms(lemma, only_whole_words=False):
    db = "lexiconservice_mnw_wnt"
    url = f"http://sk.taalbanknederlands.inl.nl/LexiconService/lexicon/get_wordforms_from_lemma?database={db}&lemma={lemma}&case_sensitive=false"
    result = requests.get(url)
    if not result.ok:
        raise ValueError(f"'{lemma}' didn't not result in a valid response.")
    root = ET.fromstring(result.text)

    return sorted(set([el.text for el in root.findall(".//found_wordforms")] + [lemma]))


import re

# assumes that `raw_vocab` is a string of comma-separated terms
def vocab2re(raw_vocab, only_whole_words=False):
    vocab_parser = re.compile("\s*,\s*")
    v_ls = vocab_parser.split(raw_vocab.strip())
    f = lambda w: rf"\b{w}\b" if only_whole_words else rf"{w}"
    return re.compile("|".join(f(w) for w in v_ls))

# assumes that vocab_re = "\bword1\b|\bword2\b|..."
def re2vocab(vocab_re):
    return vocab_re.pattern.replace("|", ",").replace(r"\b", "")




from tqdm import tqdm
import swifter
tqdm.pandas()
def find_lemmata(lemmata, texts, only_whole_words=True, return_forms=False, parallel=True):
    if isinstance(lemmata, str):
        lemmata = [lemmata]
    
    forms = {w.lower() for l in tqdm(lemmata, desc="getting wordforms from INT (internet up?)")
             for w in INT_wordforms(l)}
    regex = vocab2re(re.escape(",".join(forms)), only_whole_words=only_whole_words)
    
    
    text_series = texts.swifter if parallel else texts
    f = text_series.apply if parallel else text_series.progress_apply
    
        
    if return_forms:
            return f(regex.findall), forms
    
    return f(regex.findall)


import spacy

nlp = spacy.load("nl_core_news_sm")

def spacy_lemmata(words):
    t = nlp(words)
    
    return {str(tok): tok.lemma_ for tok in t}




##################################################################################
### MISC


def zipf_plot(found_ls):
    tok_counts = Counter([w for ls in found_ls for w in ls])
    rs, cs = list(zip(*[(r, c) for r, (w, c) in enumerate(tok_counts.most_common(), 1)]))

    plt.figure(figsize=(20, 10))
    plt.plot(rs, cs, ".")
    for r, (w, c) in enumerate(tok_counts.most_common(), 1):
        plt.annotate(w, xy=(r, c), )