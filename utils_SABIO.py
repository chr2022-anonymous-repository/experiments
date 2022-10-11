from tqdm import tqdm

import numpy as np
import pandas as pd

import requests as r
import json

import re


# NMvW DATA 

# reads the processed version of the NMvW data (= input data to SABIO)
# and parses it into a pandas.DataFrame
def get_NMvW_DF():
    df = pd.read_csv("v0_3.csv.gz")
    text_search_fields = ["Title", "Description", "Provenance", "Notes", "ObjectName"]
    df[text_search_fields] = df[text_search_fields].fillna("")
    df["Texts"] = df[text_search_fields].apply(lambda row: " ".join(row).lower(), axis=1)
    df = df.set_index("ID").sort_index()
    return df



# QUERYING THE SABIO BACKEND


voc = r"bewindhebber%2Cbewindvoerder%2Cbomba%2Cbombay%2Ccimarron%2Cderde%20wereld%2Cdwerg%2Cexpeditie%2Cgouverneur%2Chalfbloed%2Chottentot%2Cinboorling%2Cindiaan%2Cindisch%2Cindo%2Cinheems%2Cinlander%2Cjap%2Cjappen%2Cjappenkampen%2Ckaffer%2Ckaffir%2Ckafir%2Ckoelie%2Ckolonie%2Clagelonenland%2Clandhuis%2Cmarron%2Cmarronage%2Cmissie%2Cmissionaris%2Cmoor%2Cmoors%2Cmoren%2Cmulat%2Coctroon%2Contdekken%2Contdekking%2Contdekkingsreis%2Contwikkelingsland%2Coorspronkelijk%2Coosters%2Copperhoofd%2Cori%C3%ABntaals%2Cpinda%2Cpolitionele%20actie%2Cprimitief%2Cprimitieven%2Cpygmee%2Cras%2Crasch%2Cslaaf%2Cstam%2Cstamhoofd%2Ctraditioneel%2Ctropisch%2Cwesters%2Cwilden%2Czendeling%2Czendelingen%2Czending&object_param_Classification=&object_param_Department="



# queries the backend for THE ENTIRE DATASET for the scores of the given engine
# returns a pandas.Series with the scores (index are object's IDs)
def get_scores(engine):
    base_url = f"""https://sabio.diginfra.net/api/v1/objects/NMvW_v0/search?object_keywords=&object_start_date=&object_end_date=&engine_id={engine}&engine_min_score=0&engine_max_score=1&vocabulary_terms=&object_param_Classification=&object_param_Department="""
    search = r.get(base_url).content
    attrs = json.loads(search)["attributes"]
    results = json.loads(search)["results"]
    df = pd.DataFrame.from_records(results).set_index("id")
    df.index = df.index.astype("int")
    df = df.sort_index()
    return df.score


def get_scores_all_engines():
    engine_ids = ("ContentLengthEnginev0", "VocabularyEnginev0", 
                  "TypicalityEnginev0", "RandomEnginev0", "PMIEnginev0")
    
    scores = [get_scores(e) for e in tqdm(engine_ids)]
    
    df = pd.concat(scores, axis=1).fillna(0.)
    df.columns = engine_ids
    
    return df
