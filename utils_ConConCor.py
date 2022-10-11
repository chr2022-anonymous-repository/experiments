from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# transform CCC responses to categorical variables
def annotation_to_var(ann):
    if ann == "Niet omstreden": return 0
    if ann == "Omstreden naar huidige maatstaven": return 1
    if ann == "Weet ik niet": return 2
    return 3

# transform CCC user IDs to categorical variables
def annotator_ids_to_var(ids):
    num_ids = dict(zip(sorted(ids), range(len(ids))))    
    return ids.apply(lambda i: num_ids[i])


# remove the target word from a given snippet
def remove_target_from_context(row):
    return row.text.replace(row.target_compound_bolded, " ") #"[MASK]")

# load composite CSV files that make up CCC and parse/preprocess them into a 
# pandas.DataFrame
# arg `filter_level`: filter out ConConCor samples with response level > filter_level
#                     (see annotations_to_var for the meaning of levels) 
def get_CCC_DF(filter_level=1):
    extr = pd.read_csv("ConConCor_data/Extracts.csv").set_index("extract_id")
    ann = pd.read_csv("ConConCor_data/Annotations.csv")
    
    data = pd.concat([
            ann[["response", "anonymised_participant_id"]], 
            extr.loc[ann.extract_id][["target", "text", "target_compound_bolded"]].reset_index()
                 ], axis=1).set_index("extract_id")

    data["y"] = data.response.apply(annotation_to_var)
    data = data[data.y <= filter_level]
    
    data["annotator_x"] = annotator_ids_to_var(data.anonymised_participant_id)
    data["context_wo_target"] = data.apply(remove_target_from_context, axis=1)
    return data



def binH(q):
    return -(q*np.log2(q) + (1-q)*np.log2(1-q))

def vec_binH(v):
    p = v.mean()
    if p == 0. or p == 1.: 
        return 0
    return binH(p)

