from pathlib import Path
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
import numpy as np
import sgt
print(sgt.__version__)
from sgt import SGT
from joblib import Parallel, delayed
import joblib
import sys

proteins = pd.read_table(Path("data/uniprotkb_reviewed_true_2025_09_21.tsv"))


# Dropping last n rows using drop

proteins = proteins.rename(columns={'Entry': 'id', 'Sequence': 'sequence'})
proteins = proteins[proteins["sequence"].str.contains("B") == False]
proteins = proteins[proteins["sequence"].str.contains("Z") == False]
proteins = proteins[proteins["sequence"].str.contains("U") == False]
proteins = proteins[proteins["sequence"].str.contains("X") == False]

# leave only 10000 sequences so it doesn't take too long
proteins.drop(proteins.tail(len(proteins)-10000).index,
        inplace = True)

proteins['sequence'] = proteins['sequence'].map(list)
print(proteins)




sgt_ = SGT(kappa=1, 
        lengthsensitive=False, 
        mode='default')
sgtembedding_df = sgt_.fit_transform(proteins)
x = sgtembedding_df.set_index('id')

joblib.dump(sgt_, 'SGT_all.pkl')