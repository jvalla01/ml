import sgt
print(sgt.__version__)
from sgt import SGT

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

#First trying steps from tutorial before applying to data
#SGT embedding tutorial

# Learning a sgt embedding as a matrix with 
# rows and columns as the sequence alphabets. 
# This embedding shows the relationship between 
# the alphabets. The higher the value the 
# stronger the relationship.

sgt = SGT(flatten=False)
sequence = np.array(["B","B","A","C","A","C","A","A","B","A"])
sgt.fit(sequence)


# A sample corpus of two sequences.
corpus = pd.DataFrame([[1, ["B","B","A","C","A","C","A","A","B","A"]], 
                       [2, ["C", "Z", "Z", "Z", "D"]]], 
                      columns=['id', 'sequence'])
print(corpus)
print("-----------------------------------------")

# Learning the sgt embeddings as vector for
# all sequences in a corpus.
# mode: 'default'
sgt = SGT(kappa=1, 
          flatten=True, 
          lengthsensitive=False, 
          mode='default')
sgt.fit_transform(corpus)
print(sgt.fit_transform(corpus))

# will it work on my sequences




full_df = pd.read_csv(Path("data/NAMP+AMP.csv"))


# y = full_df['AMP']
# encoder = LabelEncoder()
# encoder.fit(y)
# encoded_y = encoder.transform(y)

full_df['Seq'] = full_df['Seq'].map(list)
full_df = full_df.rename(columns={'Seq': 'sequence', 'ID': 'id'})
print((full_df[0:5]))


sgt_ = SGT(kappa=1, 
           lengthsensitive=False, 
           mode='default')
sgtembedding_df = sgt_.fit_transform(full_df)
X = sgtembedding_df.set_index('id')

print("--------------------------")
print(sgtembedding_df)
print("--------------------------")
print(X)
print("------------------")
print(y)
print(encoded_y)
