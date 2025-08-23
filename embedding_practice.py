import sgt
print(sgt.__version__)
from sgt import SGT

import numpy as np
import pandas as pd

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

