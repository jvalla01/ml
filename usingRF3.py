# using the model on positive amplify results from a tsv

from pathlib import Path
import pandas as pd
from Bio import SeqIO

from joblib import Parallel, delayed
import joblib
from pathlib import Path
import pandas as pd
import sgt
print(sgt.__version__)
from sgt import SGT
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import time

def tsv2table(tsv)->pd.DataFrame:
    """
    # Makes a blank dict
    # Take the concatenated fasta file
    # Make a table with the headers ID and sequence
    # Put each fasta file into the table
    # Clean empty rows, duplicates sequences with B and Z amino acids
    """

    ids=[]
    seqs=[]

    with open(tsv) as f:
        for line in f:
            id=line.split()[0]
            ids.append(id)
            seq=line.split()[1]
            seq=seq.replace("*", "") #remove stop codon
            seqs.append(seq)
    
    
    header2seq = {}

    for a, b in zip(ids, seqs):
         header2seq.update({a: b})

    #print(header2seq)

    series1 = pd.Series(header2seq)

    #print(series1)

    df1 = series1.to_frame()

    #print(df1)
    
    df1.to_csv(Path("data/temp.csv"))
    df2 = pd.read_csv(Path("data/temp.csv"), index_col=False, names=["ID", "Seq"])
    
    # remove empty rows
    # remove duplicate sequences
    
    df3 = df2.dropna()
    df3 = df3.drop_duplicates(subset=['Seq'])

    # remove B, U and Z amino acids

    df3 = df3[df3["Seq"].str.contains("B") == False]
    df3 = df3[df3["Seq"].str.contains("Z") == False]
    df3 = df3[df3["Seq"].str.contains("U") == False]
    df3 = df3[df3["Seq"].str.contains("X") == False]

    
    return df3



def embedding(df):

    df['Seq'] = df['Seq'].map(list)
    df = df.rename(columns={'Seq': 'sequence', 'ID': 'id'})

    # now embedding

    #import embedding model
    sgt = joblib.load('SGT.pkl')

    #use embedding model
    sgtembedding_df = sgt.fit_transform(df)

    #fix the feature headers to strings to match training labels
    sgtembedding_df.columns = [str(col) for col in sgtembedding_df.columns]
    x = sgtembedding_df.set_index('id')

    return x

def embedding2(df):
    """
    With new SGT model
    """

    df['Seq'] = df['Seq'].map(list)
    df = df.rename(columns={'Seq': 'sequence', 'ID': 'id'})

    # now embedding

    #import embedding model
    sgt = joblib.load('SGT_all.pkl')

    #use embedding model
    sgtembedding_df = sgt.fit_transform(df)
    sgtembedding_df.columns = [str(col) for col in sgtembedding_df.columns]
    x = sgtembedding_df.set_index('id')

    return x


def usingRF(x):
    df = x

    ids = df.index 

    #import model
    rf = joblib.load('RF.pkl')

    probability = rf.predict_proba(df)[:, 1] 
    classification = rf.predict(df)

    answers = pd.DataFrame({
    'id': ids,
    'probability': probability, 
    'classification': classification})

    # def predicting(value):
    #     if value < 0.75:
    #         return "non-AMP"
    #     elif value >= 0.75:
    #         return "AMP"
        
    # answers['Classification'] = answers['probability'].map(predicting)


    return answers


def usingRF2(x):
    """
    With new SGT model
    """
    df = x

    ids = df.index 

    #import model
    rf = joblib.load('RF_all.pkl')

    probability = rf.predict_proba(df)[:, 1] 
    classification = rf.predict(df)

    answers = pd.DataFrame({
    'id': ids,
    'probability': probability, 
    'classification': classification})

    # def predicting(value):
    #     if value < 0.75:
    #         return "non-AMP"
    #     elif value >= 0.75:
    #         return "AMP"
        
    # answers['Classification'] = answers['probability'].map(predicting)


    return answers


for file in Path("amplify_predicted_amps").glob("*.tsv"):
    print("----------------")
    start = time.time()
    print("next file")
    print(file)
    print("basename:", file.stem)

    table = tsv2table(file)
    print("table made")
    embeddings = embedding2(table)
    print("embeddings done")
    answers = usingRF2(embeddings)
    print("predictions done")


    print(Path(f"outputs/{file.stem}_predictions.csv"))
    answers.to_csv(Path(f"output/{file.stem}_predictions.csv"))
    print("done")
    end = time.time()
    elapsed = end-start
    print(f"Time elapsed: {elapsed} seconds")

"1 is a predicted AMP, 0 is not"