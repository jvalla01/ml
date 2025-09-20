# practising using the model on a fresh whole fasta/faa file
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

def fasta2table(fasta= Path("data/Clayloam-Borders.24_proteins.faa"), outputcsv = Path("data/CL-BO.csv"))->pd.DataFrame:
    """
    # Makes a blank dict
    # Take the concatenated fasta file
    # Make a table with the headers ID and sequence
    # Put each fasta file into the table
    # Clean empty rows, duplicates sequences with B and Z amino acids
    """
    header2seq = {}

    with open(fasta) as f:
        headers = []
        seqs = []
        for record in SeqIO.parse(f, "fasta"):
            headers.append(record.id)
            sequence = record.seq
            sequence = sequence.replace("*", "") # remove stop codons
            seqs.append(sequence)

    for a, b in zip(headers, seqs):
        header2seq.update({a: b})

    # print(header2seq)

    series1 = pd.Series(header2seq)

    #print(series1)

    df1 = series1.to_frame()

    #print(df1.loc["dbAMP_12356"])
    #print(df1[])
    
    df1.to_csv(Path("data/AMP.csv"))
    df2 = pd.read_csv(Path("data/AMP.csv"), index_col=False, names=["ID", "Seq"])
    
    #remove empty rows
    # remove duplicate sequences
    
    df3 = df2.dropna()
    df3 = df3.drop_duplicates(subset=['Seq'])

    #remove B, U and Z amino acids

    df3 = df3[df3["Seq"].str.contains("B") == False]
    df3 = df3[df3["Seq"].str.contains("Z") == False]
    df3 = df3[df3["Seq"].str.contains("U") == False]
    

    df3.to_csv(outputcsv, index = False)

    
    return df3


#fasta2table()

def embedding(inputcsv = Path("data/CL-BO.csv"), outputcsv = Path("data/CL-BO_2.csv")):
    df = pd.read_csv(inputcsv)

    df['Seq'] = df['Seq'].map(list)
    df = df.rename(columns={'Seq': 'sequence', 'ID': 'id'})

    # now embedding

    #import embedding model
    sgt = joblib.load('SGT.pkl')

    #use embedding model
    sgtembedding_df = sgt.fit_transform(df)
    x = sgtembedding_df.set_index('id')
    x.to_csv(outputcsv, index = True)

    return x

#print(embedding())

def usingRF(inputcsv=Path("data/CL-BO_2.csv"), outputcsv = Path("data/CL-BO_3.csv")):
    df = pd.read_csv(inputcsv)
    ids = df.pop('id')
    print(df)
    print(ids)

    #import model
    rf = joblib.load('RF.pkl')

    preds = rf.predict_proba(df)[:, 1] 

    answers = pd.DataFrame({
    'id': ids,
    'prediction': preds})

    

    def predicting(value):
        if value < 0.75:
            return "non-AMP"
        elif value >= 0.75:
            return "AMP"
        
    answers['Classification'] = answers['prediction'].map(predicting)

    answers.to_csv(outputcsv, index = False)


    return answers



#print(usingRF())

def usingRF_nh(inputcsv=Path("data/CL-BO_2.csv"), outputcsv = Path("data/CL-BO_3_nh.csv")):
    df = pd.read_csv(inputcsv)
    ids = df.pop('id')
    print(df)
    print(ids)

    #import model
    rf = joblib.load('RF_nh.pkl')

    preds = rf.predict_proba(df)[:, 1] 

    answers = pd.DataFrame({
    'id': ids,
    'prediction': preds})

    

    def predicting(value):
        if value < 0.75:
            return "non-AMP"
        elif value >= 0.75:
            return "AMP"
        
    answers['Classification'] = answers['prediction'].map(predicting)

    answers.to_csv(outputcsv, index = False)


    return answers

usingRF_nh()