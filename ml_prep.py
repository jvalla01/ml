from pathlib import Path
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
import numpy as np
import sgt
print(sgt.__version__)
from sgt import SGT

def creating_AA_factors():
    """
    Create amino acid count factors to use to train the model
    Length is already there
        - Amino acid composition - get all amino acids from the sequence column as a list to string and then find the unique ones

    """

    full_df = pd.read_csv(Path("data/NAMP+AMP.csv"))

    AA_list = full_df["Seq"].values.tolist()
    AA_string = "".join(AA_list)
    unique_AA = list(set(AA_string))
    unique_AA.sort()
    print("The total number of unique amino acids found is:")
    print(len(unique_AA)) #24

    # Before going back and fixing

        #     ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        # - B is either asparagine or aspartic acid
        # - X is unknown
        # - U is selenocysteine
        # - Z is either glutamine or glutamic acid

    AA_list = list(AA_string)
    print("The number of occurences of amino acid B is:")
    print(AA_list.count("B")) # 4
    print("The number of occurrences of amino acid X is:")
    print(AA_list.count("X")) #116 
    print("The number of occurrences of amino acid U is:")
    print(AA_list.count("U")) # 3
    print("The number of occurrences of amino acid Z is:")
    print(AA_list.count("Z")) # 6

    #Note: now have removed all B, U and Z rows in python file AMP_files.py
    # Did it there so could readjust number of rows
    # Removing X would lose too much data
    # So will just not count up the X amino acids 

    # Tally up AA


    for c in unique_AA:
        full_df[c] = full_df["Seq"].str.count(c)

    
    full_df.to_csv(Path("data/NAMP+AMP.csv"), index = False)



    return unique_AA

#print(creating_factors())



def split_dataset(df):
    """
    80% training set 
    10% validation set to tune parameters 
    10% test set for the end
    """
    

    nrows = len(df) -1
    print(nrows)
    no_training_rows = round(nrows*0.8)

    if no_training_rows > nrows*0.8:
        print("no. of training rows rounded up")
    else:
        print("no. of training rows rounded down")
    print(no_training_rows)

    no_remaining_rows = nrows - no_training_rows

    if no_remaining_rows % 2 == 0:
        print("Even number of rows remaining for validation and test")
        no_validation = no_remaining_rows // 2
        no_test = no_remaining_rows //2
    else:
        (print("Odd number of rows remaining for validation and test"))
        no_validation = (no_remaining_rows //2) +1
        no_test = no_remaining_rows //2
    
    print(no_validation, no_test)
    print("Added up:")
    print(no_training_rows + no_test + no_validation)
        

    training_index1 = 0
    print(training_index1)
    training_index2 = no_training_rows-1
    print(training_index2)
    validation_index1 = training_index2+1
    print(validation_index1)
    validation_index2 = validation_index1 + no_validation -1
    print(validation_index2)
    test_index1 = validation_index2+1
    print(test_index1)
    test_index2 = test_index1 + no_test -1
    print(test_index2)

    training_df =  df.loc[0:training_index2]
    validation_df = df.loc[validation_index1:validation_index2]
    test_df = df.loc[test_index1:test_index2]

    print(training_df.shape, validation_df.shape, test_df.shape)

    training_df.to_csv(Path("data/training.csv"), index = False)
    validation_df.to_csv(Path("data/validation.csv"), index = False)
    test_df.to_csv(Path("data/test.csv"), index = False)
    
    return

    # 
    # - Other things i could look at notes
    # - Hydrophobicity with GRAVY score
    # - Charge
    # - "They are known to be cationic amphiphilic molecules but how are they different from other amphiphilic cationic molecules" - "https://pmc.ncbi.nlm.nih.gov/articles/PMC6513345/"
    # - embeddings https://github.com/sacdallago/bio_embeddings?tab=readme-ov-file
    # 

def embedding():


    training_df = pd.read_csv(Path("data/training.csv"))
    validation_df = pd.read_csv(Path("data/validation.csv"))
    test_df = pd.read_csv(Path("data/test.csv"))

    j = 0
    for df in training_df, validation_df, test_df:

        # extracts just the dependent variable and changes it from yes and no to numerical

        y = df['AMP']
        encoder = LabelEncoder()
        encoder.fit(y)
        y = encoder.transform(y)
        y = pd.Series(y)

        # turns the sequence into individual letters for encoding and changes names for the sgt function to understand

        df['Seq'] = df['Seq'].map(list)
        df = df.rename(columns={'Seq': 'sequence', 'ID': 'id'})

        #encodes the sequences


        sgt_ = SGT(kappa=1, 
                lengthsensitive=False, 
                mode='default')
        sgtembedding_df = sgt_.fit_transform(df)
        x = sgtembedding_df.set_index('id')

        #saves the datasets

        if j == 0:
            x.to_csv(Path("data/training_x.csv"), index = False)
            y.to_csv(Path("data/training_y.csv"), index = False)
            j+=1
        elif j ==1:
            x.to_csv(Path("data/validation_x.csv"), index = False)
            y.to_csv(Path("data/validation_y.csv"), index = False)
            j+=1
        else:
            x.to_csv(Path("data/test_x.csv"), index = False)
            y.to_csv(Path("data/test_y.csv"), index = False)

    return

#embedding()


