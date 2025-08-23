#source ~/venv/myenv/bin/activate
from pathlib import Path
import pandas as pd
from Bio import SeqIO
from sklearn.utils import shuffle


def concat_files(archaea = Path("dbAMP_archaea_2024.fasta"), bact = Path("dbAMP_bacteria_2024.fasta"))->None:
    """
    # merge the bacteria and archaea fasta files from DBAMP into one fasta file
    """
    archaea = Path("dbAMP_archaea_2024.fasta")
    bact = Path("dbAMP_bacteria_2024.fasta")
    both = Path("AMP.fasta")

    data = data2 = "";

    # Reading data from file1
    with open(archaea) as f:
        data = f.read()

        # Reading data from file2
    with open(bact) as f:
        data2 = f.read()

    # Merging 2 files
    # To add the data of file2
    # from next line
        data += "\n"
        data += data2

    with open (both, 'w') as f:
       f.write(data)

    return

#concat_files()

def fasta2table(fasta= Path("AMP.fasta"))->pd.DataFrame:
    """
    # Makes a blank dict
    # Take the concatenated fasta file
    # Make a table with the headers ID and sequence
    # Put each fasta file into the table
    """
    fasta= Path("AMP.fasta")
    ouputcsv = Path("AMP.csv")
    header2seq = {}

    with open(fasta) as f:
        headers = []
        seqs = []
        for record in SeqIO.parse(f, "fasta"):
            headers.append(record.id)
            seqs.append(record.seq)

    for a, b in zip(headers, seqs):
        header2seq.update({a: b})

    # print(header2seq)

    series1 = pd.Series(header2seq)

    #print(series1)

    df1 = series1.to_frame()

    #print(df1.loc["dbAMP_12356"])
    #print(df1[])
    
    df1.to_csv("AMP.csv")
    df2 = pd.read_csv("AMP.csv", index_col=False, names=["ID", "Seq"])
    df3 = df2.dropna()

    df3.to_csv("AMP.csv", index = False)

    
    return df3


#fasta2table()


def seqlength(df3 = Path("AMP.csv")):
    """
    # Get the sequence lengths
    # Add a column with sequence lengths
    # Create summary statistics
    """

    df3 = pd.read_csv(Path("AMP.csv"))

    df3["Length"] = df3["Seq"].str.len()

    df3.to_csv("AMP.csv", index = False)
    
    return df3.describe()



#print(seqlength())

def get_uniprot_samples(df4=Path("uniprotkb_reviewed_true_AND_existence_1_2025_08_21.tsv")):

    df4 = pd.read_csv(Path("uniprotkb_reviewed_true_AND_existence_1_2025_08_21.tsv"),sep = '\t')

    #filter out uncharacterized protein

    # filter the rows that contain the substring
    substring = 'Uncharacterized protein'
    filter = df4['Protein names'].str.contains(substring)
    filtered_df = df4[~filter]

    df5 = filtered_df.head(781)

    df5.to_csv("nonAMP.csv", index = False)


    return df5.describe()

#print(get_uniprot_samples())

# AMP dataset 
#            Length
# count  781.000000
# mean    56.793854
# std     60.582703
# min      2.000000
# 25%     20.000000
# 50%     39.000000
# 75%     62.000000
# max    255.000000

# 781 shortest length uniprot proteins matching criteria: 
# UniProtKB, reviewed yes AND protein existence - evidence at protein level AND taxonomy 1869227 OR taxonomy 2157 NOT Keyword KW-0929 NOT Keyword KW-0044 AND Sequence length 1 to 255
#           Length
# count  781.000000
# mean    87.871959
# std     30.720518
# min     10.000000
# 25%     66.000000
# 50%     92.000000
# 75%    114.000000
# max    133.000000
 


def merge_files():
    """
    read both files
    find out how to merge them together
    merge them based on the correct columns
    """

    #read in csvs, tidy column names and add column for whether it is an AMP

    AMP_df = pd.read_csv(Path("AMP.csv"))
    AMP_df["AMP"]="Yes"
    NAMP_df = pd.read_csv(Path("nonAMP.csv"), usecols=["Entry", "Sequence", "Length"])
    NAMP_df = NAMP_df.rename(columns={'Entry': 'ID', 'Sequence': 'Seq'})
    NAMP_df["AMP"]="No"

    # join them together

    both = [AMP_df, NAMP_df]
    mixed_df = pd.concat(both)

    # shuffle rows

    mixed_df = shuffle(mixed_df)

    # save as csv

    mixed_df.to_csv("NAMP+AMP.csv", index = False)



    return


merge_files()