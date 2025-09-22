# ml
Model summary:

Best -
SGT_all.pkl = The SGT model trained on 10000 uniprot sequences 
RF_all.pkl = The RF model trained on data embedded with SGT_all.pkl

Other -
SGT.pkl = The SGT model trained on training set population data
RF_all.pkl = The RF model trained on data embedded with SGT.pkl
RF_nh.pkl = The RF model trained on data embedded with SGT.pkl and without hyperparameter tuning



1. AMP_files.py 
    - to create a dataset with a balanced mixture of AMP and non-AMP sequences
2. ml_prep.py
    - to split the data
3. embedding_on_all_reviewed_uniprot_proteins.py
    - create embeddings
    # a large uniprot file needed in this script was too large to commit to Github so any uniprot file with 10000 random protein sequences is fine
3. randomforest_withnewSGTmodel.py
    - to train random forest model 
4. usingRF
    - to classify a single sequence
5. usingRF2
    - run models on a sample (high quality) bin
6. usingRF3
    - run models on amplify predicted AMP sequences

data
    - the files needed to train and test the models 

amplify_predicted_amps
    - all of the predicted AMPs (by AMPlify) from each metagenomic bin 

output
    - the output from usingRF2

highqualitybins
    - stores the fasta files for untranslated bins that were of high quality
    - this folder is not used for the model
