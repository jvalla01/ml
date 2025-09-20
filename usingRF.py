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

#import models
rf = joblib.load('RF.pkl')
sgt = joblib.load('SGT.pkl')

#X_test = pd.read_csv(Path("data/test_x.csv"))
#y_test = pd.read_csv(Path("data/test_y.csv"))
#preds = rf.predict(X_test)
X_train = pd.read_csv("data/training_x.csv")  # to get the original column names

#embed sequences
sequence = "PTAVLAFLADGESWSSSALALSLGTSQRTVQRALDSLGAAGKVQSFGRGRARRWMTPPVPGFATTLLLPAPLPID"
seqlist = list(sequence)
#print(seqlist)


embedded_seq=sgt.fit(seqlist)
embedded_seq=pd.DataFrame(embedded_seq)

embedded_seq = embedded_seq.T
embedded_seq.columns = X_train.columns
print(embedded_seq)
#use rf.predict
prediction = rf.predict(embedded_seq)

if prediction ==1:
    print("This is a predicted AMP")
elif prediction == 0:
    print("This is not a predicted AMP")
else:
    print("Something has gone wrong")