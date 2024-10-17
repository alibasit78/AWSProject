import pandas as pd
from fraud_detection.logger import logging
from fraud_detection import SEED
from sklearn.preprocessing import StandardScaler

def sample_the_dataset(df):
    #sample neg samples from the dataframe
    pos_samples = df[df["Class"] == 1]
    neg_samples = df.sample(n = 3*pos_samples.shape[0], random_state=SEED)
    df = pd.concat([pos_samples, neg_samples], axis = 0)
    df = df.sample(frac=1)
    print(df.columns)
    
    #removed class columns from the dataframe
    labels = df['Class']
    df.drop(columns=['Class'], axis=1, inplace=True)
    logging.info(f"df shape: {df.shape}")
    return df, labels

class PreprocessMethod:
    def __init__(self, columns):
        self.ss = StandardScaler()
        self.ss.fit(columns)
    def transform(self, columns):
        return self.ss.transform(columns)
    