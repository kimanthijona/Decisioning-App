import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

filepath = "D:/Credit_Decisioning/data/credit_data_10000.csv"

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    le = LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])
    
    scaler = StandardScaler()
    df[['income', 'loan_amount', 'term']] = scaler.fit_transform(df[['income', 'loan_amount', 'term']])
    
    return df

