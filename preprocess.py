# preprocess.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(file_path): 
    df = pd.read_csv(file_path)
    df_model = df[['Medu', 'Fedu', 'goout', 'Walc', 'failures', 'studytime', 'absences', 
                   'freetime', 'health', 'Dalc', 'famrel', 'romantic', 'G1', 'G3']]
    df_model['romantic'] = df_model['romantic'].map({'yes': 1, 'no': 0})
    X = df_model.drop(columns=['G3'])
    y = df_model['G3']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y
