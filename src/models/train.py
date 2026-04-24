import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# feature columns
FEATURE_COLS = [
    'Victim Age', 'Victim Sex_Enc', 'Victim Descent_Enc',
    'Hour Occurred', 'DayOfWeek', 'Month', 'IsWeekend', 'TimeBin_Enc',
    'Area ID', 'Area_Crime_Frequency',
    'Premise Code', 'Premise_Crime_Frequency',
    'Armed', 'Report_Delay_Days', 'MO_Code_Count',
    'Latitude', 'Longitude', 'Year', 'Quarter'
]
TARGET_COL = 'Crime_Category'

# function to prepare train/test split
def prepare_data(df):
    model_df = df[FEATURE_COLS + [TARGET_COL]].dropna()
    X = model_df[FEATURE_COLS]
    y = model_df[TARGET_COL]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# function to train models
def train_logistic_regression(X_train, y_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(X_scaled, y_train)
    return model, scaler

# function to train models
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100, max_depth=15,
        class_weight='balanced', n_jobs=-1, random_state=42
    )
    model.fit(X_train, y_train)
    return model

# function to train models
def train_xgboost_with_smote(X_train, y_train):
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_sm, y_sm = smote.fit_resample(X_train, y_train)

    le = LabelEncoder()
    y_enc = le.fit_transform(y_sm)

    model = XGBClassifier(
        n_estimators=200, max_depth=8,
        learning_rate=0.1, n_jobs=-1, random_state=42
    )
    model.fit(X_sm, y_enc)
    return model, le
