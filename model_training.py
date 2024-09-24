from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import joblib
import os

def train_loan_eligibility_model(df, column):
    model_path = "D:\\Codes\\Complete_project_macbook\\trained_model.pkl"
    # Checking if model already exists or not
    if os.path.exists(model_path):
        print("Loading the existing trained model...")
        model = joblib.load(model_path)
        return model
    
    # Ensure target variable is categorical
    if df[column].dtype in ['float64', 'int64']:
        print("Converting target variable to categorical")
        df[column] = pd.cut(df[column], bins=2, labels=False)  # Adjust bins as needed

    X = df.drop(columns=[column])
    y = df[column]

    print(f"Shape of X before split: {X.shape}")
    print(f"Shape of y before split: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=1000, random_state=42)
    model.fit(X_train, y_train)
    
    joblib.dump(model, model_path)
    print(f"Model saved at {model_path}")

    y_pred = model.predict(X_test)
    print("Loan eligibility model accuracy: ", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return model
