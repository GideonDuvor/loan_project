import pandas as pd

def preprocess_data(df):
    try:
        # Drop Loan_ID (not useful)
        df = df.drop('Loan_ID', axis=1)
        

        # Handle missing values
        df = df.ffill()
        df = df.fillna(0)

        # Convert categorical to numeric
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
        df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
        df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
        df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
        df['Loan_Approved'] = df['Loan_Approved'].map({'Y': 1, 'N': 0})

        # One-hot encoding
        df = pd.get_dummies(df, columns=['Property_Area', 'Dependents'], drop_first=True)

        # Split features and target
        X = df.drop('Loan_Approved', axis=1)
        y = df['Loan_Approved']

        print("Preprocessing complete")
        return X, y

    except Exception as e:
        print(f"Preprocessing error: {e}")