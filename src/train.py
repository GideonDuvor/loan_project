from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

def train_model(X, y):
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LogisticRegression(max_iter=2000)
        model.fit(X_train, y_train)

        print("Model trained successfully")
        return model, X_test, y_test

    except Exception as e:
        print(f"Training error: {e}")


def save_model(model, filename='models/model.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)