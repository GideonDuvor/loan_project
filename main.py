from src.loan_data import load_data
from src.preprocessing import preprocess_data
from src.train import train_model, save_model

print("STARTING LOAN MODEL")

df = load_data('data/credit.csv')

X, y = preprocess_data(df)

model, X_test, y_test = train_model(X, y)

save_model(model)

print("Loan model pipeline executed successfully!")