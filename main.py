from src.preprocess import load_and_preprocess
from src.model import build_model
from src.decision_engine import score_and_decide

df = load_and_preprocess("data/credit_data.csv")
model, X_test, y_test = build_model(df)
decisions, probs = score_and_decide(model, X_test)

for i in range(5):
    print(f"Applicant {i+1} - Prob: {probs[i]:.2f}, Decision: {decisions[i]}")
