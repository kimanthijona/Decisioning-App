def decision_rule(prob, threshold=0.5):
    if prob >= threshold:
        return "Decline"
    elif prob >= 0.3:
        return "Review"
    else:
        return "Approve"

def score_and_decide(model, X):
    probs = model.predict_proba(X)[:, 1]
    decisions = [decision_rule(p) for p in probs]
    return decisions, probs
