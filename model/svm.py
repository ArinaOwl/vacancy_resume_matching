from joblib import dump, load
from numpy import exp

DATA_PATH = './trained_models/'

def logistic(x):
    return 1 / (1 + exp(-x))

def predict_proba(model, data):
    return logistic(model.decision_function(data))

model = load(DATA_PATH + 'svm_v1.joblib')
