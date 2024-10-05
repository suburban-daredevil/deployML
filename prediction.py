'''
Python file with the predict function,
to load the model from the file and make predictions
'''

import joblib

def predict(data):
    clf = joblib.load('rf_model.sav')
    return clf.predict(data)