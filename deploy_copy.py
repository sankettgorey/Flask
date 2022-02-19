import pickle
from flask import Flask, request
import numpy as np
import pandas as pd

app = Flask(__name__)

# importing the pickle file
pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return 'welcome to new app'


@app.route('/predict')
def prediction():
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    kurtosis = request.args.get('kurtosis')
    entropy = request.args.get('entropy')
    x = classifier.predict([[variance, skewness, kurtosis, entropy]])
    return 'Th give note is: ' + str(x)






if __name__ == '__main__':
    app.run()