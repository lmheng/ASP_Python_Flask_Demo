import os
from flask import Flask, request, send_from_directory
import pickle
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.model_selection import train_test_split

app = Flask(__name__)


@app.route('/')
def index():
    return "Welcome!"


@app.route('/random_forest_classification', methods=['GET'])
def callModelOne():
    xValue = request.args.get('x')
    value = xValue.split(" ")

    dataset = pd.read_csv('C:/Users/LM/Desktop/pythonProject1/fetal_health.csv')
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    oversample = SMOTE()
    x, y = oversample.fit_resample(x, y)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True, random_state=42)

    sc = StandardScaler()
    sc.fit(X_train)

    value = sc.transform(value)

    modelOne = pickle.load(open('random_forest_classifier.pkl', 'rb')) # load model1 to the server
    return str(modelOne.predict(value)[0])


@app.route('/random_forest_regression', methods=['GET'])
def callModelTwo():
    xValue = request.args.get('x')
    value = xValue.split(" ")
    modelTwo = pickle.load(open('Random_Forest_Outcome.pkl', 'rb')) # load model2 to the server
    print(modelTwo.predict([value])[0])
    return str(modelTwo.predict([value])[0])


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
        'favicon.ico', mimetype='image/seed_logo.png')
